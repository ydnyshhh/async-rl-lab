from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Protocol

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.metrics import MetricsSnapshot, MetricsStore
from async_rl_lab.models import Action, GenerationRequest, GenerationResult, PolicyRef, ToolCall
from async_rl_lab.policy_store import LocalPolicyStore, SequencePolicyState


class InferenceEngine(Protocol):
    async def start(self) -> None:
        ...

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        ...

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        ...

    async def refresh_policy(self, policy: PolicyRef) -> None:
        ...

    def current_policy(self) -> PolicyRef:
        ...

    def metrics_snapshot(self) -> MetricsSnapshot:
        ...

    async def close(self) -> None:
        ...


@dataclass(slots=True)
class PendingGeneration:
    request: GenerationRequest
    future: asyncio.Future[GenerationResult]
    enqueue_ts: float


def parse_action_text(raw_text: str) -> Action:
    parsed_ts = utc_ts()
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        parts = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(parts).strip()

    try:
        payload = json.loads(cleaned)
        action_type = str(payload.get("type", "invalid"))
        if action_type == "tool_call":
            tool_call = ToolCall(
                tool_call_id=make_id("toolcall"),
                tool_name=str(payload["tool_name"]),
                arguments=dict(payload.get("arguments", {})),
                created_ts=parsed_ts,
                raw_text_span=cleaned,
            )
            return Action(
                action_id=make_id("action"),
                action_type="tool_call",
                raw_text=raw_text,
                parsed_ts=parsed_ts,
                parser_status="ok",
                tool_call=tool_call,
            )
        if action_type in {"finish", "respond"}:
            return Action(
                action_id=make_id("action"),
                action_type="finish" if action_type == "finish" else "respond",
                raw_text=raw_text,
                parsed_ts=parsed_ts,
                parser_status="ok",
                final_text=str(payload.get("answer") or payload.get("text") or ""),
            )
    except Exception:
        pass

    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        recovered = cleaned[brace_start : brace_end + 1]
        try:
            payload = json.loads(recovered)
            if payload.get("type") == "finish":
                return Action(
                    action_id=make_id("action"),
                    action_type="finish",
                    raw_text=raw_text,
                    parsed_ts=parsed_ts,
                    parser_status="recovered",
                    final_text=str(payload.get("answer", "")),
                )
        except Exception:
            pass

    return Action(
        action_id=make_id("action"),
        action_type="invalid",
        raw_text=raw_text,
        parsed_ts=parsed_ts,
        parser_status="malformed",
        metadata={"reason": "json_parse_failed"},
    )


class MockInferenceEngine:
    def __init__(
        self,
        initial_policy: PolicyRef,
        *,
        policy_store: LocalPolicyStore | None = None,
        batch_window_ms: float = 10.0,
        max_batch_size: int = 16,
        artificial_latency_ms: float = 25.0,
    ) -> None:
        self.loaded_policy = initial_policy
        self.policy_store = policy_store
        self.loaded_state: SequencePolicyState | None = None
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.artificial_latency_ms = artificial_latency_ms
        self.pending_queue: asyncio.Queue[PendingGeneration] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.worker_task: asyncio.Task[None] | None = None
        self.metrics = MetricsStore()
        self.last_refresh_latency_ms = 0.0

    async def start(self) -> None:
        await self.ensure_policy_loaded(self.loaded_policy)
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self.batch_worker())

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self.worker_task is None:
            await self.start()
        future: asyncio.Future[GenerationResult] = asyncio.get_running_loop().create_future()
        self.metrics.increment("inference.requests")
        self.metrics.observe("inference.pending_queue_depth", float(self.pending_queue.qsize() + 1))
        await self.pending_queue.put(PendingGeneration(request=request, future=future, enqueue_ts=utc_ts()))
        return await future

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results = await asyncio.gather(*(self.submit(request) for request in requests))
        return list(results)

    async def refresh_policy(self, policy: PolicyRef) -> None:
        self.last_refresh_latency_ms = await self.ensure_policy_loaded(policy)

    def current_policy(self) -> PolicyRef:
        return self.loaded_policy

    def metrics_snapshot(self) -> MetricsSnapshot:
        return self.metrics.snapshot()

    async def close(self) -> None:
        self.stop_event.set()
        if self.worker_task is not None:
            await self.worker_task
            self.worker_task = None
        self.pending_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()

    async def batch_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                first_item = await asyncio.wait_for(self.pending_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            batch = [first_item]
            await asyncio.sleep(self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size and not self.pending_queue.empty():
                batch.append(self.pending_queue.get_nowait())

            for policy_batch in self.partition_batch_by_policy(batch):
                await self.process_batch(policy_batch)

    async def process_batch(self, batch: list[PendingGeneration]) -> None:
        dispatch_ts = utc_ts()
        policy_ref = batch[0].request.policy
        policy_load_latency_ms = await self.ensure_policy_loaded(policy_ref)
        started_ts = utc_ts()
        self.metrics.increment("inference.batches")
        self.metrics.observe("inference.batch_size", float(len(batch)))
        await asyncio.sleep(self.artificial_latency_ms / 1000.0)
        for pending in batch:
            generated_text = self.generate_text(pending.request)
            action = parse_action_text(generated_text)
            score = self.score_generated_text(generated_text)
            ended_ts = utc_ts()
            prompt_tokens = max(1, len(pending.request.observation_text.split()))
            completion_tokens = max(1, len(score.token_logprobs) or len(generated_text.split()))
            queue_wait_ms = (dispatch_ts - pending.enqueue_ts) * 1000.0
            latency_ms = (ended_ts - dispatch_ts) * 1000.0
            self.metrics.observe("inference.queue_wait_ms", queue_wait_ms)
            self.metrics.observe("inference.latency_ms", latency_ms)
            self.metrics.observe("inference.prompt_tokens", float(prompt_tokens))
            self.metrics.observe("inference.completion_tokens", float(completion_tokens))
            self.metrics.observe(
                "inference.tokens_per_second",
                (prompt_tokens + completion_tokens) / max(ended_ts - started_ts, 1e-6),
            )
            if action.action_type == "invalid":
                self.metrics.increment("inference.invalid_action_count")
            result = GenerationResult(
                result_id=make_id("gen"),
                request_id=pending.request.request_id,
                task_id=pending.request.task_id,
                prompt_id=pending.request.prompt_id,
                group_id=pending.request.group_id,
                sample_index_within_group=pending.request.sample_index_within_group,
                actor_id=pending.request.actor_id,
                policy=pending.request.policy,
                generated_text=generated_text,
                finish_reason="tool_call" if action.action_type == "tool_call" else "stop",
                created_ts=pending.request.created_ts,
                started_ts=started_ts,
                ended_ts=ended_ts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                parser_status=action.parser_status,
                queue_wait_ms=queue_wait_ms,
                latency_ms=latency_ms,
                action=action,
                token_logprobs=score.token_logprobs,
                metadata={
                    "engine_policy_version": self.loaded_policy.policy_version,
                    "served_policy_version": policy_ref.policy_version,
                    "policy_load_latency_ms": policy_load_latency_ms,
                    "batch_size": len(batch),
                },
            )
            if not pending.future.done():
                pending.future.set_result(result)

    def generate_text(self, request: GenerationRequest) -> str:
        if "forced_output" in request.metadata:
            return str(request.metadata["forced_output"])

        candidates = self.build_candidates(request)
        if len(candidates) == 1:
            return candidates[0]
        if self.policy_store is None or self.loaded_state is None:
            return candidates[0]

        scores = [self.policy_store.score_text(candidate, state=self.loaded_state) for candidate in candidates]
        score_values = [score.mean_logprob for score in scores]
        temperature = max(request.temperature, 1e-4)
        if temperature <= 1e-3:
            best_index = max(range(len(candidates)), key=score_values.__getitem__)
            return candidates[best_index]

        weights = softmax_scores(score_values, temperature=temperature)
        seed = request.seed if request.seed is not None else stable_request_seed(request)
        random_state = random.Random(seed)
        selected_index = random_state.choices(range(len(candidates)), weights=weights, k=1)[0]
        return candidates[selected_index]

    def build_candidates(self, request: GenerationRequest) -> list[str]:
        gold_answer = request.metadata.get("gold_answer")
        expression = request.metadata.get("expression")
        candidates: list[str] = []
        if expression and "calculator" in request.available_tools:
            candidates.append(
                json.dumps(
                    {
                        "type": "tool_call",
                        "tool_name": "calculator",
                        "arguments": {"expression": str(expression)},
                    }
                )
            )
        if gold_answer is not None:
            candidates.append(json.dumps({"type": "finish", "answer": str(gold_answer)}))
            candidates.append(json.dumps({"type": "finish", "answer": f"{gold_answer}?"}))
            candidates.append(json.dumps({"type": "finish", "answer": request.observation_text[:16]}))
        else:
            candidates.append(json.dumps({"type": "finish", "answer": request.observation_text[:32]}))
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def score_generated_text(self, generated_text: str):
        if self.policy_store is None or self.loaded_state is None:
            token_count = max(1, len(generated_text.split()))
            return type("FallbackScore", (), {"token_logprobs": tuple(-1.0 for _ in range(token_count))})()
        return self.policy_store.score_text(generated_text, state=self.loaded_state)

    async def ensure_policy_loaded(self, policy: PolicyRef) -> float:
        if self.loaded_policy.policy_version == policy.policy_version and self.loaded_state is not None:
            return 0.0
        started_ts = utc_ts()
        await asyncio.sleep(self.artificial_latency_ms / 1000.0)
        if self.policy_store is not None:
            self.loaded_state = self.policy_store.load_policy_state(policy.policy_version)
        self.loaded_policy = policy
        ended_ts = utc_ts()
        latency_ms = (ended_ts - started_ts) * 1000.0
        self.metrics.increment("inference.policy_refresh_count")
        self.metrics.observe("inference.policy_refresh_latency_ms", latency_ms)
        return latency_ms

    def partition_batch_by_policy(self, batch: list[PendingGeneration]) -> list[list[PendingGeneration]]:
        buckets: dict[int, list[PendingGeneration]] = defaultdict(list)
        order: list[int] = []
        for pending in batch:
            policy_version = pending.request.policy.policy_version
            if policy_version not in buckets:
                order.append(policy_version)
            buckets[policy_version].append(pending)
        return [buckets[policy_version] for policy_version in order]


class HFInferenceEngine:
    def __init__(
        self,
        initial_policy: PolicyRef,
        *,
        model_name_or_path: str | None = None,
        device: str = "cpu",
        batch_window_ms: float = 10.0,
        max_batch_size: int = 8,
        generation_defaults: dict[str, object] | None = None,
    ) -> None:
        self.loaded_policy = initial_policy
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.generation_defaults = generation_defaults or {}
        self.metrics = MetricsStore()
        self.pending_queue: asyncio.Queue[PendingGeneration] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.worker_task: asyncio.Task[None] | None = None
        self.executor: ThreadPoolExecutor | None = self.create_executor()
        self.loaded_model = None
        self.loaded_tokenizer = None
        self.loaded_model_source: str | None = None
        self.last_refresh_latency_ms = 0.0

    async def start(self) -> None:
        await self.ensure_policy_loaded(self.loaded_policy)
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self.batch_worker())

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self.worker_task is None:
            await self.start()
        future: asyncio.Future[GenerationResult] = asyncio.get_running_loop().create_future()
        self.metrics.increment("inference.requests")
        self.metrics.observe("inference.pending_queue_depth", float(self.pending_queue.qsize() + 1))
        await self.pending_queue.put(PendingGeneration(request=request, future=future, enqueue_ts=utc_ts()))
        return await future

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results = await asyncio.gather(*(self.submit(request) for request in requests))
        return list(results)

    async def refresh_policy(self, policy: PolicyRef) -> None:
        self.last_refresh_latency_ms = await self.ensure_policy_loaded(policy)

    def current_policy(self) -> PolicyRef:
        return self.loaded_policy

    def metrics_snapshot(self) -> MetricsSnapshot:
        return self.metrics.snapshot()

    async def close(self) -> None:
        self.stop_event.set()
        if self.worker_task is not None:
            await self.worker_task
            self.worker_task = None
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.pending_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.loaded_model = None
        self.loaded_tokenizer = None

    async def batch_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                first_item = await asyncio.wait_for(self.pending_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue

            batch = [first_item]
            await asyncio.sleep(self.batch_window_ms / 1000.0)
            while len(batch) < self.max_batch_size and not self.pending_queue.empty():
                batch.append(self.pending_queue.get_nowait())

            for compatible_batch in self.partition_batch_by_signature(batch):
                await self.process_batch(compatible_batch)

    async def process_batch(self, batch: list[PendingGeneration]) -> None:
        dispatch_ts = utc_ts()
        policy_ref = batch[0].request.policy
        policy_load_latency_ms = await self.ensure_policy_loaded(policy_ref)
        started_ts = utc_ts()
        self.metrics.increment("inference.batches")
        self.metrics.observe("inference.batch_size", float(len(batch)))
        prompt_texts = [self.render_prompt(pending.request) for pending in batch]
        outputs = await asyncio.get_running_loop().run_in_executor(
            self.ensure_executor(),
            self.generate_batch_sync,
            prompt_texts,
            batch,
        )
        for pending, output in zip(batch, outputs):
            generated_text = str(output["generated_text"])
            action = parse_action_text(generated_text)
            ended_ts = utc_ts()
            queue_wait_ms = (dispatch_ts - pending.enqueue_ts) * 1000.0
            latency_ms = (ended_ts - dispatch_ts) * 1000.0
            prompt_tokens = int(output["prompt_tokens"])
            completion_tokens = int(output["completion_tokens"])
            self.metrics.observe("inference.queue_wait_ms", queue_wait_ms)
            self.metrics.observe("inference.latency_ms", latency_ms)
            self.metrics.observe("inference.prompt_tokens", float(prompt_tokens))
            self.metrics.observe("inference.completion_tokens", float(completion_tokens))
            self.metrics.observe(
                "inference.tokens_per_second",
                (prompt_tokens + completion_tokens) / max(ended_ts - started_ts, 1e-6),
            )
            if action.action_type == "invalid":
                self.metrics.increment("inference.invalid_action_count")
            result = GenerationResult(
                result_id=make_id("gen"),
                request_id=pending.request.request_id,
                task_id=pending.request.task_id,
                prompt_id=pending.request.prompt_id,
                group_id=pending.request.group_id,
                sample_index_within_group=pending.request.sample_index_within_group,
                actor_id=pending.request.actor_id,
                policy=pending.request.policy,
                generated_text=generated_text,
                finish_reason="tool_call" if action.action_type == "tool_call" else "stop",
                created_ts=pending.request.created_ts,
                started_ts=started_ts,
                ended_ts=ended_ts,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                parser_status=action.parser_status,
                queue_wait_ms=queue_wait_ms,
                latency_ms=latency_ms,
                action=action,
                token_logprobs=tuple(output.get("token_logprobs", ())),
                metadata={
                    "engine_policy_version": self.loaded_policy.policy_version,
                    "served_policy_version": pending.request.policy.policy_version,
                    "policy_load_latency_ms": policy_load_latency_ms,
                    "batch_size": len(batch),
                    "backend": "hf",
                },
            )
            if not pending.future.done():
                pending.future.set_result(result)

    async def ensure_policy_loaded(self, policy: PolicyRef) -> float:
        model_source = self.resolve_model_source(policy)
        if self.loaded_policy.policy_version == policy.policy_version and self.loaded_model_source == model_source:
            return 0.0
        started_ts = utc_ts()
        bundle = await asyncio.get_running_loop().run_in_executor(
            self.ensure_executor(),
            load_hf_backend_bundle,
            model_source,
            self.device,
        )
        self.loaded_tokenizer = bundle["tokenizer"]
        self.loaded_model = bundle["model"]
        self.loaded_model_source = model_source
        self.loaded_policy = policy
        ended_ts = utc_ts()
        latency_ms = (ended_ts - started_ts) * 1000.0
        self.metrics.increment("inference.policy_refresh_count")
        self.metrics.observe("inference.policy_refresh_latency_ms", latency_ms)
        return latency_ms

    def resolve_model_source(self, policy: PolicyRef) -> str:
        if isinstance(policy.metadata.get("hf_model_name_or_path"), str):
            return str(policy.metadata["hf_model_name_or_path"])
        if policy.policy_path and Path(policy.policy_path).exists() and (Path(policy.policy_path) / "config.json").exists():
            return policy.policy_path
        if self.model_name_or_path is not None:
            return self.model_name_or_path
        raise RuntimeError(
            "HFInferenceEngine requires model_name_or_path or a policy metadata/path entry that resolves to a local HF model."
        )

    def render_prompt(self, request: GenerationRequest) -> str:
        if request.available_tools:
            tool_header = ", ".join(request.available_tools)
            return f"Tools: {tool_header}\n\n{request.observation_text}"
        return request.observation_text

    def generate_batch_sync(self, prompt_texts: list[str], batch: list[PendingGeneration]) -> list[dict[str, object]]:
        if self.loaded_model is None or self.loaded_tokenizer is None:
            raise RuntimeError("HFInferenceEngine model not loaded before generation")
        tokenizer = self.loaded_tokenizer
        model = self.loaded_model
        torch = load_torch_module()
        request = batch[0].request
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        generation_kwargs = {
            **self.generation_defaults,
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        }
        do_sample = request.temperature > 1e-3
        if do_sample:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = request.top_p
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)
        input_width = int(inputs["input_ids"].shape[1])
        completion_ids = outputs[:, input_width:]
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        results: list[dict[str, object]] = []
        for index, generated_text in enumerate(decoded):
            completion_tokens = sequence_length(completion_ids[index])
            prompt_tokens = sequence_length(inputs["attention_mask"][index])
            results.append(
                {
                    "generated_text": generated_text.strip(),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "token_logprobs": (),
                }
            )
        return results

    def partition_batch_by_signature(self, batch: list[PendingGeneration]) -> list[list[PendingGeneration]]:
        buckets: dict[tuple[object, ...], list[PendingGeneration]] = defaultdict(list)
        order: list[tuple[object, ...]] = []
        for pending in batch:
            signature = (
                pending.request.policy.policy_version,
                pending.request.max_new_tokens,
                round(pending.request.temperature, 6),
                round(pending.request.top_p, 6),
                tuple(pending.request.stop_sequences),
            )
            if signature not in buckets:
                order.append(signature)
            buckets[signature].append(pending)
        return [buckets[signature] for signature in order]

    def create_executor(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-generate")

    def ensure_executor(self) -> ThreadPoolExecutor:
        if self.executor is None:
            self.executor = self.create_executor()
        return self.executor


class VLLMInferenceEngine:
    def __init__(self, initial_policy: PolicyRef) -> None:
        self.loaded_policy = initial_policy
        self.metrics = MetricsStore()

    async def start(self) -> None:
        return None

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError("VLLMInferenceEngine is a planned backend and not part of v1.")

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [await self.submit(request) for request in requests]

    async def refresh_policy(self, policy: PolicyRef) -> None:
        self.loaded_policy = policy

    def current_policy(self) -> PolicyRef:
        return self.loaded_policy

    def metrics_snapshot(self) -> MetricsSnapshot:
        return self.metrics.snapshot()

    async def close(self) -> None:
        return None


def softmax_scores(values: Sequence[float], *, temperature: float) -> list[float]:
    if not values:
        return []
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def stable_request_seed(request: GenerationRequest) -> int:
    return sum(ord(char) for char in request.request_id) + (request.policy.policy_version * 1_009)


def load_hf_backend_bundle(model_source: str, device: str) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "HFInferenceEngine requires transformers to be installed. Add the optional hf dependencies and sync the environment."
        ) from exc
    torch = load_torch_module()
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "torch": torch}


def load_torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "HFInferenceEngine requires torch to be installed. Add the optional hf dependencies and sync the environment."
        ) from exc
    return torch


def sequence_length(value) -> int:
    summed = value.sum() if hasattr(value, "sum") else len(value)
    return int(summed.item()) if hasattr(summed, "item") else int(summed)
