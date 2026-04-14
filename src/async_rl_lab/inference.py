from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
from typing import Protocol

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import Action, GenerationRequest, GenerationResult, PolicyRef, ToolCall


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
        batch_window_ms: float = 10.0,
        max_batch_size: int = 16,
        artificial_latency_ms: float = 25.0,
    ) -> None:
        self.loaded_policy = initial_policy
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.artificial_latency_ms = artificial_latency_ms
        self.pending_queue: asyncio.Queue[PendingGeneration] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self.batch_worker())

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        if self.worker_task is None:
            await self.start()
        future: asyncio.Future[GenerationResult] = asyncio.get_running_loop().create_future()
        await self.pending_queue.put(PendingGeneration(request=request, future=future, enqueue_ts=utc_ts()))
        return await future

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        results = await asyncio.gather(*(self.submit(request) for request in requests))
        return list(results)

    async def refresh_policy(self, policy: PolicyRef) -> None:
        await asyncio.sleep(self.artificial_latency_ms / 1000.0)
        self.loaded_policy = policy

    def current_policy(self) -> PolicyRef:
        return self.loaded_policy

    async def close(self) -> None:
        self.stop_event.set()
        if self.worker_task is not None:
            await self.worker_task

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

            await self.process_batch(batch)

    async def process_batch(self, batch: list[PendingGeneration]) -> None:
        started_ts = utc_ts()
        await asyncio.sleep(self.artificial_latency_ms / 1000.0)
        for pending in batch:
            generated_text = self.generate_text(pending.request)
            action = parse_action_text(generated_text)
            ended_ts = utc_ts()
            prompt_tokens = max(1, len(pending.request.observation_text.split()))
            completion_tokens = max(1, len(generated_text.split()))
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
                queue_wait_ms=(started_ts - pending.enqueue_ts) * 1000.0,
                latency_ms=(ended_ts - started_ts) * 1000.0,
                action=action,
                metadata={"engine_policy_version": self.loaded_policy.policy_version},
            )
            if not pending.future.done():
                pending.future.set_result(result)

    def generate_text(self, request: GenerationRequest) -> str:
        if "forced_output" in request.metadata:
            return str(request.metadata["forced_output"])

        gold_answer = request.metadata.get("gold_answer")
        expression = request.metadata.get("expression")
        if expression and "calculator" in request.available_tools and request.sample_index_within_group % 2 == 0:
            return json.dumps(
                {
                    "type": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": str(expression)},
                }
            )
        if gold_answer is not None:
            answer = str(gold_answer) if request.sample_index_within_group % 3 == 0 else f"{gold_answer}?"
            return json.dumps({"type": "finish", "answer": answer})
        return json.dumps({"type": "finish", "answer": request.observation_text[:32]})


class HFInferenceEngine:
    def __init__(self, initial_policy: PolicyRef) -> None:
        self.loaded_policy = initial_policy

    async def start(self) -> None:
        return None

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError("HFInferenceEngine requires transformers integration in a follow-up patch.")

    async def submit_many(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        return [await self.submit(request) for request in requests]

    async def refresh_policy(self, policy: PolicyRef) -> None:
        self.loaded_policy = policy

    def current_policy(self) -> PolicyRef:
        return self.loaded_policy

    async def close(self) -> None:
        return None


class VLLMInferenceEngine:
    def __init__(self, initial_policy: PolicyRef) -> None:
        self.loaded_policy = initial_policy

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

    async def close(self) -> None:
        return None
