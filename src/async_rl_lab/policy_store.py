from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re

from async_rl_lab.ids import utc_ts
from async_rl_lab.models import PolicyRef
from async_rl_lab.serialization import JsonSerializable, JsonValue

TokenPattern = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


@dataclass(frozen=True, slots=True)
class SequencePolicyState(JsonSerializable):
    token_logits: dict[str, float] = field(default_factory=dict)
    action_bias: dict[str, float] = field(
        default_factory=lambda: {"finish": 0.0, "tool_call": 0.0, "invalid": -1.0}
    )
    unk_logit: float = -2.0
    update_count: int = 0
    tokens_seen: int = 0
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequenceScore:
    text: str
    tokens: tuple[str, ...]
    token_logprobs: tuple[float, ...]
    total_logprob: float
    mean_logprob: float
    action_type: str


@dataclass(frozen=True, slots=True)
class PolicyUpdateStats:
    updated_state: SequencePolicyState
    gradient_norm: float
    mean_current_logprob: float
    mean_advantage_weight: float
    token_count: int


class LocalPolicyStore:
    def __init__(
        self,
        root_dir: Path,
        *,
        run_id: str,
        model_family: str = "mock-llm",
    ) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.model_family = model_family
        self.lock = asyncio.Lock()
        self.policy_dir = self.root_dir / "policies"
        self.state_dir = self.root_dir / "policy_states"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = SequencePolicyState(metadata={"published_by": "bootstrap"})
        bootstrap_policy = PolicyRef(
            run_id=run_id,
            policy_version=0,
            policy_tag="bootstrap",
            checkpoint_step=0,
            checkpoint_ts=utc_ts(),
            model_family=model_family,
            policy_path=str(self.state_path_for(0)),
            metadata=self.policy_metadata(self.current_state, extra={"published_by": "bootstrap"}),
        )
        self.write_policy_files(bootstrap_policy, self.current_state)
        self.current = bootstrap_policy

    def current_policy(self) -> PolicyRef:
        return self.current

    def current_state_snapshot(self) -> SequencePolicyState:
        return self.clone_state(self.current_state)

    async def publish_policy(
        self,
        *,
        checkpoint_step: int,
        policy_tag: str,
        checkpoint_path: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
        state: SequencePolicyState | None = None,
    ) -> PolicyRef:
        async with self.lock:
            next_state = self.clone_state(state or self.current_state)
            next_version = self.current.policy_version + 1
            state_path = checkpoint_path or str(self.state_path_for(next_version))
            next_policy = PolicyRef(
                run_id=self.run_id,
                policy_version=next_version,
                policy_tag=policy_tag,
                checkpoint_step=checkpoint_step,
                checkpoint_ts=utc_ts(),
                model_family=self.model_family,
                policy_path=state_path,
                parent_policy_version=self.current.policy_version,
                metadata=self.policy_metadata(next_state, extra=metadata),
            )
            self.write_policy_files(next_policy, next_state)
            self.current_state = next_state
            self.current = next_policy
            return next_policy

    def load_policy(self, policy_version: int) -> PolicyRef | None:
        manifest_path = self.manifest_path_for(policy_version)
        if not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PolicyRef(**payload)

    def load_policy_state(self, policy_version: int) -> SequencePolicyState:
        if policy_version == self.current.policy_version:
            return self.clone_state(self.current_state)
        state_path = self.state_path_for(policy_version)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return SequencePolicyState(**payload)

    def score_text(
        self,
        text: str,
        *,
        policy_version: int | None = None,
        state: SequencePolicyState | None = None,
    ) -> SequenceScore:
        active_state = state or self.load_policy_state(policy_version or self.current.policy_version)
        tokens = tokenize_text(text)
        token_logits = dict(active_state.token_logits)
        for token in tokens:
            token_logits.setdefault(token, active_state.unk_logit)
        if not token_logits:
            token_logits["<unk>"] = active_state.unk_logit
        normalizer = logsumexp(token_logits.values())
        token_logprobs = tuple(token_logits.get(token, active_state.unk_logit) - normalizer for token in tokens)
        action_type = infer_action_type(text)
        action_logprob = action_log_softmax(active_state.action_bias, action_type)
        total_logprob = action_logprob + sum(token_logprobs)
        mean_logprob = total_logprob / max(1, len(tokens))
        return SequenceScore(
            text=text,
            tokens=tokens,
            token_logprobs=token_logprobs,
            total_logprob=total_logprob,
            mean_logprob=mean_logprob,
            action_type=action_type,
        )

    def score_many(
        self,
        texts: Sequence[str],
        *,
        policy_version: int | None = None,
        state: SequencePolicyState | None = None,
    ) -> list[SequenceScore]:
        active_state = state or self.load_policy_state(policy_version or self.current.policy_version)
        return [self.score_text(text, state=active_state) for text in texts]

    def train_on_sequences(
        self,
        *,
        policy_version: int,
        sequences: Sequence[str],
        advantages: Sequence[float],
        sequence_weights: Sequence[float],
        learning_rate: float,
    ) -> PolicyUpdateStats:
        if not (len(sequences) == len(advantages) == len(sequence_weights)):
            raise ValueError("sequences, advantages, and sequence_weights must align")

        base_state = self.load_policy_state(policy_version)
        base_scores = self.score_many(sequences, state=base_state)
        token_logits = dict(base_state.token_logits)
        action_bias = dict(base_state.action_bias)
        total_delta_sq = 0.0
        total_token_count = 0

        for sequence, advantage, weight in zip(sequences, advantages, sequence_weights):
            scaled_advantage = advantage * weight
            tokens = tokenize_text(sequence)
            action_type = infer_action_type(sequence)
            if not tokens or abs(scaled_advantage) < 1e-12:
                continue

            total_token_count += len(tokens)
            for token in tokens:
                token_logits.setdefault(token, base_state.unk_logit)

            logit_values = tuple(token_logits.values())
            normalizer = logsumexp(logit_values)
            probabilities = {
                token: math.exp(logit - normalizer)
                for token, logit in token_logits.items()
            }
            counts = Counter(tokens)
            step_size = learning_rate * scaled_advantage / max(1, len(tokens))

            for token, probability in probabilities.items():
                gradient = counts.get(token, 0) - len(tokens) * probability
                delta = step_size * gradient
                token_logits[token] += delta
                total_delta_sq += delta * delta

            action_probabilities = action_prob_softmax(action_bias)
            for name, probability in action_probabilities.items():
                gradient = 1.0 if name == action_type else 0.0
                gradient -= probability
                delta = learning_rate * scaled_advantage * gradient
                action_bias[name] += delta
                total_delta_sq += delta * delta

        updated_state = SequencePolicyState(
            token_logits=token_logits,
            action_bias=action_bias,
            unk_logit=base_state.unk_logit,
            update_count=base_state.update_count + 1,
            tokens_seen=base_state.tokens_seen + total_token_count,
            metadata={**base_state.metadata, "last_learning_rate": learning_rate},
        )
        mean_current_logprob = (
            sum(score.mean_logprob for score in base_scores) / len(base_scores) if base_scores else 0.0
        )
        mean_advantage_weight = (
            sum(advantage * weight for advantage, weight in zip(advantages, sequence_weights))
            / len(advantages)
            if advantages
            else 0.0
        )
        return PolicyUpdateStats(
            updated_state=updated_state,
            gradient_norm=math.sqrt(total_delta_sq),
            mean_current_logprob=mean_current_logprob,
            mean_advantage_weight=mean_advantage_weight,
            token_count=total_token_count,
        )

    def clone_state(self, state: SequencePolicyState) -> SequencePolicyState:
        return SequencePolicyState(
            token_logits=dict(state.token_logits),
            action_bias=dict(state.action_bias),
            unk_logit=state.unk_logit,
            update_count=state.update_count,
            tokens_seen=state.tokens_seen,
            metadata=dict(state.metadata),
        )

    def policy_metadata(
        self,
        state: SequencePolicyState,
        *,
        extra: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        return {
            "vocab_size": len(state.token_logits),
            "update_count": state.update_count,
            "tokens_seen": state.tokens_seen,
            **(extra or {}),
        }

    def state_path_for(self, policy_version: int) -> Path:
        return self.state_dir / f"policy-v{policy_version:06d}-state.json"

    def manifest_path_for(self, policy_version: int) -> Path:
        return self.policy_dir / f"policy-v{policy_version:06d}.json"

    def write_policy_files(self, policy: PolicyRef, state: SequencePolicyState) -> None:
        state_path = Path(policy.policy_path or self.state_path_for(policy.policy_version))
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_payload = state.to_dict()
        state_text = json.dumps(state_payload, indent=2, sort_keys=True)
        state_path.write_text(state_text, encoding="utf-8")
        state_checksum = hashlib.sha256(state_text.encode("utf-8")).hexdigest()
        manifest = PolicyRef(
            run_id=policy.run_id,
            policy_version=policy.policy_version,
            policy_tag=policy.policy_tag,
            checkpoint_step=policy.checkpoint_step,
            checkpoint_ts=policy.checkpoint_ts,
            model_family=policy.model_family,
            policy_path=str(state_path),
            checkpoint_sha256=state_checksum,
            tokenizer_name="regex-json-v1",
            device_hint="cpu",
            parent_policy_version=policy.parent_policy_version,
            metadata=policy.metadata,
        )
        self.manifest_path_for(policy.policy_version).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def tokenize_text(text: str) -> tuple[str, ...]:
    return tuple(TokenPattern.findall(text))


def infer_action_type(text: str) -> str:
    if '"type"' in text and '"tool_call"' in text:
        return "tool_call"
    if '"type"' in text and ('"finish"' in text or '"respond"' in text):
        return "finish"
    return "invalid"


def action_prob_softmax(action_bias: dict[str, float]) -> dict[str, float]:
    normalizer = logsumexp(action_bias.values())
    return {name: math.exp(value - normalizer) for name, value in action_bias.items()}


def action_log_softmax(action_bias: dict[str, float], action_type: str) -> float:
    normalizer = logsumexp(action_bias.values())
    return action_bias.get(action_type, min(action_bias.values(), default=-1.0) - 1.0) - normalizer


def logsumexp(values: Iterable[float]) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    max_value = max(value_list)
    return max_value + math.log(sum(math.exp(value - max_value) for value in value_list))
