from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
import math
import re
from typing import Protocol

from async_rl_lab.models import Action, GroupedTrajectoryBatch, Trajectory

ActionTypeLabel = str
TurnLogprobRows = tuple[tuple[float, ...], ...]
TrajectoryTurnRows = tuple[TurnLogprobRows, ...]
TurnMaskRows = tuple[tuple[tuple[int, ...], ...], ...]
TurnLabels = tuple[tuple[ActionTypeLabel, ...], ...]
SemanticTokenPattern = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    batch_id: str
    group_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    advantages: tuple[float, ...]
    sequence_weights: tuple[float, ...]
    token_masks: tuple[tuple[int, ...], ...]
    training_turn_masks: TurnMaskRows
    answer_turn_masks: TurnMaskRows
    tool_turn_masks: TurnMaskRows
    turn_action_types: TurnLabels
    behavior_token_logprobs: tuple[tuple[float, ...], ...]
    behavior_turn_token_logprobs: TrajectoryTurnRows
    behavior_sequence_logprobs: tuple[float, ...]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RescoredBatch:
    current_policy_version: int
    reference_policy_version: int | None
    current_token_logprobs: tuple[tuple[float, ...], ...]
    current_turn_token_logprobs: TrajectoryTurnRows
    current_sequence_logprobs: tuple[float, ...]
    reference_token_logprobs: tuple[tuple[float, ...], ...] | None = None
    reference_turn_token_logprobs: TrajectoryTurnRows | None = None
    reference_sequence_logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    loss: float
    update_advantages: tuple[float, ...]
    sequence_losses: tuple[float, ...]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlignedTurnToken:
    behavior_logprob: float
    current_logprob: float
    reference_logprob: float | None
    training_mask: int
    answer_mask: int
    tool_mask: int


@dataclass(frozen=True, slots=True)
class AlignedTurnRow:
    tokens: tuple[AlignedTurnToken, ...]
    truncated_token_count: int = 0


class Objective(Protocol):
    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        ...

    def compute_loss(self, prepared: PreparedBatch, rescored: RescoredBatch) -> ObjectiveResult:
        ...

    def compute_metrics(
        self,
        prepared: PreparedBatch,
        rescored: RescoredBatch,
        result: ObjectiveResult,
    ) -> dict[str, float]:
        ...


class GRPOObjective:
    def __init__(self, *, kl_beta: float = 0.02, advantage_eps: float = 1e-6) -> None:
        self.kl_beta = kl_beta
        self.advantage_eps = advantage_eps

    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        rewards = tuple(trajectory.terminal_reward or 0.0 for trajectory in batch.trajectories)
        grouped_rewards: dict[str, list[float]] = defaultdict(list)
        for trajectory in batch.trajectories:
            grouped_rewards[trajectory.group_id].append(trajectory.terminal_reward or 0.0)

        raw_advantages: list[float] = []
        normalized_advantages: list[float] = []
        behavior_turn_rows: list[TurnLogprobRows] = []
        training_turn_masks: list[tuple[tuple[int, ...], ...]] = []
        answer_turn_masks: list[tuple[tuple[int, ...], ...]] = []
        tool_turn_masks: list[tuple[tuple[int, ...], ...]] = []
        turn_action_types: list[tuple[ActionTypeLabel, ...]] = []
        for trajectory in batch.trajectories:
            group_rewards = grouped_rewards[trajectory.group_id]
            group_mean = sum(group_rewards) / len(group_rewards)
            centered = (trajectory.terminal_reward or 0.0) - group_mean
            group_variance = sum((reward - group_mean) ** 2 for reward in group_rewards) / len(group_rewards)
            group_std = math.sqrt(group_variance)
            raw_advantages.append(centered)
            normalized_advantages.append(centered / (group_std + self.advantage_eps))
            turn_rows = extract_behavior_turn_token_logprobs(trajectory)
            action_types = extract_turn_action_types(trajectory, len(turn_rows))
            training_masks, answer_masks, tool_masks = build_turn_token_masks(
                trajectory,
                turn_rows,
                action_types,
            )
            behavior_turn_rows.append(turn_rows)
            turn_action_types.append(action_types)
            training_turn_masks.append(training_masks)
            answer_turn_masks.append(answer_masks)
            tool_turn_masks.append(tool_masks)

        behavior_tokens = tuple(flatten_turn_rows(turn_rows) for turn_rows in behavior_turn_rows)
        token_masks = tuple(flatten_int_turn_rows(mask_rows) for mask_rows in training_turn_masks)
        behavior_sequence_logprobs = tuple(sum(token_row) for token_row in behavior_tokens)
        return PreparedBatch(
            batch_id=batch.batch_id,
            group_ids=batch.group_ids,
            rewards=rewards,
            raw_advantages=tuple(raw_advantages),
            advantages=tuple(normalized_advantages),
            sequence_weights=tuple(1.0 for _ in batch.trajectories),
            token_masks=token_masks,
            training_turn_masks=tuple(training_turn_masks),
            answer_turn_masks=tuple(answer_turn_masks),
            tool_turn_masks=tuple(tool_turn_masks),
            turn_action_types=tuple(turn_action_types),
            behavior_token_logprobs=behavior_tokens,
            behavior_turn_token_logprobs=tuple(behavior_turn_rows),
            behavior_sequence_logprobs=behavior_sequence_logprobs,
            metrics={
                "mean_reward": mean(rewards),
                "mean_raw_advantage": mean(raw_advantages),
                "mean_sequence_length": mean(tuple(len(mask) for mask in token_masks)),
                "answer_token_fraction": token_fraction(tuple(answer_turn_masks)),
                "tool_token_fraction": token_fraction(tuple(tool_turn_masks)),
                "masked_token_fraction": masked_fraction(tuple(training_turn_masks)),
                "invalid_turn_fraction": invalid_turn_fraction(tuple(turn_action_types)),
                "truncated_trajectory_fraction": truncated_fraction(batch.trajectories),
            },
        )

    def compute_loss(self, prepared: PreparedBatch, rescored: RescoredBatch) -> ObjectiveResult:
        return self.compute_token_level_objective(prepared, rescored, ratio_clip=None)

    def compute_metrics(
        self,
        prepared: PreparedBatch,
        rescored: RescoredBatch,
        result: ObjectiveResult,
    ) -> dict[str, float]:
        return {
            "mean_reward": prepared.metrics.get("mean_reward", 0.0),
            "mean_advantage": mean(prepared.advantages),
            "mean_raw_advantage": prepared.metrics.get("mean_raw_advantage", 0.0),
            "mean_sequence_length": prepared.metrics.get("mean_sequence_length", 0.0),
            "answer_token_fraction": prepared.metrics.get("answer_token_fraction", 0.0),
            "tool_token_fraction": prepared.metrics.get("tool_token_fraction", 0.0),
            "masked_token_fraction": prepared.metrics.get("masked_token_fraction", 0.0),
            "invalid_turn_fraction": prepared.metrics.get("invalid_turn_fraction", 0.0),
            "truncated_trajectory_fraction": prepared.metrics.get("truncated_trajectory_fraction", 0.0),
            "mean_behavior_sequence_logprob": mean(prepared.behavior_sequence_logprobs),
            "mean_current_sequence_logprob": mean(rescored.current_sequence_logprobs),
            "mean_reference_sequence_logprob": mean(rescored.reference_sequence_logprobs or ()),
            **result.metrics,
        }

    def compute_token_level_objective(
        self,
        prepared: PreparedBatch,
        rescored: RescoredBatch,
        *,
        ratio_clip: float | None,
    ) -> ObjectiveResult:
        sequence_losses: list[float] = []
        update_advantages: list[float] = []
        clipped_tokens = 0
        active_tokens = 0
        alignment_truncation_tokens = 0
        answer_tokens = 0
        tool_tokens = 0
        other_tokens = 0
        mean_ratios: list[float] = []
        mean_kls: list[float] = []
        effective_lengths: list[float] = []

        for index, advantage in enumerate(prepared.advantages):
            aligned_rows = align_turn_logprob_rows(
                prepared.behavior_turn_token_logprobs[index],
                rescored.current_turn_token_logprobs[index],
                rescored.reference_turn_token_logprobs[index]
                if rescored.reference_turn_token_logprobs is not None
                else None,
                prepared.training_turn_masks[index],
                prepared.answer_turn_masks[index],
                prepared.tool_turn_masks[index],
            )

            token_surrogates: list[float] = []
            ratio_values: list[float] = []
            kl_values: list[float] = []
            for aligned_row in aligned_rows:
                alignment_truncation_tokens += aligned_row.truncated_token_count
                for token in aligned_row.tokens:
                    if token.training_mask == 0:
                        continue
                    active_tokens += 1
                    ratio = math.exp(token.current_logprob - token.behavior_logprob)
                    effective_ratio = ratio
                    if ratio_clip is not None:
                        clipped_ratio = clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                        if clipped_ratio != ratio:
                            clipped_tokens += 1
                        effective_ratio = clipped_ratio
                    surrogate = advantage * effective_ratio
                    token_surrogates.append(surrogate)
                    ratio_values.append(effective_ratio)
                    if token.reference_logprob is not None:
                        kl_values.append(
                            sampled_forward_kl_penalty(
                                current_logprob=token.current_logprob,
                                reference_logprob=token.reference_logprob,
                            )
                        )
                    if token.answer_mask == 1:
                        answer_tokens += 1
                    elif token.tool_mask == 1:
                        tool_tokens += 1
                    else:
                        other_tokens += 1

            if not token_surrogates:
                fallback_ratio = math.exp(
                    rescored.current_sequence_logprobs[index] - prepared.behavior_sequence_logprobs[index]
                )
                if ratio_clip is not None:
                    fallback_ratio = clip(fallback_ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                token_surrogates.append(advantage * fallback_ratio)
                ratio_values.append(fallback_ratio)
                if rescored.reference_sequence_logprobs is not None:
                    kl_values.append(
                        sampled_forward_kl_penalty(
                            current_logprob=rescored.current_sequence_logprobs[index],
                            reference_logprob=rescored.reference_sequence_logprobs[index],
                        )
                    )

            sequence_surrogate = mean(tuple(token_surrogates))
            sequence_kl = mean(tuple(kl_values))
            sequence_weight = prepared.sequence_weights[index]
            sequence_loss = -(sequence_weight * sequence_surrogate) + (self.kl_beta * sequence_kl)
            sequence_losses.append(sequence_loss)
            update_advantages.append(sequence_weight * advantage * mean(tuple(ratio_values)))
            mean_ratios.extend(ratio_values)
            mean_kls.append(sequence_kl)
            effective_lengths.append(float(len(token_surrogates)))

        loss = mean(tuple(sequence_losses))
        role_token_total = answer_tokens + tool_tokens + other_tokens
        return ObjectiveResult(
            loss=loss,
            update_advantages=tuple(update_advantages),
            sequence_losses=tuple(sequence_losses),
            metrics={
                "mean_ratio": mean(tuple(mean_ratios)),
                "mean_kl": mean(tuple(mean_kls)),
                "clip_fraction": clipped_tokens / max(1, active_tokens),
                "mean_staleness_weight": mean(prepared.sequence_weights),
                "mean_effective_tokens": mean(tuple(effective_lengths)),
                "alignment_truncation_tokens": float(alignment_truncation_tokens),
                "objective_answer_token_fraction": answer_tokens / max(1, role_token_total),
                "objective_tool_token_fraction": tool_tokens / max(1, role_token_total),
            },
        )


class DAPOObjective(GRPOObjective):
    def __init__(
        self,
        *,
        clip_ratio: float = 0.2,
        clip_advantage_abs: float = 5.0,
        min_weight: float = 0.2,
        max_weight: float = 2.0,
        kl_beta: float = 0.02,
    ) -> None:
        super().__init__(kl_beta=kl_beta)
        self.clip_ratio = clip_ratio
        self.clip_advantage_abs = clip_advantage_abs
        self.min_weight = min_weight
        self.max_weight = max_weight

    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        prepared = super().prepare_batch(batch)
        clipped_advantages = tuple(
            clip(advantage, -self.clip_advantage_abs, self.clip_advantage_abs)
            for advantage in prepared.advantages
        )
        reward_scale = math.sqrt(sum(reward * reward for reward in prepared.rewards) / max(1, len(prepared.rewards))) or 1.0
        weights = tuple(
            max(self.min_weight, min(self.max_weight, abs(reward) / reward_scale))
            for reward in prepared.rewards
        )
        return PreparedBatch(
            batch_id=prepared.batch_id,
            group_ids=prepared.group_ids,
            rewards=prepared.rewards,
            raw_advantages=prepared.raw_advantages,
            advantages=clipped_advantages,
            sequence_weights=weights,
            token_masks=prepared.token_masks,
            training_turn_masks=prepared.training_turn_masks,
            answer_turn_masks=prepared.answer_turn_masks,
            tool_turn_masks=prepared.tool_turn_masks,
            turn_action_types=prepared.turn_action_types,
            behavior_token_logprobs=prepared.behavior_token_logprobs,
            behavior_turn_token_logprobs=prepared.behavior_turn_token_logprobs,
            behavior_sequence_logprobs=prepared.behavior_sequence_logprobs,
            metrics={
                **prepared.metrics,
                "reward_scale": reward_scale,
                "advantage_clip_fraction": sum(
                    1.0 for old, new in zip(prepared.advantages, clipped_advantages) if old != new
                )
                / max(1, len(clipped_advantages)),
            },
        )

    def compute_loss(self, prepared: PreparedBatch, rescored: RescoredBatch) -> ObjectiveResult:
        return self.compute_token_level_objective(prepared, rescored, ratio_clip=self.clip_ratio)


class StalenessWeightedGRPO(GRPOObjective):
    def __init__(self, *, lag_penalty: float = 0.15, kl_beta: float = 0.02) -> None:
        super().__init__(kl_beta=kl_beta)
        self.lag_penalty = lag_penalty

    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        prepared = super().prepare_batch(batch)
        lag_by_trajectory = (
            batch.staleness_stats.lag_by_trajectory
            if batch.staleness_stats
            else tuple(0 for _ in batch.trajectories)
        )
        weights = tuple(math.exp(-self.lag_penalty * lag) for lag in lag_by_trajectory)
        return PreparedBatch(
            batch_id=prepared.batch_id,
            group_ids=prepared.group_ids,
            rewards=prepared.rewards,
            raw_advantages=prepared.raw_advantages,
            advantages=prepared.advantages,
            sequence_weights=weights,
            token_masks=prepared.token_masks,
            training_turn_masks=prepared.training_turn_masks,
            answer_turn_masks=prepared.answer_turn_masks,
            tool_turn_masks=prepared.tool_turn_masks,
            turn_action_types=prepared.turn_action_types,
            behavior_token_logprobs=prepared.behavior_token_logprobs,
            behavior_turn_token_logprobs=prepared.behavior_turn_token_logprobs,
            behavior_sequence_logprobs=prepared.behavior_sequence_logprobs,
            metrics={**prepared.metrics, "mean_staleness_weight": mean(weights)},
        )


def extract_behavior_turn_token_logprobs(trajectory: Trajectory) -> TurnLogprobRows:
    rows = tuple(tuple(float(token) for token in row) for row in (trajectory.behavior_token_logprobs or ()))
    if not rows and trajectory.behavior_logprobs:
        rows = tuple((float(value),) for value in trajectory.behavior_logprobs)
    if not rows:
        fallback_count = max(1, len(trajectory.parsed_action_trace))
        rows = tuple((0.0,) for _ in range(fallback_count))

    fallback_values = tuple(float(value) for value in (trajectory.behavior_logprobs or ()))
    target_count = max(len(rows), len(trajectory.parsed_action_trace))
    normalized_rows: list[tuple[float, ...]] = []
    for index in range(target_count):
        if index < len(rows) and rows[index]:
            normalized_rows.append(rows[index])
        else:
            fallback_value = fallback_values[index] if index < len(fallback_values) else 0.0
            normalized_rows.append((fallback_value,))
    return tuple(normalized_rows)


def extract_turn_action_types(trajectory: Trajectory, default_count: int) -> tuple[ActionTypeLabel, ...]:
    if trajectory.parsed_action_trace:
        return tuple(action.action_type for action in trajectory.parsed_action_trace)
    return tuple("finish" for _ in range(max(1, default_count)))


def build_turn_token_masks(
    trajectory: Trajectory,
    turn_rows: TurnLogprobRows,
    action_types: tuple[ActionTypeLabel, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    training_masks: list[tuple[int, ...]] = []
    answer_masks: list[tuple[int, ...]] = []
    tool_masks: list[tuple[int, ...]] = []
    last_turn_index = max(0, len(turn_rows) - 1)
    for index, row in enumerate(turn_rows):
        action_type = action_types[index] if index < len(action_types) else "finish"
        token_count = max(1, len(row))
        is_truncated_tail = trajectory.is_truncated and index == last_turn_index
        action = trajectory.parsed_action_trace[index] if index < len(trajectory.parsed_action_trace) else None
        training_row, answer_row, tool_row = build_action_semantic_masks(
            action=action,
            action_type=action_type,
            target_token_count=token_count,
        )
        if action_type == "invalid" or is_truncated_tail:
            zero_row = tuple(0 for _ in range(token_count))
            training_masks.append(zero_row)
            answer_masks.append(zero_row)
            tool_masks.append(zero_row)
            continue
        if sum(training_row) == 0:
            training_row = tuple(1 for _ in range(token_count))
            answer_row = tuple(1 if action_type in {"finish", "respond"} else 0 for _ in range(token_count))
            tool_row = tuple(1 if action_type == "tool_call" else 0 for _ in range(token_count))
        training_masks.append(training_row)
        answer_masks.append(answer_row)
        tool_masks.append(tool_row)
    return tuple(training_masks), tuple(answer_masks), tuple(tool_masks)


def align_turn_logprob_rows(
    behavior_turns: TurnLogprobRows,
    current_turns: TurnLogprobRows,
    reference_turns: TurnLogprobRows | None,
    training_masks: tuple[tuple[int, ...], ...],
    answer_masks: tuple[tuple[int, ...], ...],
    tool_masks: tuple[tuple[int, ...], ...],
) -> tuple[AlignedTurnRow, ...]:
    turn_limit = min(
        len(behavior_turns),
        len(current_turns),
        len(reference_turns) if reference_turns is not None else len(current_turns),
        len(training_masks),
        len(answer_masks),
        len(tool_masks),
    )
    aligned_rows: list[AlignedTurnRow] = []
    for turn_index in range(turn_limit):
        behavior_row = behavior_turns[turn_index]
        current_row = current_turns[turn_index]
        reference_row = reference_turns[turn_index] if reference_turns is not None else None
        training_mask_row = training_masks[turn_index]
        answer_mask_row = answer_masks[turn_index]
        tool_mask_row = tool_masks[turn_index]
        token_limit = min(
            len(behavior_row),
            len(current_row),
            len(reference_row) if reference_row is not None else len(current_row),
            len(training_mask_row),
            len(answer_mask_row),
            len(tool_mask_row),
        )
        tokens = tuple(
            AlignedTurnToken(
                behavior_logprob=float(behavior_row[token_index]),
                current_logprob=float(current_row[token_index]),
                reference_logprob=float(reference_row[token_index]) if reference_row is not None else None,
                training_mask=int(training_mask_row[token_index]),
                answer_mask=int(answer_mask_row[token_index]),
                tool_mask=int(tool_mask_row[token_index]),
            )
            for token_index in range(token_limit)
        )
        aligned_rows.append(
            AlignedTurnRow(
                tokens=tokens,
                truncated_token_count=max(
                    len(behavior_row),
                    len(current_row),
                    len(reference_row) if reference_row is not None else len(current_row),
                    len(training_mask_row),
                    len(answer_mask_row),
                    len(tool_mask_row),
                )
                - token_limit,
            )
        )
    return tuple(aligned_rows)


def flatten_turn_rows(turn_rows: TurnLogprobRows) -> tuple[float, ...]:
    return tuple(token for row in turn_rows for token in row)


def flatten_int_turn_rows(turn_rows: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(value for row in turn_rows for value in row)


def build_action_semantic_masks(
    *,
    action: Action | None,
    action_type: ActionTypeLabel,
    target_token_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if action is None:
        zero_row = tuple(0 for _ in range(target_token_count))
        return zero_row, zero_row, zero_row

    raw_tokens = tokenize_semantic_text(action.raw_text)
    if not raw_tokens:
        zero_row = tuple(0 for _ in range(target_token_count))
        return zero_row, zero_row, zero_row

    answer_source = tuple(0 for _ in raw_tokens)
    tool_source = tuple(0 for _ in raw_tokens)
    if action_type in {"finish", "respond"}:
        answer_source = mark_token_subsequence(raw_tokens, extract_answer_tokens(action))
    elif action_type == "tool_call":
        tool_source = mark_token_subsequence(raw_tokens, extract_tool_tokens(action))

    training_source = tuple(1 if answer == 1 or tool == 1 else 0 for answer, tool in zip(answer_source, tool_source))
    return (
        project_mask_row(training_source, target_token_count),
        project_mask_row(answer_source, target_token_count),
        project_mask_row(tool_source, target_token_count),
    )


def tokenize_semantic_text(text: str) -> tuple[str, ...]:
    return tuple(SemanticTokenPattern.findall(text))


def extract_answer_tokens(action: Action) -> tuple[str, ...]:
    if action.final_text:
        return tokenize_semantic_text(action.final_text)
    payload = parse_action_payload(action.raw_text)
    answer_value = payload.get("answer") or payload.get("text")
    if isinstance(answer_value, str):
        return tokenize_semantic_text(answer_value)
    return ()


def extract_tool_tokens(action: Action) -> tuple[str, ...]:
    if action.tool_call is not None:
        direct_tool_tokens = list(tokenize_semantic_text(action.tool_call.tool_name))
        for value in action.tool_call.arguments.values():
            direct_tool_tokens.extend(flatten_json_value_tokens(value))
        return tuple(direct_tool_tokens)

    payload = parse_action_payload(action.raw_text)
    payload_tool_tokens: list[str] = []
    tool_name = payload.get("tool_name")
    if isinstance(tool_name, str):
        payload_tool_tokens.extend(tokenize_semantic_text(tool_name))
    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        for value in arguments.values():
            payload_tool_tokens.extend(flatten_json_value_tokens(value))
    return tuple(payload_tool_tokens)


def parse_action_payload(raw_text: str) -> dict[str, object]:
    try:
        payload = json.loads(raw_text.strip())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def flatten_json_value_tokens(value: object) -> list[str]:
    if isinstance(value, str):
        return list(tokenize_semantic_text(value))
    if isinstance(value, bool):
        return [str(value).lower()]
    if isinstance(value, int | float):
        return list(tokenize_semantic_text(str(value)))
    if isinstance(value, list):
        list_tokens: list[str] = []
        for item in value:
            list_tokens.extend(flatten_json_value_tokens(item))
        return list_tokens
    if isinstance(value, dict):
        dict_tokens: list[str] = []
        for item in value.values():
            dict_tokens.extend(flatten_json_value_tokens(item))
        return dict_tokens
    return []


def mark_token_subsequence(source_tokens: tuple[str, ...], target_tokens: tuple[str, ...]) -> tuple[int, ...]:
    if not source_tokens or not target_tokens:
        return tuple(0 for _ in source_tokens)
    marked = [0 for _ in source_tokens]
    cursor = 0
    for target_token in target_tokens:
        found_index = find_token(source_tokens, target_token, start=cursor)
        if found_index is None:
            continue
        marked[found_index] = 1
        cursor = found_index + 1
    return tuple(marked)


def find_token(source_tokens: tuple[str, ...], target_token: str, *, start: int) -> int | None:
    for index in range(start, len(source_tokens)):
        if source_tokens[index] == target_token:
            return index
    return None


def project_mask_row(mask_row: tuple[int, ...], target_length: int) -> tuple[int, ...]:
    if target_length <= 0:
        return ()
    if not mask_row:
        return tuple(0 for _ in range(target_length))
    if len(mask_row) == target_length:
        return mask_row
    projected: list[int] = []
    source_length = len(mask_row)
    for target_index in range(target_length):
        source_index = min(source_length - 1, int((target_index * source_length) / target_length))
        projected.append(mask_row[source_index])
    return tuple(projected)


def sampled_forward_kl_penalty(*, current_logprob: float, reference_logprob: float) -> float:
    """
    Uses `exp(log pi_ref - log pi_cur) - (log pi_ref - log pi_cur) - 1`, which is a
    per-sample estimator for `KL(current || reference)` under actions sampled from the
    current policy. This convention is explicit here because sign mistakes are easy to make.
    """

    delta = reference_logprob - current_logprob
    return math.exp(delta) - delta - 1.0


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def token_fraction(mask_rows: TurnMaskRows) -> float:
    total_tokens = sum(len(row) for trajectory_rows in mask_rows for row in trajectory_rows)
    active_tokens = sum(value for trajectory_rows in mask_rows for row in trajectory_rows for value in row)
    return active_tokens / max(1, total_tokens)


def masked_fraction(mask_rows: TurnMaskRows) -> float:
    total_tokens = sum(len(row) for trajectory_rows in mask_rows for row in trajectory_rows)
    masked_tokens = sum(
        1
        for trajectory_rows in mask_rows
        for row in trajectory_rows
        for value in row
        if value == 0
    )
    return masked_tokens / max(1, total_tokens)


def invalid_turn_fraction(turn_labels: TurnLabels) -> float:
    total_turns = sum(len(labels) for labels in turn_labels)
    invalid_turns = sum(1 for labels in turn_labels for label in labels if label == "invalid")
    return invalid_turns / max(1, total_turns)


def truncated_fraction(trajectories: tuple[Trajectory, ...]) -> float:
    return sum(1.0 for trajectory in trajectories if trajectory.is_truncated) / max(1, len(trajectories))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
