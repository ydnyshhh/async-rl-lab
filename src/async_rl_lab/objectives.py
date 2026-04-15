from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol

from async_rl_lab.models import GroupedTrajectoryBatch, Trajectory


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    batch_id: str
    group_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    advantages: tuple[float, ...]
    sequence_weights: tuple[float, ...]
    token_masks: tuple[tuple[int, ...], ...]
    behavior_token_logprobs: tuple[tuple[float, ...], ...]
    behavior_sequence_logprobs: tuple[float, ...]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RescoredBatch:
    current_policy_version: int
    reference_policy_version: int | None
    current_token_logprobs: tuple[tuple[float, ...], ...]
    current_sequence_logprobs: tuple[float, ...]
    reference_token_logprobs: tuple[tuple[float, ...], ...] | None = None
    reference_sequence_logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    loss: float
    update_advantages: tuple[float, ...]
    sequence_losses: tuple[float, ...]
    metrics: dict[str, float] = field(default_factory=dict)


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
        for trajectory in batch.trajectories:
            group_rewards = grouped_rewards[trajectory.group_id]
            group_mean = sum(group_rewards) / len(group_rewards)
            centered = (trajectory.terminal_reward or 0.0) - group_mean
            group_variance = sum((reward - group_mean) ** 2 for reward in group_rewards) / len(group_rewards)
            group_std = math.sqrt(group_variance)
            raw_advantages.append(centered)
            normalized_advantages.append(centered / (group_std + self.advantage_eps))

        behavior_tokens = tuple(extract_behavior_token_logprobs(trajectory) for trajectory in batch.trajectories)
        token_masks = tuple(tuple(1 for _ in token_row) or (1,) for token_row in behavior_tokens)
        behavior_sequence_logprobs = tuple(sum(token_row) for token_row in behavior_tokens)
        return PreparedBatch(
            batch_id=batch.batch_id,
            group_ids=batch.group_ids,
            rewards=rewards,
            raw_advantages=tuple(raw_advantages),
            advantages=tuple(normalized_advantages),
            sequence_weights=tuple(1.0 for _ in batch.trajectories),
            token_masks=token_masks,
            behavior_token_logprobs=behavior_tokens,
            behavior_sequence_logprobs=behavior_sequence_logprobs,
            metrics={
                "mean_reward": mean(rewards),
                "mean_raw_advantage": mean(raw_advantages),
                "mean_sequence_length": mean(tuple(len(mask) for mask in token_masks)),
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
        clip_count = 0
        token_count = 0
        mean_ratios: list[float] = []
        mean_kls: list[float] = []
        effective_lengths: list[float] = []

        for index, advantage in enumerate(prepared.advantages):
            aligned = align_logprob_rows(
                prepared.behavior_token_logprobs[index],
                rescored.current_token_logprobs[index],
                rescored.reference_token_logprobs[index] if rescored.reference_token_logprobs is not None else None,
            )
            if not aligned:
                aligned = ((prepared.behavior_sequence_logprobs[index], rescored.current_sequence_logprobs[index], None),)

            token_surrogates: list[float] = []
            ratio_values: list[float] = []
            kl_values: list[float] = []
            for behavior_logprob, current_logprob, reference_logprob in aligned:
                ratio = math.exp(current_logprob - behavior_logprob)
                effective_ratio = ratio
                if ratio_clip is not None:
                    clipped_ratio = clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                    clip_count += 1 if clipped_ratio != ratio else 0
                    effective_ratio = select_clipped_ratio(advantage, ratio, clipped_ratio)
                surrogate = advantage * effective_ratio
                token_surrogates.append(surrogate)
                ratio_values.append(effective_ratio)
                if reference_logprob is not None:
                    logprob_delta = reference_logprob - current_logprob
                    kl_values.append(math.exp(logprob_delta) - logprob_delta - 1.0)

            sequence_surrogate = mean(tuple(token_surrogates))
            sequence_kl = mean(tuple(kl_values))
            sequence_weight = prepared.sequence_weights[index]
            sequence_loss = -(sequence_weight * sequence_surrogate) + (self.kl_beta * sequence_kl)
            sequence_losses.append(sequence_loss)
            update_advantages.append(sequence_weight * advantage * mean(tuple(ratio_values)))
            mean_ratios.extend(ratio_values)
            mean_kls.append(sequence_kl)
            effective_lengths.append(float(len(token_surrogates)))
            token_count += len(token_surrogates)

        loss = mean(tuple(sequence_losses))
        return ObjectiveResult(
            loss=loss,
            update_advantages=tuple(update_advantages),
            sequence_losses=tuple(sequence_losses),
            metrics={
                "mean_ratio": mean(tuple(mean_ratios)),
                "mean_kl": mean(tuple(mean_kls)),
                "clip_fraction": clip_count / max(1, token_count),
                "mean_staleness_weight": mean(prepared.sequence_weights),
                "mean_effective_tokens": mean(tuple(effective_lengths)),
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
            behavior_token_logprobs=prepared.behavior_token_logprobs,
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
            behavior_token_logprobs=prepared.behavior_token_logprobs,
            behavior_sequence_logprobs=prepared.behavior_sequence_logprobs,
            metrics={**prepared.metrics, "mean_staleness_weight": mean(weights)},
        )


def extract_behavior_token_logprobs(trajectory: Trajectory) -> tuple[float, ...]:
    if trajectory.behavior_token_logprobs:
        flattened = [token for row in trajectory.behavior_token_logprobs for token in row]
        if flattened:
            return tuple(flattened)
    if trajectory.behavior_logprobs:
        return tuple(trajectory.behavior_logprobs)
    return (0.0,)


def align_logprob_rows(
    behavior_row: Sequence[float],
    current_row: Sequence[float],
    reference_row: Sequence[float] | None,
) -> tuple[tuple[float, float, float | None], ...]:
    limit = min(
        len(behavior_row),
        len(current_row),
        len(reference_row) if reference_row is not None else len(current_row),
    )
    return tuple(
        (
            float(behavior_row[index]),
            float(current_row[index]),
            float(reference_row[index]) if reference_row is not None else None,
        )
        for index in range(limit)
    )


def select_clipped_ratio(advantage: float, ratio: float, clipped_ratio: float) -> float:
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    if advantage >= 0.0:
        return min(unclipped, clipped) / max(advantage, 1e-12)
    return max(unclipped, clipped) / min(advantage, -1e-12)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
