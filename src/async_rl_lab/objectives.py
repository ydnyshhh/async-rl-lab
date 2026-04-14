from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import Protocol

from async_rl_lab.models import GroupedTrajectoryBatch


@dataclass(frozen=True, slots=True)
class PreparedBatch:
    batch_id: str
    group_ids: tuple[str, ...]
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    sequence_weights: tuple[float, ...]
    token_masks: tuple[tuple[int, ...], ...]
    metrics: dict[str, float] = field(default_factory=dict)


class Objective(Protocol):
    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        ...

    def compute_loss(self, prepared: PreparedBatch, model_logprob_summaries: Sequence[float]) -> float:
        ...

    def compute_metrics(self, prepared: PreparedBatch) -> dict[str, float]:
        ...


class GRPOObjective:
    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        rewards = tuple(trajectory.terminal_reward or 0.0 for trajectory in batch.trajectories)
        grouped_rewards: dict[str, list[float]] = defaultdict(list)
        for trajectory in batch.trajectories:
            grouped_rewards[trajectory.group_id].append(trajectory.terminal_reward or 0.0)

        advantages: list[float] = []
        for trajectory in batch.trajectories:
            group_rewards = grouped_rewards[trajectory.group_id]
            group_mean = sum(group_rewards) / len(group_rewards)
            advantages.append((trajectory.terminal_reward or 0.0) - group_mean)

        token_masks = tuple(tuple(1 for _ in trajectory.parsed_action_trace) or (1,) for trajectory in batch.trajectories)
        return PreparedBatch(
            batch_id=batch.batch_id,
            group_ids=batch.group_ids,
            rewards=rewards,
            advantages=tuple(advantages),
            sequence_weights=tuple(1.0 for _ in batch.trajectories),
            token_masks=token_masks,
            metrics={"mean_reward": sum(rewards) / len(rewards) if rewards else 0.0},
        )

    def compute_loss(self, prepared: PreparedBatch, model_logprob_summaries: Sequence[float]) -> float:
        if len(model_logprob_summaries) != len(prepared.advantages):
            raise ValueError("model_logprob_summaries must align with trajectories.")
        total = 0.0
        for advantage, weight, logprob_summary in zip(
            prepared.advantages,
            prepared.sequence_weights,
            model_logprob_summaries,
        ):
            total += -(advantage * weight * logprob_summary)
        return total / max(1, len(model_logprob_summaries))

    def compute_metrics(self, prepared: PreparedBatch) -> dict[str, float]:
        mean_advantage = sum(prepared.advantages) / len(prepared.advantages) if prepared.advantages else 0.0
        return {"mean_reward": prepared.metrics.get("mean_reward", 0.0), "mean_advantage": mean_advantage}


class DAPOObjective(GRPOObjective):
    def __init__(self, *, clip_advantage_abs: float = 5.0, min_weight: float = 0.2, max_weight: float = 2.0) -> None:
        self.clip_advantage_abs = clip_advantage_abs
        self.min_weight = min_weight
        self.max_weight = max_weight

    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        prepared = super().prepare_batch(batch)
        clipped_advantages = tuple(
            max(-self.clip_advantage_abs, min(self.clip_advantage_abs, advantage))
            for advantage in prepared.advantages
        )
        rewards = prepared.rewards
        reward_scale = math.sqrt(sum(reward * reward for reward in rewards) / max(1, len(rewards))) or 1.0
        weights = tuple(
            max(self.min_weight, min(self.max_weight, abs(reward) / reward_scale))
            for reward in rewards
        )
        return PreparedBatch(
            batch_id=prepared.batch_id,
            group_ids=prepared.group_ids,
            rewards=prepared.rewards,
            advantages=clipped_advantages,
            sequence_weights=weights,
            token_masks=prepared.token_masks,
            metrics={
                **prepared.metrics,
                "reward_scale": reward_scale,
                "clipped_fraction": sum(
                    1.0 for old, new in zip(prepared.advantages, clipped_advantages) if old != new
                )
                / max(1, len(clipped_advantages)),
            },
        )


class StalenessWeightedGRPO(GRPOObjective):
    def __init__(self, *, lag_penalty: float = 0.15) -> None:
        self.lag_penalty = lag_penalty

    def prepare_batch(self, batch: GroupedTrajectoryBatch) -> PreparedBatch:
        prepared = super().prepare_batch(batch)
        lag_by_trajectory = batch.staleness_stats.lag_by_trajectory if batch.staleness_stats else tuple(0 for _ in batch.trajectories)
        weights = tuple(math.exp(-self.lag_penalty * lag) for lag in lag_by_trajectory)
        return PreparedBatch(
            batch_id=prepared.batch_id,
            group_ids=prepared.group_ids,
            rewards=prepared.rewards,
            advantages=prepared.advantages,
            sequence_weights=weights,
            token_masks=prepared.token_masks,
            metrics={**prepared.metrics, "mean_staleness_weight": sum(weights) / len(weights) if weights else 0.0},
        )
