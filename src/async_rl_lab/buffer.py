from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import GroupedTrajectoryBatch, StalenessStats, Trajectory


@dataclass(frozen=True, slots=True)
class BufferInsertResult:
    inserted_group_id: str
    inserted_trajectories: int
    dropped_group_ids: tuple[str, ...]


class InMemoryGroupedRolloutBuffer:
    def __init__(
        self,
        *,
        capacity_groups: int,
        required_group_size: int,
        drop_policy: str = "drop_oldest",
        max_policy_lag: int | None = None,
        max_age_ms: float | None = None,
        keep_groups_intact: bool = True,
    ) -> None:
        self.capacity_groups = capacity_groups
        self.required_group_size = required_group_size
        self.drop_policy = drop_policy
        self.max_policy_lag = max_policy_lag
        self.max_age_ms = max_age_ms
        self.keep_groups_intact = keep_groups_intact
        self.group_order: deque[str] = deque()
        self.groups: dict[str, tuple[Trajectory, ...]] = {}
        self.actor_counter: Counter[str] = Counter()

    def current_size(self) -> int:
        return sum(len(group) for group in self.groups.values())

    def group_count(self) -> int:
        return len(self.groups)

    def group_completeness(self) -> dict[str, int]:
        return {group_id: len(group) for group_id, group in self.groups.items()}

    def per_actor_contribution(self) -> dict[str, int]:
        return dict(self.actor_counter)

    def average_age_ms(self, now_ts: float | None = None) -> float:
        now_ts = now_ts or utc_ts()
        ages = [self.age_ms(trajectory, now_ts) for group in self.groups.values() for trajectory in group]
        return sum(ages) / len(ages) if ages else 0.0

    def oldest_item_age_ms(self, now_ts: float | None = None) -> float:
        now_ts = now_ts or utc_ts()
        ages = [self.age_ms(trajectory, now_ts) for group in self.groups.values() for trajectory in group]
        return max(ages, default=0.0)

    def staleness_histogram(self, learner_policy_version: int) -> dict[int, int]:
        histogram: dict[int, int] = {}
        for group in self.groups.values():
            for trajectory in group:
                lag = max(0, learner_policy_version - trajectory.behavior_policy_version)
                histogram[lag] = histogram.get(lag, 0) + 1
        return histogram

    def insert_group(self, trajectories: tuple[Trajectory, ...]) -> BufferInsertResult:
        if not trajectories:
            raise ValueError("insert_group requires at least one trajectory.")
        group_id = trajectories[0].group_id
        if self.keep_groups_intact and len(trajectories) != self.required_group_size:
            raise ValueError("partial groups are disabled in v1.")
        if len({trajectory.group_id for trajectory in trajectories}) != 1:
            raise ValueError("all trajectories in an insert must share the same group_id.")

        insert_ts = utc_ts()
        inserted_group = tuple(replace(trajectory, queue_insert_ts=insert_ts) for trajectory in trajectories)
        dropped_group_ids: list[str] = []

        while len(self.group_order) >= self.capacity_groups:
            dropped_group_ids.append(self.drop_one_group())

        self.group_order.append(group_id)
        self.groups[group_id] = inserted_group
        for trajectory in inserted_group:
            self.actor_counter[trajectory.actor_id] += 1

        return BufferInsertResult(
            inserted_group_id=group_id,
            inserted_trajectories=len(inserted_group),
            dropped_group_ids=tuple(dropped_group_ids),
        )

    def sample_groups(self, *, max_groups: int, learner_policy_version: int) -> GroupedTrajectoryBatch | None:
        if not self.group_order:
            return None
        now_ts = utc_ts()
        self.drop_stale_groups(learner_policy_version=learner_policy_version, now_ts=now_ts)
        if not self.group_order:
            return None

        selected_group_ids: list[str] = []
        selected: list[Trajectory] = []

        while self.group_order and len(selected_group_ids) < max_groups:
            group_id = self.group_order.popleft()
            group = self.groups.pop(group_id)
            selected_group_ids.append(group_id)
            for trajectory in group:
                self.actor_counter[trajectory.actor_id] -= 1
                selected.append(replace(trajectory, learner_consume_ts=now_ts))

        lag_by_trajectory = tuple(max(0, learner_policy_version - trajectory.behavior_policy_version) for trajectory in selected)
        age_ms_by_trajectory = tuple(self.age_ms(trajectory, now_ts) for trajectory in selected)
        freshness_mask = tuple(
            (self.max_policy_lag is None or lag <= self.max_policy_lag)
            and (self.max_age_ms is None or age_ms <= self.max_age_ms)
            for lag, age_ms in zip(lag_by_trajectory, age_ms_by_trajectory)
        )
        staleness = StalenessStats(
            batch_id=make_id("stale"),
            learner_policy_version=learner_policy_version,
            lag_by_trajectory=lag_by_trajectory,
            age_ms_by_trajectory=age_ms_by_trajectory,
            freshness_mask=freshness_mask,
            created_ts=now_ts,
            min_policy_version=min((trajectory.behavior_policy_version for trajectory in selected), default=None),
            max_policy_version=max((trajectory.behavior_policy_version for trajectory in selected), default=None),
            mean_lag=sum(lag_by_trajectory) / len(lag_by_trajectory) if lag_by_trajectory else 0.0,
            max_lag=max(lag_by_trajectory, default=0),
        )
        group_sizes = {group_id: self.required_group_size for group_id in selected_group_ids}
        return GroupedTrajectoryBatch(
            batch_id=make_id("batch"),
            trajectories=tuple(selected),
            group_sizes=group_sizes,
            created_ts=now_ts,
            required_group_size=self.required_group_size,
            group_ids=tuple(selected_group_ids),
            sampled_ts=now_ts,
            allow_partial_groups=False,
            staleness_stats=staleness,
        )

    def drop_stale_groups(self, *, learner_policy_version: int, now_ts: float) -> None:
        keep: deque[str] = deque()
        for group_id in self.group_order:
            group = self.groups[group_id]
            should_drop = any(
                (
                    self.max_policy_lag is not None
                    and learner_policy_version - trajectory.behavior_policy_version > self.max_policy_lag
                )
                or (
                    self.max_age_ms is not None
                    and self.age_ms(trajectory, now_ts) > self.max_age_ms
                )
                for trajectory in group
            )
            if should_drop:
                self.remove_group(group_id)
            else:
                keep.append(group_id)
        self.group_order = keep

    def remove_group(self, group_id: str) -> None:
        group = self.groups.pop(group_id, ())
        for trajectory in group:
            self.actor_counter[trajectory.actor_id] -= 1

    def drop_one_group(self) -> str:
        if self.drop_policy not in {"fifo", "drop_oldest", "drop_most_stale"}:
            raise ValueError(f"unsupported drop policy: {self.drop_policy}")
        group_id = self.group_order.popleft()
        self.remove_group(group_id)
        return group_id

    def age_ms(self, trajectory: Trajectory, now_ts: float) -> float:
        if trajectory.queue_insert_ts is None:
            return 0.0
        return max(0.0, (now_ts - trajectory.queue_insert_ts) * 1000.0)
