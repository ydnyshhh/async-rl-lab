from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Callable, Protocol

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import RewardResult, Trajectory


class Verifier(Protocol):
    name: str

    async def verify(self, trajectory: Trajectory) -> RewardResult:
        ...

    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        ...


class ExactMatchVerifier:
    name = "exact_match"

    async def verify(self, trajectory: Trajectory) -> RewardResult:
        started_ts = utc_ts()
        expected = str(trajectory.metadata.get("expected_answer", ""))
        predicted = ""
        for action in trajectory.parsed_action_trace:
            if action.final_text:
                predicted = action.final_text
        reward = 1.0 if predicted.strip() == expected.strip() else 0.0
        ended_ts = utc_ts()
        return RewardResult(
            reward_id=make_id("reward"),
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            prompt_id=trajectory.prompt_id,
            group_id=trajectory.group_id,
            sample_index_within_group=trajectory.sample_index_within_group,
            reward_value=reward,
            verifier_name=self.name,
            verifier_status="ok",
            started_ts=started_ts,
            ended_ts=ended_ts,
            reward_components={"exact_match": reward},
        )

    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        return [await self.verify(trajectory) for trajectory in trajectories]


class ProgrammaticVerifier:
    name = "programmatic"

    def __init__(self, scorer: Callable[[Trajectory], float]) -> None:
        self.scorer = scorer

    async def verify(self, trajectory: Trajectory) -> RewardResult:
        started_ts = utc_ts()
        failure_tag = None
        try:
            reward = float(self.scorer(trajectory))
            status = "ok"
        except Exception as exc:
            reward = 0.0
            status = "error"
            failure_tag = type(exc).__name__
        ended_ts = utc_ts()
        return RewardResult(
            reward_id=make_id("reward"),
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            prompt_id=trajectory.prompt_id,
            group_id=trajectory.group_id,
            sample_index_within_group=trajectory.sample_index_within_group,
            reward_value=reward,
            verifier_name=self.name,
            verifier_status=status,
            started_ts=started_ts,
            ended_ts=ended_ts,
            reward_components={"programmatic": reward},
            failure_tag=failure_tag,
        )

    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        return [await self.verify(trajectory) for trajectory in trajectories]


class ToolTraceAwareVerifier:
    name = "tool_trace_aware"

    def __init__(self, *, required_tool_name: str, answer_weight: float = 0.7, tool_weight: float = 0.3) -> None:
        self.required_tool_name = required_tool_name
        self.answer_weight = answer_weight
        self.tool_weight = tool_weight

    async def verify(self, trajectory: Trajectory) -> RewardResult:
        started_ts = utc_ts()
        expected = str(trajectory.metadata.get("expected_answer", ""))
        predicted = ""
        for action in trajectory.parsed_action_trace:
            if action.final_text:
                predicted = action.final_text
        answer_score = 1.0 if predicted.strip() == expected.strip() else 0.0
        used_tool = any(tool_call.tool_name == self.required_tool_name for tool_call in trajectory.tool_calls)
        tool_score = 1.0 if used_tool else 0.0
        reward = self.answer_weight * answer_score + self.tool_weight * tool_score
        ended_ts = utc_ts()
        return RewardResult(
            reward_id=make_id("reward"),
            trajectory_id=trajectory.trajectory_id,
            task_id=trajectory.task_id,
            prompt_id=trajectory.prompt_id,
            group_id=trajectory.group_id,
            sample_index_within_group=trajectory.sample_index_within_group,
            reward_value=reward,
            verifier_name=self.name,
            verifier_status="ok",
            started_ts=started_ts,
            ended_ts=ended_ts,
            reward_components={"answer": answer_score, "tool_usage": tool_score},
        )

    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        return [await self.verify(trajectory) for trajectory in trajectories]


class DelayedVerifierWrapper:
    name = "delayed"

    def __init__(self, inner: Verifier, *, delay_ms: float = 50.0) -> None:
        self.inner = inner
        self.delay_ms = delay_ms

    async def verify(self, trajectory: Trajectory) -> RewardResult:
        await asyncio.sleep(self.delay_ms / 1000.0)
        return await self.inner.verify(trajectory)

    async def verify_many(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        results: list[RewardResult] = []
        for trajectory in trajectories:
            results.append(await self.verify(trajectory))
        return results
