from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace
import random
from typing import Protocol

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.environments import EnvironmentSession, TaskSpec
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.inference import InferenceEngine
from async_rl_lab.models import Action, ActorHeartbeat, GenerationRequest, LearnerStepResult, PolicyRef, Trajectory
from async_rl_lab.objectives import Objective, RescoredBatch
from async_rl_lab.policy_store import LocalPolicyStore
from async_rl_lab.verifiers import Verifier


@dataclass(slots=True)
class ActorConfig:
    group_size: int = 4
    max_episode_steps: int = 4
    heartbeat_interval_s: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95


@dataclass(slots=True)
class LearnerConfig:
    max_groups_per_batch: int = 2
    publish_every_steps: int = 1
    learning_rate: float = 0.05


class PolicyAdoptionController:
    def __init__(
        self,
        initial_policy: PolicyRef,
        *,
        adoption_delay_ms: float = 0.0,
        adoption_jitter_ms: float = 0.0,
    ) -> None:
        self.latest_published = initial_policy
        self.latest_adopted = initial_policy
        self.adoption_delay_ms = adoption_delay_ms
        self.adoption_jitter_ms = adoption_jitter_ms
        self.pending_queue: asyncio.Queue[PolicyRef] = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def note_published(self, policy: PolicyRef) -> None:
        async with self.lock:
            self.latest_published = policy
        await self.pending_queue.put(policy)

    async def mark_adopted(self, policy: PolicyRef) -> None:
        async with self.lock:
            self.latest_adopted = policy

    def current_published_policy(self) -> PolicyRef:
        return self.latest_published

    def current_adopted_policy(self) -> PolicyRef:
        return self.latest_adopted

    def adoption_gap(self) -> int:
        return self.latest_published.policy_version - self.latest_adopted.policy_version


class TaskSource(Protocol):
    async def next_task(self) -> TaskSpec | None:
        ...


class RoundRobinTaskSource:
    def __init__(self, tasks: Iterable[TaskSpec], *, repeat: bool = True) -> None:
        self.tasks = tuple(tasks)
        self.repeat = repeat
        self.lock = asyncio.Lock()
        self.index = 0

    async def next_task(self) -> TaskSpec | None:
        async with self.lock:
            if not self.tasks:
                return None
            if not self.repeat and self.index >= len(self.tasks):
                return None
            task = self.tasks[self.index % len(self.tasks)]
            self.index += 1
            return task


class PendingVerifiedGroupAssembler:
    def __init__(self, required_group_size: int) -> None:
        self.required_group_size = required_group_size
        self.pending: dict[str, dict[int, Trajectory]] = {}

    def add(self, trajectory: Trajectory) -> tuple[Trajectory, ...] | None:
        entries = self.pending.setdefault(trajectory.group_id, {})
        entries[trajectory.sample_index_within_group] = trajectory
        if len(entries) < self.required_group_size:
            return None
        ordered = tuple(entries[index] for index in sorted(entries))
        del self.pending[trajectory.group_id]
        return ordered

    def pending_group_count(self) -> int:
        return len(self.pending)

    def pending_sample_count(self) -> int:
        return sum(len(entries) for entries in self.pending.values())


async def actor_main_loop(
    *,
    actor_id: str,
    task_stream: Iterable[TaskSpec] | TaskSource,
    environment,
    inference_engine: InferenceEngine,
    verifier: Verifier,
    rollout_buffer: InMemoryGroupedRolloutBuffer,
    policy_store: LocalPolicyStore,
    event_logger: JsonlEventLogger,
    stop_event: asyncio.Event,
    config: ActorConfig,
    verifier_pending_queue: asyncio.Queue[Trajectory] | None = None,
    policy_adoption_controller: PolicyAdoptionController | None = None,
) -> None:
    completed_episodes = 0
    failed_episodes = 0
    task_iterator = iter(task_stream) if not hasattr(task_stream, "next_task") else None
    last_heartbeat_ts = 0.0

    while not stop_event.is_set():
        now_ts = utc_ts()
        inference_metrics = inference_engine.metrics_snapshot()
        generation_latencies = inference_metrics.histograms.get("inference.latency_ms", ())
        queue_waits = inference_metrics.histograms.get("inference.queue_wait_ms", ())
        served_policy_version = inference_engine.current_policy().policy_version
        published_policy_version = (
            policy_adoption_controller.current_published_policy().policy_version
            if policy_adoption_controller is not None
            else policy_store.current_policy().policy_version
        )
        if now_ts - last_heartbeat_ts >= config.heartbeat_interval_s:
            heartbeat = ActorHeartbeat(
                actor_id=actor_id,
                current_policy_version=served_policy_version,
                queue_depth=rollout_buffer.group_count(),
                running_episodes=0,
                completed_episodes=completed_episodes,
                failed_episodes=failed_episodes,
                last_seen_ts=now_ts,
                mean_generation_latency_ms=mean_or_none(generation_latencies),
                mean_verifier_latency_ms=None,
                metadata={
                    "published_policy_version": published_policy_version,
                    "policy_version_gap": published_policy_version - served_policy_version,
                    "mean_queue_wait_ms": mean_or_none(queue_waits),
                },
            )
            event_logger.log(
                "ActorHeartbeat",
                actor_id=actor_id,
                policy_version=heartbeat.current_policy_version,
                payload=heartbeat.to_dict(),
            )
            last_heartbeat_ts = now_ts

        task = await next_task_from_source(task_stream, task_iterator)
        if task is None:
            return

        policy_ref = await policy_refresh_logic(
            actor_id=actor_id,
            inference_engine=inference_engine,
            policy_store=policy_store,
            event_logger=event_logger,
            policy_adoption_controller=policy_adoption_controller,
        )
        group_id = make_id("group")
        group_trajectories: list[Trajectory] = []

        for sample_index in range(config.group_size):
            try:
                trajectory = await execute_episode(
                    actor_id=actor_id,
                    task=task,
                    group_id=group_id,
                    sample_index=sample_index,
                    environment=environment,
                    inference_engine=inference_engine,
                    verifier=verifier,
                    policy_ref=policy_ref,
                    event_logger=event_logger,
                    config=config,
                    inline_verify=verifier_pending_queue is None,
                )
                group_trajectories.append(trajectory)
                completed_episodes += 1
            except Exception as exc:
                failed_episodes += 1
                event_logger.log(
                    "ActorEpisodeFailed",
                    actor_id=actor_id,
                    policy_version=policy_ref.policy_version,
                    group_id=group_id,
                    payload={"task_id": task.task_id, "error": str(exc)},
                )
                break

        if len(group_trajectories) == config.group_size:
            if verifier_pending_queue is None:
                insert_result = rollout_buffer.insert_group(tuple(group_trajectories))
                event_logger.log(
                    "TrajectoryQueued",
                    actor_id=actor_id,
                    policy_version=policy_ref.policy_version,
                    group_id=group_id,
                    payload={
                        "inserted_group_id": insert_result.inserted_group_id,
                        "inserted_trajectories": insert_result.inserted_trajectories,
                        "dropped_group_ids": list(insert_result.dropped_group_ids),
                    },
                )
            else:
                for trajectory in group_trajectories:
                    await verifier_pending_queue.put(trajectory)
                event_logger.log(
                    "TrajectoryQueuedForVerification",
                    actor_id=actor_id,
                    policy_version=policy_ref.policy_version,
                    group_id=group_id,
                    payload={"queued_trajectories": len(group_trajectories)},
                )


async def execute_episode(
    *,
    actor_id: str,
    task: TaskSpec,
    group_id: str,
    sample_index: int,
    environment,
    inference_engine: InferenceEngine,
    verifier: Verifier,
    policy_ref: PolicyRef,
    event_logger: JsonlEventLogger,
    config: ActorConfig,
    inline_verify: bool = True,
) -> Trajectory:
    session: EnvironmentSession = await environment.start_episode(task, actor_id, policy_ref)
    event_logger.log(
        "ActorEpisodeStarted",
        actor_id=actor_id,
        policy_version=policy_ref.policy_version,
        group_id=group_id,
        payload={"task_id": task.task_id, "prompt_id": task.prompt_id, "sample_index": sample_index},
    )
    generated_text_parts: list[str] = []
    generation_results = []

    while not session.done and len(session.transitions) < config.max_episode_steps:
        observation = session.observations[-1]
        request = GenerationRequest(
            request_id=make_id("req"),
            task_id=task.task_id,
            prompt_id=task.prompt_id,
            group_id=group_id,
            sample_index_within_group=sample_index,
            actor_id=actor_id,
            policy=policy_ref,
            observation_text=observation.text,
            available_tools=observation.available_tools,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            created_ts=utc_ts(),
            metadata={"gold_answer": task.expected_answer or "", "expression": task.metadata.get("expression", "")},
        )
        generation = await inference_engine.submit(request)
        generation_results.append(generation)
        generated_text_parts.append(generation.generated_text)
        action = generation.action or Action(
            action_id=make_id("action"),
            action_type="invalid",
            raw_text=generation.generated_text,
            parsed_ts=utc_ts(),
            parser_status="malformed",
        )
        await environment.step(session, action)

    episode = environment.build_episode(session)
    trajectory = build_trajectory(
        session=session,
        episode=episode,
        group_id=group_id,
        sample_index=sample_index,
        generation_results=generation_results,
        generated_text=" ".join(generated_text_parts),
    )
    if not inline_verify:
        event_logger.log(
            "ActorEpisodeFinished",
            actor_id=actor_id,
            policy_version=policy_ref.policy_version,
            group_id=group_id,
            trajectory_id=trajectory.trajectory_id,
            payload={"task_id": task.task_id, "reward_pending": True, "env_steps": trajectory.env_steps},
        )
        return trajectory
    verifier_start_ts = utc_ts()
    reward_result = await verifier.verify(replace(trajectory, verifier_start_ts=verifier_start_ts))
    verifier_end_ts = utc_ts()
    completed = replace(
        trajectory,
        verifier_start_ts=verifier_start_ts,
        verifier_end_ts=verifier_end_ts,
        verifier_latency_ms=(verifier_end_ts - verifier_start_ts) * 1000.0,
        reward_result=reward_result,
        terminal_reward=reward_result.reward_value,
    )
    event_logger.log(
        "ActorEpisodeFinished",
        actor_id=actor_id,
        policy_version=policy_ref.policy_version,
        group_id=group_id,
        trajectory_id=completed.trajectory_id,
        payload={"task_id": task.task_id, "reward": reward_result.reward_value, "env_steps": completed.env_steps},
    )
    return completed


def build_trajectory(
    *,
    session: EnvironmentSession,
    episode,
    group_id: str,
    sample_index: int,
    generation_results,
    generated_text: str,
) -> Trajectory:
    prompt_tokens = sum(result.prompt_tokens for result in generation_results)
    completion_tokens = sum(result.completion_tokens for result in generation_results)
    tool_calls = tuple(tool_call for tool_call in session.tool_calls)
    tool_results = tuple(tool_result for tool_result in session.tool_results)
    parsed_actions = tuple(transition.action for transition in episode.transitions)
    observations = tuple(session.observations)
    per_step_rewards = tuple(transition.reward or 0.0 for transition in episode.transitions)
    behavior_token_logprobs = tuple(result.token_logprobs or () for result in generation_results)
    behavior_logprobs = tuple(
        sum(result.token_logprobs or ()) / len(result.token_logprobs) if result.token_logprobs else -0.1
        for result in generation_results
    )
    return Trajectory(
        trajectory_id=make_id("traj"),
        task_id=episode.task_id,
        prompt_id=episode.prompt_id,
        group_id=group_id,
        sample_index_within_group=sample_index,
        behavior_policy_version=episode.behavior_policy_version,
        policy_ref=episode.policy_ref,
        policy_checksum_or_ref=episode.policy_ref.policy_tag,
        actor_id=episode.actor_id,
        episode_start_ts=episode.started_ts,
        episode_end_ts=episode.ended_ts,
        env_steps=len(episode.transitions),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        raw_text=generated_text,
        termination_reason=episode.termination_reason or "unknown",
        transitions=episode.transitions,
        parsed_action_trace=parsed_actions,
        observations=observations,
        tool_calls=tool_calls,
        tool_results=tool_results,
        per_step_rewards=per_step_rewards,
        metadata=episode.metadata,
        is_truncated=episode.is_truncated,
        behavior_logprobs=behavior_logprobs,
        behavior_token_logprobs=behavior_token_logprobs,
        logprob_stats={"mean_behavior_logprob": sum(behavior_logprobs) / len(behavior_logprobs) if behavior_logprobs else 0.0},
        invalid_action_count=sum(1 for action in parsed_actions if action.action_type == "invalid"),
        generation_latency_ms=sum(result.latency_ms for result in generation_results),
        queue_wait_ms=sum(result.queue_wait_ms for result in generation_results),
    )


async def learner_main_loop(
    *,
    objective: Objective,
    rollout_buffer: InMemoryGroupedRolloutBuffer,
    policy_store: LocalPolicyStore,
    inference_engine: InferenceEngine,
    event_logger: JsonlEventLogger,
    stop_event: asyncio.Event,
    config: LearnerConfig,
    max_steps: int | None = None,
    policy_adoption_controller: PolicyAdoptionController | None = None,
) -> list[LearnerStepResult]:
    results: list[LearnerStepResult] = []
    learner_step = 0
    while True:
        if max_steps is not None and learner_step >= max_steps:
            break
        batch = rollout_buffer.sample_groups(
            max_groups=config.max_groups_per_batch,
            learner_policy_version=policy_store.current_policy().policy_version,
        )
        if batch is None:
            if stop_event.is_set() and rollout_buffer.group_count() == 0:
                break
            await asyncio.sleep(0.05)
            continue

        step_start_ts = utc_ts()
        event_logger.log(
            "LearnerBatchBuilt",
            learner_step=learner_step,
            policy_version=policy_store.current_policy().policy_version,
            payload={
                "batch_id": batch.batch_id,
                "group_ids": list(batch.group_ids),
                "dropped_stale_groups": batch.staleness_stats.dropped_for_staleness if batch.staleness_stats else 0,
                "drop_policy": rollout_buffer.drop_policy,
            },
        )
        prepared = objective.prepare_batch(batch)
        scoring_policy = policy_store.current_policy()
        current_scores = policy_store.score_many(
            [trajectory.raw_text for trajectory in batch.trajectories],
            policy_version=scoring_policy.policy_version,
        )
        reference_policy_version = scoring_policy.parent_policy_version or scoring_policy.policy_version
        reference_scores = policy_store.score_many(
            [trajectory.raw_text for trajectory in batch.trajectories],
            policy_version=reference_policy_version,
        )
        rescored = RescoredBatch(
            current_policy_version=scoring_policy.policy_version,
            reference_policy_version=reference_policy_version,
            current_token_logprobs=tuple(score.token_logprobs for score in current_scores),
            current_sequence_logprobs=tuple(score.total_logprob for score in current_scores),
            reference_token_logprobs=tuple(score.token_logprobs for score in reference_scores),
            reference_sequence_logprobs=tuple(score.total_logprob for score in reference_scores),
        )
        objective_result = objective.compute_loss(prepared, rescored)
        update_stats = policy_store.train_on_sequences(
            policy_version=scoring_policy.policy_version,
            sequences=[trajectory.raw_text for trajectory in batch.trajectories],
            advantages=objective_result.update_advantages,
            sequence_weights=tuple(1.0 for _ in batch.trajectories),
            learning_rate=config.learning_rate,
        )
        metrics = {
            **objective.compute_metrics(prepared, rescored, objective_result),
            "mean_current_logprob": update_stats.mean_current_logprob,
            "mean_behavior_logprob": mean_behavior_logprob(batch.trajectories),
            "mean_rollout_age_ms": mean_or_zero(batch.staleness_stats.age_ms_by_trajectory if batch.staleness_stats else ()),
            "oldest_rollout_age_ms": max(batch.staleness_stats.age_ms_by_trajectory, default=0.0)
            if batch.staleness_stats
            else 0.0,
            "mean_policy_lag": batch.staleness_stats.mean_lag if batch.staleness_stats else 0.0,
            "max_policy_lag": float(batch.staleness_stats.max_lag) if batch.staleness_stats else 0.0,
            "dropped_stale_trajectories": float(batch.staleness_stats.dropped_for_staleness)
            if batch.staleness_stats
            else 0.0,
            "mean_queue_wait_ms": mean_or_zero(
                tuple(trajectory.queue_wait_ms or 0.0 for trajectory in batch.trajectories)
            ),
            "mean_staleness_weight": mean_or_zero(prepared.sequence_weights),
            "reference_policy_version": float(reference_policy_version),
        }
        learner_step += 1
        published_policy = None
        if learner_step % config.publish_every_steps == 0:
            published_policy = await policy_store.publish_policy(
                checkpoint_step=learner_step,
                policy_tag=f"learner-step-{learner_step}",
                metadata={
                    "loss": objective_result.loss,
                    "gradient_norm": update_stats.gradient_norm,
                    "mean_current_logprob": update_stats.mean_current_logprob,
                },
                state=update_stats.updated_state,
            )
            if policy_adoption_controller is None:
                await inference_engine.refresh_policy(published_policy)
            else:
                await policy_adoption_controller.note_published(published_policy)
            event_logger.log(
                "PolicyPublished",
                learner_step=learner_step,
                policy_version=published_policy.policy_version,
                payload={
                    **published_policy.to_dict(),
                    "adoption_mode": "async" if policy_adoption_controller is not None else "inline",
                },
            )
        step_end_ts = utc_ts()
        result = LearnerStepResult(
            learner_step=learner_step,
            consumed_batch_id=batch.batch_id,
            consumed_groups=len(batch.group_ids),
            consumed_trajectories=len(batch.trajectories),
            learner_policy_version=policy_store.current_policy().policy_version,
            objective_name=type(objective).__name__,
            step_start_ts=step_start_ts,
            step_end_ts=step_end_ts,
            mean_reward=metrics.get("mean_reward", 0.0),
            mean_advantage=metrics.get("mean_advantage"),
            loss=objective_result.loss,
            gradient_norm=update_stats.gradient_norm,
            clipped_fraction=metrics.get("clip_fraction"),
            stale_fraction=(
                1.0
                - (
                    sum(1.0 for flag in batch.staleness_stats.freshness_mask if flag)
                    / len(batch.staleness_stats.freshness_mask)
                )
                if batch.staleness_stats and batch.staleness_stats.freshness_mask
                else 0.0
            ),
            metrics=metrics,
            published_policy=published_policy,
            metadata={
                "published_policy_version": published_policy.policy_version if published_policy is not None else None,
                "adopted_policy_version": (
                    policy_adoption_controller.current_adopted_policy().policy_version
                    if policy_adoption_controller is not None
                    else inference_engine.current_policy().policy_version
                ),
            },
        )
        results.append(result)
        event_logger.log(
            "LearnerStepCompleted",
            learner_step=learner_step,
            policy_version=result.learner_policy_version,
            payload=result.to_dict(),
        )

    return results


async def verifier_loop(
    *,
    verifier: Verifier,
    pending_queue: asyncio.Queue[Trajectory],
    completed_queue: asyncio.Queue[Trajectory],
    stop_event: asyncio.Event,
    event_logger: JsonlEventLogger,
    done_event: asyncio.Event | None = None,
) -> None:
    try:
        while True:
            try:
                trajectory = await asyncio.wait_for(pending_queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if stop_event.is_set() and pending_queue.empty():
                    break
                continue
            try:
                event_logger.log(
                    "VerifierStarted",
                    trajectory_id=trajectory.trajectory_id,
                    group_id=trajectory.group_id,
                    payload={"task_id": trajectory.task_id},
                )
                verifier_start_ts = utc_ts()
                reward_result = await verifier.verify(trajectory)
                verifier_end_ts = utc_ts()
                completed = replace(
                    trajectory,
                    verifier_start_ts=verifier_start_ts,
                    verifier_end_ts=verifier_end_ts,
                    verifier_latency_ms=(verifier_end_ts - verifier_start_ts) * 1000.0,
                    terminal_reward=reward_result.reward_value,
                    reward_result=reward_result,
                )
                await completed_queue.put(completed)
                event_logger.log(
                    "VerifierFinished",
                    trajectory_id=trajectory.trajectory_id,
                    group_id=trajectory.group_id,
                    payload={"reward": reward_result.reward_value},
                )
            finally:
                pending_queue.task_done()
    finally:
        if done_event is not None:
            done_event.set()


async def verified_group_collector_loop(
    *,
    completed_queue: asyncio.Queue[Trajectory],
    rollout_buffer: InMemoryGroupedRolloutBuffer,
    required_group_size: int,
    stop_event: asyncio.Event,
    event_logger: JsonlEventLogger,
    upstream_done_event: asyncio.Event | None = None,
) -> None:
    assembler = PendingVerifiedGroupAssembler(required_group_size)
    while True:
        try:
            trajectory = await asyncio.wait_for(completed_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            if (
                stop_event.is_set()
                and completed_queue.empty()
                and assembler.pending_group_count() == 0
                and (upstream_done_event is None or upstream_done_event.is_set())
            ):
                break
            continue

        try:
            maybe_group = assembler.add(trajectory)
            event_logger.log(
                "VerifiedTrajectoryCollected",
                trajectory_id=trajectory.trajectory_id,
                group_id=trajectory.group_id,
                payload={
                    "pending_groups": assembler.pending_group_count(),
                    "pending_samples": assembler.pending_sample_count(),
                },
            )
            if maybe_group is None:
                continue

            insert_result = rollout_buffer.insert_group(maybe_group)
            event_logger.log(
                "TrajectoryQueued",
                actor_id=maybe_group[0].actor_id,
                policy_version=maybe_group[0].behavior_policy_version,
                group_id=maybe_group[0].group_id,
                payload={
                    "inserted_group_id": insert_result.inserted_group_id,
                    "inserted_trajectories": insert_result.inserted_trajectories,
                    "dropped_group_ids": list(insert_result.dropped_group_ids),
                    "source": "verified_group_collector_loop",
                },
            )
        finally:
            completed_queue.task_done()


async def policy_adoption_loop(
    *,
    controller: PolicyAdoptionController,
    inference_engine: InferenceEngine,
    stop_event: asyncio.Event,
    event_logger: JsonlEventLogger,
) -> None:
    while True:
        try:
            policy = await asyncio.wait_for(controller.pending_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            if stop_event.is_set() and controller.pending_queue.empty():
                break
            continue

        try:
            adoption_delay_ms = controller.adoption_delay_ms
            if controller.adoption_jitter_ms > 0.0:
                adoption_delay_ms += random.uniform(0.0, controller.adoption_jitter_ms)
            if adoption_delay_ms > 0.0:
                await asyncio.sleep(adoption_delay_ms / 1000.0)
            await inference_engine.refresh_policy(policy)
            await controller.mark_adopted(policy)
            event_logger.log(
                "PolicyAdopted",
                policy_version=policy.policy_version,
                payload={
                    "policy_tag": policy.policy_tag,
                    "published_policy_version": controller.current_published_policy().policy_version,
                    "adopted_policy_version": controller.current_adopted_policy().policy_version,
                    "adoption_delay_ms": adoption_delay_ms,
                },
            )
        finally:
            controller.pending_queue.task_done()


async def policy_refresh_logic(
    *,
    actor_id: str,
    inference_engine: InferenceEngine,
    policy_store: LocalPolicyStore,
    event_logger: JsonlEventLogger,
    policy_adoption_controller: PolicyAdoptionController | None = None,
) -> PolicyRef:
    if policy_adoption_controller is not None:
        adopted = policy_adoption_controller.current_adopted_policy()
        published = policy_adoption_controller.current_published_policy()
        if published.policy_version != adopted.policy_version:
            event_logger.log(
                "PolicyAdoptionLagObserved",
                actor_id=actor_id,
                policy_version=adopted.policy_version,
                payload={
                    "published_policy_version": published.policy_version,
                    "adopted_policy_version": adopted.policy_version,
                    "adoption_gap": published.policy_version - adopted.policy_version,
                },
            )
        return adopted
    latest = policy_store.current_policy()
    if inference_engine.current_policy().policy_version != latest.policy_version:
        await inference_engine.refresh_policy(latest)
        event_logger.log(
            "PolicyRefreshed",
            actor_id=actor_id,
            policy_version=latest.policy_version,
            payload={"policy_tag": latest.policy_tag},
        )
    return latest


def mean_or_none(values) -> float | None:
    values_tuple = tuple(values)
    if not values_tuple:
        return None
    return sum(values_tuple) / len(values_tuple)


def mean_or_zero(values) -> float:
    result = mean_or_none(values)
    return result if result is not None else 0.0


def mean_behavior_logprob(trajectories: tuple[Trajectory, ...]) -> float:
    summaries = []
    for trajectory in trajectories:
        if trajectory.behavior_logprobs:
            summaries.append(sum(trajectory.behavior_logprobs) / len(trajectory.behavior_logprobs))
    return sum(summaries) / len(summaries) if summaries else 0.0


async def next_task_from_source(
    task_stream: Iterable[TaskSpec] | TaskSource,
    task_iterator,
) -> TaskSpec | None:
    if hasattr(task_stream, "next_task"):
        return await task_stream.next_task()
    if task_iterator is None:
        return None
    try:
        return next(task_iterator)
    except StopIteration:
        return None
