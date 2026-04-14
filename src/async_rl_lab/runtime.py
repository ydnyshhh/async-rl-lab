from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.environments import EnvironmentSession, TaskSpec
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.inference import InferenceEngine
from async_rl_lab.models import Action, ActorHeartbeat, GenerationRequest, LearnerStepResult, PolicyRef, Trajectory
from async_rl_lab.objectives import Objective
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


async def actor_main_loop(
    *,
    actor_id: str,
    task_stream: Iterable[TaskSpec],
    environment,
    inference_engine: InferenceEngine,
    verifier: Verifier,
    rollout_buffer: InMemoryGroupedRolloutBuffer,
    policy_store: LocalPolicyStore,
    event_logger: JsonlEventLogger,
    stop_event: asyncio.Event,
    config: ActorConfig,
) -> None:
    completed_episodes = 0
    failed_episodes = 0
    task_iterator = iter(task_stream)
    last_heartbeat_ts = 0.0

    while not stop_event.is_set():
        now_ts = utc_ts()
        if now_ts - last_heartbeat_ts >= config.heartbeat_interval_s:
            heartbeat = ActorHeartbeat(
                actor_id=actor_id,
                current_policy_version=policy_store.current_policy().policy_version,
                queue_depth=rollout_buffer.group_count(),
                running_episodes=0,
                completed_episodes=completed_episodes,
                failed_episodes=failed_episodes,
                last_seen_ts=now_ts,
            )
            event_logger.log(
                "ActorHeartbeat",
                actor_id=actor_id,
                policy_version=heartbeat.current_policy_version,
                payload=heartbeat.to_dict(),
            )
            last_heartbeat_ts = now_ts

        try:
            task = next(task_iterator)
        except StopIteration:
            stop_event.set()
            return

        policy_ref = await policy_refresh_logic(
            actor_id=actor_id,
            inference_engine=inference_engine,
            policy_store=policy_store,
            event_logger=event_logger,
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
            payload={"batch_id": batch.batch_id, "group_ids": list(batch.group_ids)},
        )
        prepared = objective.prepare_batch(batch)
        model_logprob_summaries = [sum(trajectory.behavior_logprobs or (-0.1,)) for trajectory in batch.trajectories]
        loss = objective.compute_loss(prepared, model_logprob_summaries=model_logprob_summaries)
        metrics = objective.compute_metrics(prepared)
        learner_step += 1
        published_policy = None
        if learner_step % config.publish_every_steps == 0:
            published_policy = await policy_store.publish_policy(
                checkpoint_step=learner_step,
                policy_tag=f"learner-step-{learner_step}",
                metadata={"loss": loss},
            )
            await inference_engine.refresh_policy(published_policy)
            event_logger.log(
                "PolicyPublished",
                learner_step=learner_step,
                policy_version=published_policy.policy_version,
                payload=published_policy.to_dict(),
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
            loss=loss,
            clipped_fraction=prepared.metrics.get("clipped_fraction"),
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
) -> None:
    while not stop_event.is_set():
        try:
            trajectory = await asyncio.wait_for(pending_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
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


async def policy_refresh_logic(
    *,
    actor_id: str,
    inference_engine: InferenceEngine,
    policy_store: LocalPolicyStore,
    event_logger: JsonlEventLogger,
) -> PolicyRef:
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
