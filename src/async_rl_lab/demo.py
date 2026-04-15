from __future__ import annotations

import asyncio
from pathlib import Path

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.environments import DelayedVerifierEnvironment, SingleTurnExactMatchEnvironment, TaskSpec
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.ids import make_id
from async_rl_lab.inference import MockInferenceEngine
from async_rl_lab.objectives import StalenessWeightedGRPO
from async_rl_lab.policy_store import LocalPolicyStore
from async_rl_lab.runtime import (
    ActorConfig,
    LearnerConfig,
    PolicyAdoptionController,
    RoundRobinTaskSource,
    actor_main_loop,
    learner_main_loop,
    policy_adoption_loop,
    verified_group_collector_loop,
    verifier_loop,
)
from async_rl_lab.verifiers import DelayedVerifierWrapper, ExactMatchVerifier


async def run_demo() -> None:
    run_id = make_id("run")
    artifact_dir = Path("artifacts") / run_id
    policy_store = LocalPolicyStore(artifact_dir, run_id=run_id)
    inference_engine = MockInferenceEngine(policy_store.current_policy(), policy_store=policy_store)
    adoption_controller = PolicyAdoptionController(
        policy_store.current_policy(),
        adoption_delay_ms=40.0,
        adoption_jitter_ms=20.0,
        actor_skew_step_ms=25.0,
        rollout_fraction=0.75,
        random_seed=7,
    )
    rollout_buffer = InMemoryGroupedRolloutBuffer(
        capacity_groups=8,
        required_group_size=4,
        max_policy_lag=4,
        max_age_ms=30_000.0,
    )
    environment = DelayedVerifierEnvironment(SingleTurnExactMatchEnvironment())
    verifier = DelayedVerifierWrapper(ExactMatchVerifier(), delay_ms=20.0)
    logger = JsonlEventLogger(artifact_dir / "events.jsonl", run_id=run_id)
    tasks = RoundRobinTaskSource(
        [
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 2 + 2?", expected_answer="4", metadata={"expression": "2 + 2"}),
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 3 + 3?", expected_answer="6", metadata={"expression": "3 + 3"}),
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 5 + 7?", expected_answer="12", metadata={"expression": "5 + 7"}),
        ],
        repeat=True,
    )
    stop_event = asyncio.Event()
    pending_verification_queue: asyncio.Queue = asyncio.Queue()
    completed_verification_queue: asyncio.Queue = asyncio.Queue()
    verifier_done_event = asyncio.Event()
    actor_tasks = [
        asyncio.create_task(
        actor_main_loop(
            actor_id=f"actor-{index}",
            task_stream=tasks,
            environment=environment,
            inference_engine=inference_engine,
            verifier=verifier,
            rollout_buffer=rollout_buffer,
            policy_store=policy_store,
            event_logger=logger,
            stop_event=stop_event,
            config=ActorConfig(group_size=4, max_episode_steps=1),
            verifier_pending_queue=pending_verification_queue,
            policy_adoption_controller=adoption_controller,
        )
        )
        for index in range(2)
    ]
    adoption_task = asyncio.create_task(
        policy_adoption_loop(
            controller=adoption_controller,
            inference_engine=inference_engine,
            stop_event=stop_event,
            event_logger=logger,
        )
    )
    verifier_task = asyncio.create_task(
        verifier_loop(
            verifier=verifier,
            pending_queue=pending_verification_queue,
            completed_queue=completed_verification_queue,
            stop_event=stop_event,
            event_logger=logger,
            done_event=verifier_done_event,
        )
    )
    collector_task = asyncio.create_task(
        verified_group_collector_loop(
            completed_queue=completed_verification_queue,
            rollout_buffer=rollout_buffer,
            required_group_size=4,
            stop_event=stop_event,
            event_logger=logger,
            upstream_done_event=verifier_done_event,
        )
    )
    learner_results = await learner_main_loop(
        objective=StalenessWeightedGRPO(),
        rollout_buffer=rollout_buffer,
        policy_store=policy_store,
        inference_engine=inference_engine,
        event_logger=logger,
        stop_event=stop_event,
        config=LearnerConfig(max_groups_per_batch=1, publish_every_steps=1),
        max_steps=4,
        policy_adoption_controller=adoption_controller,
    )
    stop_event.set()
    await asyncio.gather(*actor_tasks)
    await adoption_task
    await verifier_task
    await collector_task
    await inference_engine.close()
    print(f"demo run complete: {run_id}")
    for result in learner_results:
        print(result.to_json_line())


def main() -> None:
    asyncio.run(run_demo())
