from __future__ import annotations

import asyncio
from pathlib import Path

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.environments import SingleTurnExactMatchEnvironment, TaskSpec
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.ids import make_id
from async_rl_lab.inference import MockInferenceEngine
from async_rl_lab.objectives import StalenessWeightedGRPO
from async_rl_lab.policy_store import LocalPolicyStore
from async_rl_lab.runtime import ActorConfig, LearnerConfig, actor_main_loop, learner_main_loop
from async_rl_lab.verifiers import ExactMatchVerifier


async def run_demo() -> None:
    run_id = make_id("run")
    artifact_dir = Path("artifacts") / run_id
    policy_store = LocalPolicyStore(artifact_dir, run_id=run_id)
    inference_engine = MockInferenceEngine(policy_store.current_policy(), policy_store=policy_store)
    rollout_buffer = InMemoryGroupedRolloutBuffer(
        capacity_groups=8,
        required_group_size=4,
        max_policy_lag=4,
        max_age_ms=30_000.0,
    )
    environment = SingleTurnExactMatchEnvironment()
    verifier = ExactMatchVerifier()
    logger = JsonlEventLogger(artifact_dir / "events.jsonl", run_id=run_id)
    tasks = [
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 2 + 2?", expected_answer="4", metadata={"expression": "2 + 2"}),
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 3 + 3?", expected_answer="6", metadata={"expression": "3 + 3"}),
        TaskSpec(task_id=make_id("task"), prompt_id=make_id("prompt"), prompt_text="What is 5 + 7?", expected_answer="12", metadata={"expression": "5 + 7"}),
    ]
    stop_event = asyncio.Event()
    actor_task = asyncio.create_task(
        actor_main_loop(
            actor_id="actor-0",
            task_stream=tasks,
            environment=environment,
            inference_engine=inference_engine,
            verifier=verifier,
            rollout_buffer=rollout_buffer,
            policy_store=policy_store,
            event_logger=logger,
            stop_event=stop_event,
            config=ActorConfig(group_size=4, max_episode_steps=1),
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
        max_steps=2,
    )
    await actor_task
    print(f"demo run complete: {run_id}")
    for result in learner_results:
        print(result.to_json_line())


def main() -> None:
    asyncio.run(run_demo())
