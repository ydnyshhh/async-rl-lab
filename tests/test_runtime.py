from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.environments import TaskSpec
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.inference import MockInferenceEngine
from async_rl_lab.objectives import GRPOObjective
from async_rl_lab.policy_store import LocalPolicyStore
from async_rl_lab.runtime import (
    LearnerConfig,
    RoundRobinTaskSource,
    PendingVerifiedGroupAssembler,
    learner_main_loop,
    verified_group_collector_loop,
    verifier_loop,
)
from async_rl_lab.verifiers import DelayedVerifierWrapper, ExactMatchVerifier
from tests.test_buffer import make_trajectory


class RuntimeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_learner_drains_buffer_after_stop_when_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            inference_engine = MockInferenceEngine(policy_store.current_policy(), policy_store=policy_store)
            buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
            buffer.insert_group((make_trajectory("group-a", 0, 0), make_trajectory("group-a", 1, 0)))
            logger = JsonlEventLogger(root / "events.jsonl", run_id="run-test")
            stop_event = asyncio.Event()
            stop_event.set()

            results = await learner_main_loop(
                objective=GRPOObjective(),
                rollout_buffer=buffer,
                policy_store=policy_store,
                inference_engine=inference_engine,
                event_logger=logger,
                stop_event=stop_event,
                config=LearnerConfig(max_groups_per_batch=1, publish_every_steps=1),
                max_steps=None,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(buffer.group_count(), 0)

    async def test_learner_max_steps_does_not_force_global_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            inference_engine = MockInferenceEngine(policy_store.current_policy(), policy_store=policy_store)
            buffer = InMemoryGroupedRolloutBuffer(capacity_groups=4, required_group_size=2)
            buffer.insert_group((make_trajectory("group-a", 0, 0), make_trajectory("group-a", 1, 0)))
            buffer.insert_group((make_trajectory("group-b", 0, 0), make_trajectory("group-b", 1, 0)))
            logger = JsonlEventLogger(root / "events.jsonl", run_id="run-test")
            stop_event = asyncio.Event()

            results = await learner_main_loop(
                objective=GRPOObjective(),
                rollout_buffer=buffer,
                policy_store=policy_store,
                inference_engine=inference_engine,
                event_logger=logger,
                stop_event=stop_event,
                config=LearnerConfig(max_groups_per_batch=1, publish_every_steps=1),
                max_steps=1,
            )

            self.assertEqual(len(results), 1)
            self.assertFalse(stop_event.is_set())
            self.assertEqual(buffer.group_count(), 1)

    async def test_learner_updates_published_policy_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            inference_engine = MockInferenceEngine(policy_store.current_policy(), policy_store=policy_store)
            buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
            buffer.insert_group(
                (
                    make_trajectory("group-a", 0, 0, reward=1.0, answer_text="4"),
                    make_trajectory("group-a", 1, 0, reward=0.0, answer_text="5"),
                )
            )
            logger = JsonlEventLogger(root / "events.jsonl", run_id="run-test")
            stop_event = asyncio.Event()
            stop_event.set()

            results = await learner_main_loop(
                objective=GRPOObjective(),
                rollout_buffer=buffer,
                policy_store=policy_store,
                inference_engine=inference_engine,
                event_logger=logger,
                stop_event=stop_event,
                config=LearnerConfig(max_groups_per_batch=1, publish_every_steps=1, learning_rate=0.1),
                max_steps=None,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(policy_store.current_policy().policy_version, 1)
            self.assertGreater(results[0].gradient_norm or 0.0, 0.0)
            self.assertGreater(len(policy_store.current_state_snapshot().token_logits), 0)

    async def test_verifier_loop_and_collector_insert_completed_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pending_queue: asyncio.Queue = asyncio.Queue()
            completed_queue: asyncio.Queue = asyncio.Queue()
            buffer = InMemoryGroupedRolloutBuffer(capacity_groups=2, required_group_size=2)
            logger = JsonlEventLogger(root / "events.jsonl", run_id="run-test")
            stop_event = asyncio.Event()
            verifier_task = asyncio.create_task(
                verifier_loop(
                    verifier=DelayedVerifierWrapper(ExactMatchVerifier(), delay_ms=1.0),
                    pending_queue=pending_queue,
                    completed_queue=completed_queue,
                    stop_event=stop_event,
                    event_logger=logger,
                )
            )
            collector_task = asyncio.create_task(
                verified_group_collector_loop(
                    completed_queue=completed_queue,
                    rollout_buffer=buffer,
                    required_group_size=2,
                    stop_event=stop_event,
                    event_logger=logger,
                )
            )
            await pending_queue.put(make_trajectory("group-a", 0, 0, answer_text="4", expected_answer="4"))
            await pending_queue.put(make_trajectory("group-a", 1, 0, answer_text="5", expected_answer="4", reward=0.0))
            await asyncio.sleep(0.2)
            stop_event.set()
            await verifier_task
            await collector_task

            self.assertEqual(buffer.group_count(), 1)
            batch = buffer.sample_groups(max_groups=1, learner_policy_version=0)
            self.assertEqual(len(batch.trajectories), 2)
            self.assertEqual(tuple(t.terminal_reward for t in batch.trajectories), (1.0, 0.0))

    async def test_round_robin_task_source_repeats(self) -> None:
        task_source = RoundRobinTaskSource(
            (
                TaskSpec(
                    task_id="task-0",
                    prompt_id="prompt-0",
                    prompt_text="prompt-0",
                ),
                TaskSpec(
                    task_id="task-1",
                    prompt_id="prompt-1",
                    prompt_text="prompt-1",
                ),
            ),
            repeat=True,
        )
        seen = [await task_source.next_task() for _ in range(5)]
        self.assertEqual([task.task_id for task in seen], ["task-0", "task-1", "task-0", "task-1", "task-0"])

    def test_pending_group_assembler_waits_for_full_group(self) -> None:
        assembler = PendingVerifiedGroupAssembler(required_group_size=2)
        first = assembler.add(make_trajectory("group-a", 0, 0))
        second = assembler.add(make_trajectory("group-a", 1, 0))
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(tuple(t.sample_index_within_group for t in second), (0, 1))
