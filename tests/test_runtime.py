from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from async_rl_lab.buffer import InMemoryGroupedRolloutBuffer
from async_rl_lab.events import JsonlEventLogger
from async_rl_lab.inference import MockInferenceEngine
from async_rl_lab.objectives import GRPOObjective
from async_rl_lab.policy_store import LocalPolicyStore
from async_rl_lab.runtime import LearnerConfig, learner_main_loop
from tests.test_buffer import make_trajectory


class RuntimeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_learner_drains_buffer_after_stop_when_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_store = LocalPolicyStore(root, run_id="run-test")
            inference_engine = MockInferenceEngine(policy_store.current_policy())
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
            inference_engine = MockInferenceEngine(policy_store.current_policy())
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
