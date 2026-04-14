from __future__ import annotations

import unittest

from async_rl_lab.environments import SingleTurnExactMatchEnvironment, TaskSpec
from async_rl_lab.ids import make_id
from tests.test_buffer import make_policy
from async_rl_lab.models import Action


class RuntimeEnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_turn_env_is_deterministic(self) -> None:
        env = SingleTurnExactMatchEnvironment()
        task = TaskSpec(
            task_id=make_id("task"),
            prompt_id=make_id("prompt"),
            prompt_text="What is 2 + 2?",
            expected_answer="4",
        )
        session = await env.start_episode(task, "actor-0", make_policy(0))
        action = Action(
            action_id=make_id("action"),
            action_type="finish",
            raw_text='{"type":"finish","answer":"4"}',
            parsed_ts=0.0,
            parser_status="ok",
            final_text="4",
        )
        transition = await env.step(session, action)
        self.assertTrue(transition.done)
        self.assertEqual(transition.reward, 1.0)
