from __future__ import annotations

import unittest

from async_rl_lab.environments import (
    DelayedVerifierEnvironment,
    SingleTurnExactMatchEnvironment,
    TaskSpec,
    ToolUseMultiTurnEnvironment,
)
from async_rl_lab.ids import make_id
from tests.test_buffer import make_policy
from async_rl_lab.models import Action, ToolCall
from async_rl_lab.tools import CalculatorTool, ToolRegistry


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

    async def test_tool_use_environment_routes_tool_and_finishes(self) -> None:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        env = ToolUseMultiTurnEnvironment(registry, max_steps=3)
        task = TaskSpec(
            task_id=make_id("task"),
            prompt_id=make_id("prompt"),
            prompt_text="Use the calculator for 2 + 2 and answer.",
            expected_answer="4.0",
        )
        session = await env.start_episode(task, "actor-0", make_policy(0))
        tool_action = Action(
            action_id=make_id("action"),
            action_type="tool_call",
            raw_text='{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2 + 2"}}',
            parsed_ts=0.0,
            parser_status="ok",
            tool_call=ToolCall(
                tool_call_id=make_id("toolcall"),
                tool_name="calculator",
                arguments={"expression": "2 + 2"},
                created_ts=0.0,
            ),
        )
        tool_transition = await env.step(session, tool_action)
        self.assertFalse(tool_transition.done)
        self.assertEqual(tool_transition.next_observation.role, "tool")

        finish_action = Action(
            action_id=make_id("action"),
            action_type="finish",
            raw_text='{"type":"finish","answer":"4.0"}',
            parsed_ts=0.0,
            parser_status="ok",
            final_text="4.0",
        )
        finish_transition = await env.step(session, finish_action)
        self.assertTrue(finish_transition.done)
        self.assertEqual(finish_transition.reward, 1.0)

    async def test_delayed_verifier_environment_marks_reward_pending(self) -> None:
        env = DelayedVerifierEnvironment(SingleTurnExactMatchEnvironment())
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
        self.assertTrue(transition.verifier_pending)
        self.assertIsNone(transition.reward)
