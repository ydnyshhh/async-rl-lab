from __future__ import annotations

from dataclasses import replace
import unittest

from async_rl_lab.ids import make_id, utc_ts
from async_rl_lab.models import ToolCall
from async_rl_lab.verifiers import ProgrammaticVerifier, ToolTraceAwareVerifier
from tests.test_buffer import make_trajectory


class VerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_programmatic_verifier_returns_reward(self) -> None:
        verifier = ProgrammaticVerifier(lambda trajectory: 0.5 if trajectory.terminal_reward else 0.0)
        trajectory = make_trajectory("group-a", 0, 0, reward=1.0)

        result = await verifier.verify(trajectory)

        self.assertEqual(result.verifier_status, "ok")
        self.assertEqual(result.reward_value, 0.5)

    async def test_programmatic_verifier_tags_errors(self) -> None:
        def scorer(_trajectory):
            raise ValueError("boom")

        verifier = ProgrammaticVerifier(scorer)
        trajectory = make_trajectory("group-a", 0, 0)

        result = await verifier.verify(trajectory)

        self.assertEqual(result.verifier_status, "error")
        self.assertEqual(result.failure_tag, "ValueError")
        self.assertEqual(result.reward_value, 0.0)

    async def test_tool_trace_aware_verifier_rewards_answer_and_tool_use(self) -> None:
        tool_call = ToolCall(
            tool_call_id=make_id("toolcall"),
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
            created_ts=utc_ts(),
        )
        trajectory = make_trajectory("group-a", 0, 0, answer_text="4", expected_answer="4")
        trajectory = replace(trajectory, tool_calls=(tool_call,))
        verifier = ToolTraceAwareVerifier(required_tool_name="calculator", answer_weight=0.7, tool_weight=0.3)

        result = await verifier.verify(trajectory)

        self.assertEqual(result.verifier_status, "ok")
        self.assertAlmostEqual(result.reward_value, 1.0)
