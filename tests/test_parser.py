from __future__ import annotations

import unittest

from async_rl_lab.inference import parse_action_text


class ParserTests(unittest.TestCase):
    def test_parse_finish_json(self) -> None:
        action = parse_action_text('{"type":"finish","answer":"4"}')
        self.assertEqual(action.action_type, "finish")
        self.assertEqual(action.final_text, "4")
        self.assertEqual(action.parser_status, "ok")

    def test_parse_tool_call_json(self) -> None:
        action = parse_action_text(
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"2+2"}}'
        )
        self.assertEqual(action.action_type, "tool_call")
        self.assertIsNotNone(action.tool_call)
        self.assertEqual(action.tool_call.tool_name, "calculator")

    def test_parse_malformed_text(self) -> None:
        action = parse_action_text("not-json-at-all")
        self.assertEqual(action.action_type, "invalid")
        self.assertEqual(action.parser_status, "malformed")
