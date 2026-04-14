from __future__ import annotations

import asyncio
import ast
from typing import Protocol

from async_rl_lab.ids import utc_ts
from async_rl_lab.models import ToolCall, ToolResult


class Tool(Protocol):
    name: str
    description: str

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        ...


class CalculatorTool:
    name = "calculator"
    description = "Evaluates a basic arithmetic expression."

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        started_ts = utc_ts()
        try:
            expression = str(tool_call.arguments["expression"])
            answer = safe_eval_expression(expression)
            ended_ts = utc_ts()
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                status="ok",
                started_ts=started_ts,
                ended_ts=ended_ts,
                output_text=str(answer),
                output_json={"value": answer},
            )
        except KeyError:
            ended_ts = utc_ts()
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                status="validation_error",
                started_ts=started_ts,
                ended_ts=ended_ts,
                output_text="missing expression",
                error_type="missing_argument",
                error_message="expression is required",
            )
        except Exception as exc:
            ended_ts = utc_ts()
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=self.name,
                status="error",
                started_ts=started_ts,
                ended_ts=ended_ts,
                output_text=str(exc),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


class SearchRetrievalMockTool:
    name = "search"
    description = "Returns mock documents from an in-memory corpus."

    def __init__(self, corpus: dict[str, str] | None = None) -> None:
        self.corpus = corpus or {
            "grpo": "GRPO groups multiple completions per prompt and compares relative reward.",
            "staleness": "Policy lag grows as actors sample older checkpoints.",
        }

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        started_ts = utc_ts()
        query = str(tool_call.arguments.get("query", "")).lower()
        result = self.corpus.get(query, "no result")
        ended_ts = utc_ts()
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=self.name,
            status="ok",
            started_ts=started_ts,
            ended_ts=ended_ts,
            output_text=result,
            output_json={"query": query, "result": result},
        )


class TextPatchTool:
    name = "text_patch"
    description = "Applies a plain replace patch to an in-memory document."

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        started_ts = utc_ts()
        document_id = str(tool_call.arguments.get("document_id", "default"))
        find_text = str(tool_call.arguments.get("find", ""))
        replace_text = str(tool_call.arguments.get("replace", ""))
        original = self.documents.get(document_id, "")
        updated = original.replace(find_text, replace_text)
        self.documents[document_id] = updated
        ended_ts = utc_ts()
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=self.name,
            status="ok",
            started_ts=started_ts,
            ended_ts=ended_ts,
            output_text=updated,
            output_json={"document_id": document_id, "length": len(updated)},
        )


class ShellMockTool:
    name = "shell"
    description = "Returns a deterministic mock shell result for safe sandboxed demos."

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        started_ts = utc_ts()
        command = str(tool_call.arguments.get("command", ""))
        await asyncio.sleep(0.01)
        ended_ts = utc_ts()
        return ToolResult(
            tool_call_id=tool_call.tool_call_id,
            tool_name=self.name,
            status="ok",
            started_ts=started_ts,
            ended_ts=ended_ts,
            output_text=f"mock-executed: {command}",
            output_json={"command": command},
        )


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tools))

    async def invoke(self, tool_call: ToolCall, *, timeout_s: float | None = None) -> ToolResult:
        if tool_call.tool_name not in self.tools:
            started_ts = utc_ts()
            ended_ts = utc_ts()
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.tool_name,
                status="validation_error",
                started_ts=started_ts,
                ended_ts=ended_ts,
                output_text="tool not registered",
                error_type="unknown_tool",
                error_message=f"unknown tool: {tool_call.tool_name}",
            )
        tool = self.tools[tool_call.tool_name]
        if timeout_s is None:
            return await tool.invoke(tool_call)
        return await asyncio.wait_for(tool.invoke(tool_call, timeout_s=timeout_s), timeout=timeout_s)


def safe_eval_expression(expression: str) -> float:
    parsed = ast.parse(expression, mode="eval")
    return float(evaluate_ast(parsed.body))


def evaluate_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = evaluate_ast(node.left)
        right = evaluate_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError("unsupported expression")
