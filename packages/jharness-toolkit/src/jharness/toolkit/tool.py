"""Concrete Python tool protocol and function adapter."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import Any, Protocol, cast, runtime_checkable

from jharness.kernel import (
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolResult,
    ToolRisk,
    ToolSpec,
)


@runtime_checkable
class Tool(Protocol):
    """One async invocation operation and one immutable specification."""

    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult: ...


ToolFunction = Callable[[ToolCall, ToolContext], Coroutine[Any, Any, ToolResult]]


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Adapt one async function to the concrete tool protocol."""

    spec: ToolSpec
    function: ToolFunction

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.spec), ToolSpec):
            raise TypeError("function tool spec must be ToolSpec")
        if not iscoroutinefunction(self.function):
            raise TypeError("function tool must be async")

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        return await self.function(call, context)


def function_tool(
    *,
    name: str,
    description: str,
    input_schema: Mapping[str, Any] | bool,
    output_schema: Mapping[str, Any] | bool | None = None,
    execution: ToolExecution | None = None,
    risk: ToolRisk | None = None,
) -> Callable[[ToolFunction], FunctionTool]:
    """Create a `FunctionTool` while keeping schemas and policies explicit."""

    tool_execution = ToolExecution() if execution is None else execution
    tool_risk = ToolRisk() if risk is None else risk
    spec = ToolSpec(
        name,
        description,
        input_schema,
        output_schema,
        tool_execution,
        tool_risk,
    )

    def decorate(function: ToolFunction) -> FunctionTool:
        return FunctionTool(spec, function)

    return decorate
