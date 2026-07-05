"""Tool registry doubles for controlled runtime scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import TypeAlias

from kernel import (
    ContentPart,
    InvalidToolCall,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolSpec,
)

ToolHandler: TypeAlias = Callable[[ToolCall, RuntimeContext], ToolOutput | Awaitable[ToolOutput]]


def _empty_mapping() -> Mapping[str, object]:
    return {}


@dataclass(slots=True, frozen=True)
class ToolInvocationRecord:
    """A copied record of one fake registry invocation."""

    call: ToolCall
    context: RuntimeContext
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)


class ScriptedToolRegistry:
    """Registry double with explicit handlers by tool name."""

    def __init__(
        self,
        specs: Sequence[ToolSpec],
        handlers: Mapping[str, ToolHandler],
    ) -> None:
        self._specs = tuple(ToolSpec.from_dict(spec.to_dict()) for spec in specs)
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        if len(self._specs_by_name) != len(self._specs):
            raise ValueError("scripted tool specs must have unique names")
        self._handlers = dict(handlers)
        unknown_handlers = set(self._handlers) - set(self._specs_by_name)
        if unknown_handlers:
            names = ", ".join(sorted(unknown_handlers))
            raise ValueError(f"scripted tool handlers without specs: {names}")
        self.records: list[ToolInvocationRecord] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(ToolSpec.from_dict(spec.to_dict()) for spec in self._specs)

    def spec_for(self, name: str) -> ToolSpec | None:
        spec = self._specs_by_name.get(name)
        if spec is None:
            return None
        return ToolSpec.from_dict(spec.to_dict())

    def validate_call(self, call: ToolCall) -> None:
        spec = self._specs_by_name.get(call.name)
        if spec is None:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        if call.mode not in spec.modes:
            raise InvalidToolCall(f"unsupported tool mode for {call.name}: {call.mode}")

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContext,
        *,
        progress_emitter: Callable[[Mapping[str, object]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> ToolOutput:
        _ = progress_emitter, cancel_checker
        self.validate_call(call)
        self.records.append(
            ToolInvocationRecord(
                call=ToolCall.from_dict(call.to_dict()),
                context=RuntimeContext.from_dict(context.to_dict()),
            )
        )
        handler = self._handlers.get(call.name)
        if handler is None:
            raise InvalidToolCall(f"no scripted handler for tool: {call.name}")
        output = handler(call, context)
        if isawaitable(output):
            output = await output
        return ToolOutput.from_dict(output.to_dict())


class RecordingToolRegistry(ScriptedToolRegistry):
    """Registry double with one tool that records invocation ids and echoes them."""

    def __init__(self, spec: ToolSpec | None = None) -> None:
        self.calls: list[str] = []
        resolved_spec = spec or ToolSpec(
            name="record",
            description="Record executed call ids.",
            input_schema={"type": "object", "properties": {}},
        )
        super().__init__([resolved_spec], {resolved_spec.name: self._record})

    def _record(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = context
        call_id = str(call.arguments["id"])
        self.calls.append(call_id)
        return ToolObservation.text(call_id)


class FixtureToolRegistry(ScriptedToolRegistry):
    """Registry double with reusable runtime-behavior tools by name."""

    _DEFAULT_TOOL_NAMES = ("echo",)

    def __init__(self, *tool_names: str) -> None:
        names = tool_names or self._DEFAULT_TOOL_NAMES
        specs: list[ToolSpec] = []
        handlers: dict[str, ToolHandler] = {}
        for name in names:
            spec, handler = self._tool_definition(name)
            specs.append(spec)
            handlers[spec.name] = handler
        super().__init__(specs, handlers)

    def _tool_definition(self, name: str) -> tuple[ToolSpec, ToolHandler]:
        if name == "echo":
            return (
                ToolSpec(
                    name="echo",
                    description="Return input text.",
                    input_schema={"type": "object", "properties": {}},
                ),
                self._echo,
            )
        if name == "fail":
            return (
                ToolSpec(
                    name="fail",
                    description="Raise an error.",
                    input_schema={"type": "object", "properties": {}},
                ),
                self._fail,
            )
        if name == "wait":
            return (
                ToolSpec(
                    name="wait",
                    description="Start external work and pause the run.",
                    input_schema={"type": "object", "properties": {}},
                ),
                self._wait,
            )
        if name == "parallel_wait":
            return (
                ToolSpec(
                    name="parallel_wait",
                    description="Start external work and pause the run.",
                    input_schema={"type": "object", "properties": {}},
                    annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
                ),
                self._parallel_wait,
            )
        if name == "slow":
            return (
                ToolSpec(
                    name="slow",
                    description="Sleep too long.",
                    input_schema={"type": "object", "properties": {}},
                ),
                self._slow,
            )
        if name == "metadata_tool":
            return (
                ToolSpec(
                    name="metadata_tool",
                    description="Return metadata-bearing content.",
                    input_schema={"type": "object", "properties": {}},
                ),
                self._metadata,
            )
        raise ValueError(f"unknown fixture tool: {name}")

    def _echo(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = context
        return ToolObservation.text(str(call.arguments.get("text", "")))

    def _fail(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = call, context
        return ToolObservation.text("tool failed", is_error=True)

    def _wait(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = context
        return ToolObservation.waiting(
            "external job started",
            wait_id=str(call.arguments["wait_id"]),
            reason="external_callback",
        )

    async def _parallel_wait(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = context
        await asyncio.sleep(float(call.arguments.get("delay", 0)))
        wait_id = str(call.arguments["wait_id"])
        return ToolObservation.waiting(
            wait_id,
            wait_id=wait_id,
            reason="external_callback",
        )

    async def _slow(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = call, context
        await asyncio.sleep(1)
        return ToolObservation.text("late")

    def _metadata(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        _ = call, context
        return ToolObservation(
            parts=[ContentPart.text_part("tool", metadata={"secret": "part"})],
            metadata={"secret": "result"},
        )
