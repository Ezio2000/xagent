"""Standard portable tool fixtures used by conformance cases."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, cast

from kernel import (
    BackgroundTask,
    ContentPart,
    ToolAcceptance,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)
from toolkit import Tool, ToolExecutionContext, ToolInvocation


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.text(str(invocation.arguments.get("text", "")))


class AcceptTool:
    spec = ToolSpec(
        name="accept",
        description="Accept an external operation.",
        input_schema={"type": "object", "properties": {}},
        modes=("accept",),
    )

    async def accept(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolAcceptance | ToolRejection:
        _ = context
        if invocation.arguments.get("reject") is True:
            return ToolRejection.text(str(invocation.arguments.get("text", "rejected")))
        return ToolAcceptance.text(
            str(invocation.arguments.get("text", "accepted")),
            correlation_id=str(invocation.arguments.get("correlation_id", invocation.id)),
        )


class HandoffTool:
    spec = ToolSpec(
        name="handoff",
        description="Return generic custom-mode tool output.",
        input_schema={"type": "object", "properties": {}},
        modes=("handoff",),
    )

    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolOutput:
        _ = context
        return ToolOutput(
            kind=str(invocation.arguments.get("kind", "handoff")),
            parts=[ContentPart.text_part(str(invocation.arguments.get("text", "handoff")))],
            is_error=bool(invocation.arguments.get("is_error", False)),
            correlation_id=str(invocation.arguments.get("correlation_id", invocation.id)),
        )


class FailTool:
    spec = ToolSpec(
        name="fail",
        description="Raise an error.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        raise RuntimeError("tool failed")


class DelayedEchoTool:
    spec = ToolSpec(
        name="delayed_echo",
        description="Return input text after an optional delay.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        await asyncio.sleep(float(invocation.arguments.get("delay", 0)))
        return ToolObservation.text(str(invocation.arguments.get("text", "")))


class WaitTool:
    spec = ToolSpec(
        name="wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        raw_background_task = invocation.arguments.get("background_task")
        background_task = None
        if raw_background_task is not None:
            if not isinstance(raw_background_task, Mapping):
                raise TypeError("wait background_task must be an object")
            background_task = BackgroundTask.from_dict(cast(Mapping[str, Any], raw_background_task))
        return ToolObservation.waiting(
            str(invocation.arguments.get("text", "external wait started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason=str(invocation.arguments.get("reason", "external_wait")),
            background_task=background_task,
        )


class ProgressTool:
    spec = ToolSpec(
        name="progress",
        description="Emit live progress records.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        raw_steps = invocation.arguments.get("steps", [])
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes):
            raise TypeError("progress steps must be a sequence")
        steps = cast(Sequence[Any], raw_steps)
        for step in steps:
            context.emit_progress({"step": step})
        return ToolObservation.text(str(invocation.arguments.get("text", "progress complete")))


class ParallelWaitTool:
    spec = ToolSpec(
        name="parallel_wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        await asyncio.sleep(float(invocation.arguments.get("delay", 0)))
        return ToolObservation.waiting(
            str(invocation.arguments.get("text", "external wait started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason=str(invocation.arguments.get("reason", "external_wait")),
        )


class StrictCountTool:
    def __init__(self) -> None:
        self.calls = 0

    spec = ToolSpec(
        name="strict_count",
        description="Require an integer count.",
        input_schema={
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
            "additionalProperties": False,
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        self.calls += 1
        return ToolObservation.text(str(invocation.arguments["count"]))


def standard_case_tools() -> list[Tool]:
    return [
        EchoTool(),
        AcceptTool(),
        HandoffTool(),
        FailTool(),
        DelayedEchoTool(),
        WaitTool(),
        ProgressTool(),
        ParallelWaitTool(),
        StrictCountTool(),
    ]
