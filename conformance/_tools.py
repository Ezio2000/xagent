"""Normative deterministic tools described by tools.contract.json."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conformance._schemas import SchemaSuite
from conformance._values import boolean, load_object, mapping, number, sequence, string
from jharness.kernel import (
    BatchPolicy,
    ContentPart,
    RunLimits,
    SettledResult,
    Suspension,
    ToolAccepted,
    ToolBatch,
    ToolCall,
    ToolCatalog,
    ToolContext,
    ToolFailure,
    ToolResult,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
    thaw_json_value,
)
from jharness.kernel.wire import decode_tool_spec
from jharness.toolkit import ToolRegistry


class _FaultBatchPolicy:
    __slots__ = ("_kind",)

    def __init__(self, kind: str) -> None:
        self._kind = kind

    def select(
        self,
        pending: Sequence[ToolCall],
        catalog: ToolCatalog,
        limits: RunLimits,
    ) -> ToolBatch:
        del catalog, limits
        if self._kind == "empty":
            raise ValueError("tool batch requires calls")
        if self._kind == "skip_prefix":
            return ToolBatch("invalid", (pending[1],))
        if self._kind == "serial_multiple":
            raise ValueError("serial tool batch must contain exactly one call")
        if self._kind == "oversized":
            first = pending[0]
            extra = ToolCall(f"{first.id}-extra", first.name, {})
            return ToolBatch("invalid", (first, extra), parallel=True)
        return ToolBatch("invalid", tuple(pending[:2]), parallel=True)


def fixture_batch_policy(value: object) -> BatchPolicy | None:
    """Create the default strategy or one controlled invalid implementation."""

    return None if value is None else _FaultBatchPolicy(string(value, "batch policy fault"))


class StandardTool:
    __slots__ = ("behavior", "defaults", "spec")

    def __init__(
        self,
        spec: ToolSpec,
        behavior: str,
        defaults: Mapping[str, Any],
    ) -> None:
        self.spec = spec
        self.behavior = behavior
        self.defaults = dict(defaults)

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        arguments = dict(self.defaults)
        raw_arguments = thaw_json_value(call.arguments)
        arguments.update(mapping(raw_arguments, "tool arguments"))
        for key, value in tuple(arguments.items()):
            if value == "$call_id":
                arguments[key] = call.id
        if self.behavior == "exception":
            raise RuntimeError("deterministic tool failure")
        if self.behavior in {"success", "strict"}:
            return await self._success(arguments)
        if self.behavior == "accepted":
            return SettledResult(
                ToolAccepted(
                    (ContentPart.text_part(string(arguments["text"], "accept text")),),
                    string(arguments["correlation_id"], "accept correlation_id"),
                )
            )
        if self.behavior in {"waiting", "progress"}:
            return await self._waiting_or_progress(context, arguments)
        if self.behavior == "invalid_output":
            return SettledResult(
                ToolSuccess(
                    (ContentPart.text_part("invalid output"),),
                    {"value": "not-an-integer"},
                )
            )
        raise AssertionError(f"unsupported standard tool behavior: {self.behavior}")

    async def _success(self, arguments: Mapping[str, Any]) -> ToolResult:
        delay = number(arguments.get("delay", 0), "tool delay")
        if delay:
            await asyncio.sleep(delay)
        text = (
            str(arguments["count"])
            if self.behavior == "strict"
            else string(arguments.get("text", ""), "tool text")
        )
        return SettledResult(ToolSuccess((ContentPart.text_part(text),)))

    async def _waiting_or_progress(
        self,
        context: ToolContext,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        if self.behavior == "waiting":
            delay = number(arguments.get("delay", 0), "wait delay")
            if delay:
                await asyncio.sleep(delay)
            suspension = Suspension(
                string(arguments["reason"], "wait reason"),
                string(arguments["source"], "wait source"),
                string(arguments["wait_id"], "wait_id"),
            )
            return WaitingResult(
                ToolWaiting((ContentPart.text_part(string(arguments["text"], "wait text")),)),
                suspension,
            )

        steps = sequence(arguments.get("steps", []), "progress steps")
        delay = number(arguments.get("step_delay", 0), "progress step_delay")
        burst = boolean(arguments.get("burst", False), "progress burst")
        for step in steps:
            if delay and not burst:
                await asyncio.sleep(delay)
            await context.emit_progress({"step": step})
            if not burst:
                await asyncio.sleep(0)
            if context.cancel_requested:
                return _cancelled(arguments)
        if context.cancel_requested:
            return _cancelled(arguments)
        return SettledResult(
            ToolSuccess((ContentPart.text_part(string(arguments["text"], "progress text")),))
        )


def _cancelled(arguments: Mapping[str, Any]) -> SettledResult:
    return SettledResult(
        ToolFailure.from_error(
            "cancelled",
            string(arguments["cancel_text"], "progress cancel_text"),
        )
    )


def load_standard_tools(
    manifest_path: Path,
    schema_path: Path,
    schemas: SchemaSuite,
) -> ToolRegistry:
    document = load_object(manifest_path, "tool contract")
    schemas.validate_document(schema_path, document)
    tools: list[StandardTool] = []
    for raw_entry in sequence(document["tools"], "standard tools"):
        entry = mapping(raw_entry, "standard tool")
        tools.append(
            StandardTool(
                decode_tool_spec(entry["spec"]),
                string(entry["behavior"], "standard tool behavior"),
                mapping(entry["defaults"], "standard tool defaults"),
            )
        )
    return ToolRegistry(tuple(tools))
