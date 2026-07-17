from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pytest

from jharness.kernel import (
    ContentPart,
    RunContext,
    SettledResult,
    Suspension,
    ToolCall,
    ToolContext,
    ToolError,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.toolkit import (
    CircuitBreakingTool,
    FunctionTool,
    RetryingTool,
    ToolRegistry,
    function_tool,
)


async def no_progress(_value: Mapping[str, Any]) -> None: ...


def context() -> ToolContext:
    return ToolContext(RunContext("run-1", 1.0), no_progress, lambda: False)


@dataclass(slots=True)
class ValueTool:
    spec: ToolSpec
    value: object
    calls: int = 0

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        self.calls += 1
        return SettledResult(ToolSuccess((ContentPart.text_part("ok"),), self.value))


async def test_registry_opens_immutable_invocation_catalogs() -> None:
    first = ValueTool(ToolSpec("first", "first", {"type": "object"}), {"value": 1})
    second = ValueTool(ToolSpec("second", "second", {"type": "object"}), {"value": 2})
    registry = ToolRegistry((first,))
    before = await registry.open_catalog()
    binding = before.bind(ToolCall("call", "first"))
    registry.register(second)
    after = await registry.open_catalog()

    assert [spec.name for spec in before.specs()] == ["first"]
    assert before.spec("second") is None
    assert [spec.name for spec in after.specs()] == ["first", "second"]
    assert binding.spec is first.spec
    assert isinstance(await binding.invoke(context()), SettledResult)


async def test_binding_validates_input_and_output_schemas() -> None:
    tool = ValueTool(
        ToolSpec(
            "strict",
            "strict",
            {
                "type": "object",
                "required": ["count"],
                "properties": {"count": {"type": "integer"}},
                "additionalProperties": False,
            },
            {
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "integer"}},
                "additionalProperties": False,
            },
        ),
        {"value": 1},
    )
    catalog = await ToolRegistry((tool,)).open_catalog()

    with pytest.raises(ToolError, match="input_schema"):
        catalog.bind(ToolCall("bad", "strict", {"count": "one"}))
    call = ToolCall("ok", "strict", {"count": 1})
    binding = catalog.bind(call)
    result = await binding.invoke(context())
    assert binding.call is call
    assert isinstance(result, SettledResult)
    assert isinstance(result.outcome, ToolSuccess)
    assert result.outcome.structured_content == {"value": 1}

    invalid_output = ValueTool(tool.spec, {"value": "one"})
    invalid_catalog = await ToolRegistry((invalid_output,)).open_catalog()
    with pytest.raises(ToolError, match="output_schema"):
        await invalid_catalog.bind(ToolCall("bad-output", "strict", {"count": 1})).invoke(context())


async def test_binding_preserves_waiting_result_and_validates_its_output() -> None:
    spec = ToolSpec(
        "wait",
        "wait",
        {"type": "object"},
        {
            "type": "object",
            "required": ["ticket"],
            "properties": {"ticket": {"type": "string"}},
        },
    )

    @dataclass(frozen=True, slots=True)
    class WaitingTool:
        spec: ToolSpec

        async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
            return WaitingResult(
                ToolWaiting(
                    (ContentPart.text_part("pending"),),
                    structured_content={"ticket": "ticket-1"},
                ),
                Suspension("external", "tool", "ticket-1"),
            )

    result = (
        await (await ToolRegistry((WaitingTool(spec),)).open_catalog())
        .bind(ToolCall("call", "wait"))
        .invoke(context())
    )

    assert isinstance(result, WaitingResult)
    assert result.outcome.kind == "waiting"
    assert result.suspension.wait_id == "ticket-1"


async def test_binding_rejects_an_unwrapped_outcome() -> None:
    @dataclass(frozen=True, slots=True)
    class InvalidTool:
        spec: ToolSpec

        async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
            return cast(Any, ToolSuccess((ContentPart.text_part("invalid"),)))

    catalog = await ToolRegistry(
        (InvalidTool(ToolSpec("invalid", "invalid", {"type": "object"})),)
    ).open_catalog()
    with pytest.raises(ToolError, match="SettledResult or WaitingResult"):
        await catalog.bind(ToolCall("call", "invalid")).invoke(context())


async def test_function_tool_decorator_preserves_explicit_spec() -> None:
    @function_tool(
        name="sum",
        description="sum values",
        input_schema={"type": "object"},
        execution=ToolExecution(read_only=True, idempotent=True),
    )
    async def sum_tool(call: ToolCall, tool_context: ToolContext) -> ToolResult:
        return SettledResult(
            ToolSuccess((ContentPart.text_part(str(sum(call.arguments.values()))),))
        )

    result = await sum_tool.invoke(ToolCall("call", "sum", {"a": 1, "b": 2}), context())
    assert sum_tool.spec.name == "sum"
    assert isinstance(result, SettledResult)
    assert result.outcome.parts[0].text == "3"


def test_function_tool_and_registry_reject_synchronous_implementations() -> None:
    spec = ToolSpec("sync", "sync", {"type": "object"})

    def sync_function(call: ToolCall, tool_context: ToolContext) -> ToolResult:
        return SettledResult(ToolSuccess((ContentPart.text_part("no"),)))

    with pytest.raises(TypeError, match="must be async"):
        FunctionTool(spec, cast(Any, sync_function))

    @dataclass(frozen=True, slots=True)
    class SyncTool:
        spec: ToolSpec

        def invoke(self, call: ToolCall, tool_context: ToolContext) -> ToolResult:
            return sync_function(call, tool_context)

    with pytest.raises(TypeError, match="must be async"):
        ToolRegistry((cast(Any, SyncTool(spec)),))


def test_tool_adapters_reject_invalid_values_and_policy_limits() -> None:
    spec = ToolSpec(
        "valid",
        "valid",
        {"type": "object"},
        execution=ToolExecution(read_only=True, idempotent=True),
    )

    async def valid_function(call: ToolCall, tool_context: ToolContext) -> ToolResult:
        return SettledResult(ToolSuccess((ContentPart.text_part("ok"),)))

    with pytest.raises(TypeError, match="spec must be ToolSpec"):
        FunctionTool(cast(Any, object()), valid_function)
    with pytest.raises(TypeError, match="implement Tool"):
        ToolRegistry((cast(Any, object()),))

    @dataclass(frozen=True, slots=True)
    class InvalidSpecTool:
        spec: object

        async def invoke(self, call: ToolCall, tool_context: ToolContext) -> ToolResult:
            return await valid_function(call, tool_context)

    with pytest.raises(TypeError, match="spec must be ToolSpec"):
        ToolRegistry((cast(Any, InvalidSpecTool(object())),))
    base = FunctionTool(spec, valid_function)
    with pytest.raises(ValueError, match="max_attempts"):
        RetryingTool(base, max_attempts=0)
    with pytest.raises(ValueError, match="attempt_timeout_seconds"):
        RetryingTool(base, attempt_timeout_seconds=0)
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreakingTool(base, failure_threshold=0)


class FlakyTool:
    spec = ToolSpec(
        "flaky",
        "flaky",
        {"type": "object"},
        execution=ToolExecution(read_only=True, idempotent=True),
    )

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("temporary")
        return SettledResult(ToolSuccess((ContentPart.text_part("ok"),)))


async def test_retry_decorator_retries_one_logical_idempotent_call() -> None:
    base = FlakyTool(2)
    result = await RetryingTool(base, max_attempts=3).invoke(ToolCall("call", "flaky"), context())

    assert isinstance(result, SettledResult)
    assert isinstance(result.outcome, ToolSuccess)
    assert base.calls == 3
    with pytest.raises(ValueError, match="idempotent"):
        RetryingTool(
            ValueTool(ToolSpec("unsafe", "unsafe", {"type": "object"}), None),
            max_attempts=2,
        )


async def test_retry_decorator_owns_attempt_timeout() -> None:
    @dataclass(slots=True)
    class SlowTool:
        spec: ToolSpec
        calls: int = 0

        async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
            self.calls += 1
            await asyncio.sleep(1)
            return SettledResult(ToolSuccess((ContentPart.text_part("late"),)))

    slow = SlowTool(
        ToolSpec(
            "slow",
            "slow",
            {"type": "object"},
            execution=ToolExecution(read_only=True, idempotent=True),
        )
    )
    with pytest.raises(TimeoutError):
        await RetryingTool(slow, max_attempts=2, attempt_timeout_seconds=0.001).invoke(
            ToolCall("call", "slow"), context()
        )
    assert slow.calls == 2


async def test_decorators_propagate_cancellation_without_recording_failure() -> None:
    @dataclass(slots=True)
    class CancellingTool:
        spec: ToolSpec
        calls: int = 0

        async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
            self.calls += 1
            raise asyncio.CancelledError

    base = CancellingTool(
        ToolSpec(
            "cancel",
            "cancel",
            {"type": "object"},
            execution=ToolExecution(read_only=True, idempotent=True),
        )
    )
    call = ToolCall("call", "cancel")
    with pytest.raises(asyncio.CancelledError):
        await RetryingTool(base, max_attempts=2).invoke(call, context())
    breaker = CircuitBreakingTool(base, failure_threshold=1)
    assert breaker.spec is base.spec
    with pytest.raises(asyncio.CancelledError):
        await breaker.invoke(call, context())
    with pytest.raises(asyncio.CancelledError):
        await breaker.invoke(call, context())
    assert base.calls == 3


async def test_circuit_breaker_success_resets_consecutive_failure_count() -> None:
    @dataclass(slots=True)
    class AlternatingTool:
        spec: ToolSpec
        calls: int = 0

        async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
            self.calls += 1
            if self.calls % 2:
                raise RuntimeError("temporary")
            return SettledResult(ToolSuccess((ContentPart.text_part("ok"),)))

    base = AlternatingTool(ToolSpec("alternating", "alternating", {"type": "object"}))
    breaker = CircuitBreakingTool(base, failure_threshold=2)
    call = ToolCall("call", "alternating")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="temporary"):
            await breaker.invoke(call, context())
        result = await breaker.invoke(call, context())
        assert isinstance(result, SettledResult)
        assert isinstance(result.outcome, ToolSuccess)
    assert base.calls == 4


async def test_circuit_breaker_opens_after_consecutive_failures() -> None:
    base = FlakyTool(10)
    breaker = CircuitBreakingTool(base, failure_threshold=2)
    call = ToolCall("call", "flaky")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="temporary"):
            await breaker.invoke(call, context())

    result = await breaker.invoke(call, context())
    assert isinstance(result, SettledResult)
    assert isinstance(result.outcome, ToolFailure)
    assert result.outcome.error.code == "circuit_open"
    assert base.calls == 2


def test_registry_rejects_duplicate_names_and_invalid_schemas() -> None:
    tool = ValueTool(ToolSpec("same", "same", {"type": "object"}), None)
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry((tool, tool))
    with pytest.raises(ValueError, match="valid JSON Schema"):
        ToolRegistry(
            (
                ValueTool(
                    ToolSpec("bad", "bad", {"type": "not-a-json-schema-type"}),
                    None,
                ),
            )
        )
    with pytest.raises(ValueError, match="unresolvable"):
        ToolRegistry((ValueTool(ToolSpec("ref", "ref", {"$ref": "missing"}), None),))
