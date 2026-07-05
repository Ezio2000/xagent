from __future__ import annotations

from typing import Any, cast

import pytest
import toolkit._registry.api as registry_module
from kernel import (
    ContentPart,
    DuplicateToolError,
    InvalidToolCall,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)
from toolkit import ToolExecutionContext, ToolInvocation, ToolRegistry


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


class MutatingTool:
    spec = ToolSpec(
        name="mutate",
        description="Mutate arguments.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        cast(dict[str, Any], invocation.arguments)["changed"] = True
        return ToolObservation.text("ok")


class InvalidResultTool:
    spec = ToolSpec(
        name="invalid_result",
        description="Return a non-ToolObservation value.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        return cast(Any, {"kind": "observation", "parts": [], "is_error": False})


class MutableSpecTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="mutable",
            description="Initial contract.",
            input_schema={"type": "object", "properties": {}},
            annotations={"parallel_safe": False},
        )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        return ToolObservation.text("ok")


class CustomModeTool:
    spec = ToolSpec(
        name="custom",
        description="Handle a custom invocation mode.",
        input_schema={"type": "object", "properties": {}},
        modes=("handoff",),
    )

    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolOutput:
        _ = context
        return ToolOutput(
            kind="handoff",
            parts=[ContentPart.text_part(str(invocation.arguments.get("text", "")))],
            correlation_id=invocation.id,
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


class OutputSchemaMismatchTool:
    spec = ToolSpec(
        name="output_schema_mismatch",
        description="Return a valid observation whose payload does not match output_schema.",
        input_schema={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "required": ["required_field"],
            "properties": {"required_field": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        return ToolObservation.text("not schema validated")


class RejectingAcceptTool:
    spec = ToolSpec(
        name="accepting",
        description="Reject accept-mode invocations.",
        input_schema={"type": "object", "properties": {}},
        modes=("accept",),
    )

    async def accept(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolRejection:
        _ = context
        return ToolRejection.text(
            str(invocation.arguments.get("text", "rejected")),
            correlation_id=invocation.id,
        )


class ReservedKindCustomModeTool:
    spec = ToolSpec(
        name="reserved_custom",
        description="Return a reserved result kind from a custom invocation mode.",
        input_schema={"type": "object", "properties": {}},
        modes=("handoff",),
    )

    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolOutput:
        _ = invocation, context
        return ToolOutput(
            kind="observation",
            parts=[ContentPart.text_part("invalid")],
        )


def test_tool_specs_are_defensive_copies() -> None:
    registry = ToolRegistry([EchoTool()])

    first = registry.specs()
    second = registry.specs()
    assert first is not second
    assert first[0].name == "echo"
    cast(dict[str, Any], first[0].input_schema)["mutated"] = True
    assert "mutated" not in registry.specs()[0].input_schema


def test_tool_spec_lookup_uses_registered_snapshot() -> None:
    tool = MutableSpecTool()
    registry = ToolRegistry([tool])
    tool.spec = ToolSpec(
        name="mutable",
        description="Changed contract.",
        input_schema={"type": "object", "properties": {"changed": {"type": "boolean"}}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    spec = registry.spec_for("mutable")

    assert spec is not None
    assert spec.description == "Initial contract."
    assert spec.annotations == {"parallel_safe": False}
    cast(dict[str, Any], spec.input_schema)["mutated"] = True
    fresh = registry.spec_for("mutable")
    assert fresh is not None
    assert "mutated" not in fresh.input_schema


def test_tool_register_preserves_existing_spec_snapshots() -> None:
    tool = MutableSpecTool()
    registry = ToolRegistry([tool])
    tool.spec = ToolSpec(
        name="mutable",
        description="Changed contract.",
        input_schema={"type": "object", "properties": {"changed": {"type": "boolean"}}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    registry.register(EchoTool())

    spec = registry.spec_for("mutable")
    assert spec is not None
    assert spec.description == "Initial contract."
    assert spec.annotations == {"parallel_safe": False}


def test_duplicate_tool_name_rejected() -> None:
    with pytest.raises(DuplicateToolError):
        ToolRegistry([EchoTool(), EchoTool()])


def test_tool_registry_rejects_invalid_json_schema() -> None:
    class InvalidSchemaTool:
        spec = ToolSpec(name="tool", description="tool", input_schema={"type": 1})

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation, context
            return ToolObservation.text("unreachable")

    with pytest.raises(ValueError, match="valid JSON Schema"):
        ToolRegistry([InvalidSchemaTool()])


def test_tool_registry_rejects_invalid_output_json_schema() -> None:
    class InvalidOutputSchemaTool:
        spec = ToolSpec(
            name="tool",
            description="tool",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": 1},
        )

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation, context
            return ToolObservation.text("unreachable")

    with pytest.raises(ValueError, match="tool output_schema"):
        ToolRegistry([InvalidOutputSchemaTool()])


@pytest.mark.asyncio
async def test_tool_arguments_are_defensive_copies() -> None:
    call = ToolCall(id="call-1", name="mutate", arguments={"value": 1})
    result = await ToolRegistry([MutatingTool()]).invoke(call, RuntimeContext())

    assert result.text_content == "ok"
    assert call.arguments == {"value": 1}


@pytest.mark.asyncio
async def test_tool_registry_rejects_invalid_tool_result_type() -> None:
    call = ToolCall(id="call-1", name="invalid_result", arguments={})

    with pytest.raises(TypeError, match="ToolObservation"):
        await ToolRegistry([InvalidResultTool()]).invoke(call, RuntimeContext())


@pytest.mark.asyncio
async def test_tool_registry_validates_arguments_against_input_schema() -> None:
    tool = StrictCountTool()
    call = ToolCall(id="call-1", name="strict_count", arguments={"count": "bad"})

    with pytest.raises(InvalidToolCall, match="input_schema"):
        await ToolRegistry([tool]).invoke(call, RuntimeContext())

    assert tool.calls == 0


@pytest.mark.asyncio
async def test_tool_registry_does_not_validate_output_against_output_schema() -> None:
    call = ToolCall(id="call-1", name="output_schema_mismatch", arguments={})

    result = await ToolRegistry([OutputSchemaMismatchTool()]).invoke(call, RuntimeContext())

    assert result.text_content == "not schema validated"


@pytest.mark.asyncio
async def test_tool_registry_dispatches_custom_modes_to_invoke() -> None:
    call = ToolCall(
        id="call-1",
        name="custom",
        mode="handoff",
        arguments={"text": "accepted"},
    )

    result = await ToolRegistry([CustomModeTool()]).invoke(call, RuntimeContext())

    assert result.kind == "handoff"
    assert result.text_content == "accepted"
    assert result.correlation_id == "call-1"


@pytest.mark.asyncio
async def test_tool_registry_accepts_accept_rejections() -> None:
    call = ToolCall(
        id="call-1",
        name="accepting",
        mode="accept",
        arguments={"text": "not accepted"},
    )

    result = await ToolRegistry([RejectingAcceptTool()]).invoke(call, RuntimeContext())

    assert isinstance(result, ToolRejection)
    assert result.kind == "rejection"
    assert result.is_error is True
    assert result.text_content == "not accepted"
    assert result.correlation_id == "call-1"


@pytest.mark.asyncio
async def test_tool_registry_rejects_reserved_result_kind_for_custom_mode() -> None:
    call = ToolCall(id="call-1", name="reserved_custom", mode="handoff", arguments={})

    with pytest.raises(TypeError, match="extension ToolOutput kind"):
        await ToolRegistry([ReservedKindCustomModeTool()]).invoke(call, RuntimeContext())


@pytest.mark.asyncio
async def test_tool_registry_reuses_cached_input_schema_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry([StrictCountTool()])

    def fail_validator(schema: object) -> Any:
        _ = schema
        raise AssertionError("validator must be cached at registration")

    monkeypatch.setattr(registry_module, "Draft202012Validator", fail_validator)

    result = await registry.invoke(
        ToolCall(id="call-1", name="strict_count", arguments={"count": 3}),
        RuntimeContext(),
    )

    assert result.text_content == "3"
