from __future__ import annotations

from typing import Any, cast

import pytest

import agent_runtime.tools as tools_module
from agent_runtime import (
    BackgroundTask,
    ContentPart,
    DuplicateToolError,
    InvalidToolCall,
    PauseRequest,
    RuntimeContext,
    ToolCall,
    ToolExecutionContext,
    ToolInvocation,
    ToolObservation,
    ToolOutput,
    ToolRegistry,
    ToolRejection,
    ToolSpec,
    normalized_tool_risk,
)


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


def test_tool_risk_annotation_is_normalized_for_approval() -> None:
    spec = ToolSpec(
        name="risk",
        description="Risky tool.",
        input_schema={"type": "object", "properties": {}},
        annotations={
            "read_only": False,
            "parallel_safe": False,
            "risk": {
                "filesystem": "delete",
                "network": "write",
                "subprocess": True,
                "destructive": True,
                "requires_approval": True,
                "custom": "allowed",
            },
        },
    )

    assert normalized_tool_risk(spec.annotations) == {
        "filesystem": "delete",
        "network": "write",
        "subprocess": True,
        "destructive": True,
        "requires_approval": True,
        "custom": "allowed",
        "read_only": False,
        "parallel_safe": False,
    }


def test_tool_risk_annotation_rejects_invalid_standard_fields() -> None:
    with pytest.raises(ValueError, match="filesystem"):
        ToolSpec(
            name="risk",
            description="Risky tool.",
            input_schema={"type": "object", "properties": {}},
            annotations={"risk": {"filesystem": ""}},
        )


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


def test_waiting_tool_result_round_trips_pause_request() -> None:
    result = ToolObservation.waiting(
        "started",
        wait_id="job-1",
        reason="external_callback",
        pause_metadata={"kind": "job"},
    )
    restored = ToolObservation.from_dict(result.to_dict())

    assert restored.text_content == "started"
    assert restored.pause is not None
    assert restored.pause.reason == "external_callback"
    assert restored.pause.source == "tool"
    assert restored.pause.wait_id == "job-1"
    assert restored.pause.metadata == {"kind": "job"}


def test_waiting_tool_result_round_trips_background_task() -> None:
    task = BackgroundTask(
        id="research-1",
        status="accepted",
        kind="research",
        correlation_id="call-1",
        metadata={"topic": "runtime"},
    )
    result = ToolObservation.waiting(
        "started",
        wait_id=task.id,
        reason="research_callback",
        background_task=task,
    )
    restored = ToolObservation.from_dict(result.to_dict())

    assert restored.background_task == task
    assert restored.pause is not None
    assert restored.pause.metadata["background_task"] == task.to_dict()


def test_tool_result_rejects_interrupt_pause() -> None:
    with pytest.raises(ValueError, match="cannot interrupt"):
        ToolObservation(
            parts=[],
            pause=PauseRequest(
                reason="bad",
                source="tool",
                wait_id="job-1",
                metadata={},
                interrupt=True,
            ),
        )


def test_tool_result_to_message_strips_host_metadata() -> None:
    result = ToolObservation(
        parts=[ContentPart.text_part("created", metadata={"secret": "part"})],
        metadata={"secret": "result"},
        is_error=True,
    )

    message = result.to_message(ToolInvocation.from_tool_call(ToolCall(id="call-1", name="tool")))

    assert message.metadata == {"result_kind": "observation", "is_error": True}
    assert message.parts[0].metadata == {}


def test_tool_result_from_dict_rejects_invalid_required_fields() -> None:
    with pytest.raises(KeyError):
        ToolObservation.from_dict({"is_error": False})

    with pytest.raises(KeyError):
        ToolObservation.from_dict({"kind": "observation", "parts": []})

    with pytest.raises(KeyError):
        ToolOutput.from_dict({"kind": "observation", "parts": []})

    with pytest.raises(ValueError, match="kind"):
        ToolObservation.from_dict({"kind": "acceptance", "parts": [], "is_error": False})

    with pytest.raises(TypeError, match="is_error"):
        ToolObservation.from_dict({"kind": "observation", "parts": [], "is_error": "false"})

    with pytest.raises(TypeError, match="metadata"):
        ToolObservation.from_dict(
            {"kind": "observation", "parts": [], "is_error": False, "metadata": None}
        )

    with pytest.raises(TypeError, match="metadata"):
        ToolObservation.from_dict(
            {"kind": "observation", "parts": [], "is_error": False, "metadata": []}
        )

    with pytest.raises(TypeError, match="metadata"):
        ToolObservation.text("ok", metadata=cast(Any, []))

    with pytest.raises(TypeError, match="parts items"):
        ToolObservation(parts=[cast(Any, object())])

    with pytest.raises(ValueError, match="is_error"):
        ToolRejection.from_dict({"kind": "rejection", "parts": [], "is_error": False})


def test_tool_spec_from_dict_rejects_schema_invalid_fields() -> None:
    with pytest.raises(TypeError, match="tool name"):
        ToolSpec.from_dict(
            {"name": 1, "description": "tool", "input_schema": {}, "modes": ["execute"]}
        )

    with pytest.raises(KeyError):
        ToolSpec.from_dict({"name": "tool", "description": "tool"})

    with pytest.raises(KeyError):
        ToolSpec.from_dict({"name": "tool", "description": "tool", "input_schema": {}})

    with pytest.raises(TypeError, match="metadata"):
        ToolSpec.from_dict(
            {
                "name": "tool",
                "description": "tool",
                "input_schema": {},
                "modes": ["execute"],
                "metadata": None,
            }
        )


def test_tool_spec_constructor_rejects_invalid_core_types() -> None:
    with pytest.raises(TypeError, match="tool name"):
        ToolSpec(name=cast(Any, 1), description="tool", input_schema={})

    with pytest.raises(TypeError, match="tool description"):
        ToolSpec(name="tool", description=cast(Any, 1), input_schema={})

    with pytest.raises(ValueError, match="valid JSON Schema"):
        ToolSpec(name="tool", description="tool", input_schema={"type": 1})


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
async def test_tool_registry_reuses_cached_input_schema_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry([StrictCountTool()])

    def fail_validator(schema: object) -> Any:
        _ = schema
        raise AssertionError("validator must be cached at registration")

    monkeypatch.setattr(tools_module, "Draft202012Validator", fail_validator)

    result = await registry.invoke(
        ToolCall(id="call-1", name="strict_count", arguments={"count": 3}),
        RuntimeContext(),
    )

    assert result.text_content == "3"


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
