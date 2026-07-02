from __future__ import annotations

from typing import Any, cast

import pytest

from agent_runtime import (
    ContentPart,
    DuplicateToolError,
    PauseRequest,
    RuntimeContext,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.text(str(arguments.get("text", "")))


class MutatingTool:
    spec = ToolSpec(
        name="mutate",
        description="Mutate arguments.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        arguments["changed"] = True
        return ToolResult.text("ok")


class InvalidResultTool:
    spec = ToolSpec(
        name="invalid_result",
        description="Return a non-ToolResult value.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        return cast(Any, {"parts": [], "is_error": False})


class MutableSpecTool:
    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="mutable",
            description="Initial contract.",
            input_schema={"type": "object", "properties": {}},
            annotations={"parallel_safe": False},
        )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        return ToolResult.text("ok")


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


def test_waiting_tool_result_round_trips_pause_request() -> None:
    result = ToolResult.waiting(
        "started",
        wait_id="job-1",
        reason="external_callback",
        pause_metadata={"kind": "job"},
    )
    restored = ToolResult.from_dict(result.to_dict())

    assert restored.text_content == "started"
    assert restored.pause is not None
    assert restored.pause.reason == "external_callback"
    assert restored.pause.source == "tool"
    assert restored.pause.wait_id == "job-1"
    assert restored.pause.metadata == {"kind": "job"}


def test_tool_result_rejects_interrupt_pause() -> None:
    with pytest.raises(ValueError, match="cannot interrupt"):
        ToolResult(
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
    result = ToolResult(
        parts=[ContentPart.text_part("created", metadata={"secret": "part"})],
        metadata={"secret": "result"},
        is_error=True,
    )

    message = result.to_message(ToolCall(id="call-1", name="tool"))

    assert message.metadata == {"is_error": True}
    assert message.parts[0].metadata == {}


def test_tool_result_from_dict_rejects_invalid_required_fields() -> None:
    with pytest.raises(KeyError):
        ToolResult.from_dict({"is_error": False})

    with pytest.raises(KeyError):
        ToolResult.from_dict({"parts": []})

    with pytest.raises(TypeError, match="is_error"):
        ToolResult.from_dict({"parts": [], "is_error": "false"})

    with pytest.raises(TypeError, match="metadata"):
        ToolResult.from_dict({"parts": [], "is_error": False, "metadata": None})

    with pytest.raises(TypeError, match="metadata"):
        ToolResult.from_dict({"parts": [], "is_error": False, "metadata": []})

    with pytest.raises(TypeError, match="metadata"):
        ToolResult.text("ok", metadata=cast(Any, []))

    with pytest.raises(TypeError, match="parts items"):
        ToolResult(parts=[cast(Any, object())])


def test_tool_spec_from_dict_rejects_schema_invalid_fields() -> None:
    with pytest.raises(TypeError, match="tool name"):
        ToolSpec.from_dict({"name": 1, "description": "tool", "input_schema": {}})

    with pytest.raises(KeyError):
        ToolSpec.from_dict({"name": "tool", "description": "tool"})

    with pytest.raises(TypeError, match="metadata"):
        ToolSpec.from_dict(
            {
                "name": "tool",
                "description": "tool",
                "input_schema": {},
                "metadata": None,
            }
        )


def test_tool_spec_constructor_rejects_invalid_core_types() -> None:
    with pytest.raises(TypeError, match="tool name"):
        ToolSpec(name=cast(Any, 1), description="tool", input_schema={})

    with pytest.raises(TypeError, match="tool description"):
        ToolSpec(name="tool", description=cast(Any, 1), input_schema={})


@pytest.mark.asyncio
async def test_tool_arguments_are_defensive_copies() -> None:
    call = ToolCall(id="call-1", name="mutate", arguments={"value": 1})
    result = await ToolRegistry([MutatingTool()]).execute(call, RuntimeContext())

    assert result.text_content == "ok"
    assert call.arguments == {"value": 1}


@pytest.mark.asyncio
async def test_tool_registry_rejects_invalid_tool_result_type() -> None:
    call = ToolCall(id="call-1", name="invalid_result", arguments={})

    with pytest.raises(TypeError, match="ToolResult"):
        await ToolRegistry([InvalidResultTool()]).execute(call, RuntimeContext())
