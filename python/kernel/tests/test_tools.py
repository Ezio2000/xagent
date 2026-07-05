from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import (
    BackgroundTask,
    ContentPart,
    PauseRequest,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
    normalized_tool_risk,
)


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
    }


def test_tool_scheduling_annotations_are_not_approval_risk() -> None:
    assert (
        normalized_tool_risk({"parallel_safe": True, "read_only": True, "idempotent": True}) == {}
    )


def test_tool_risk_annotation_rejects_invalid_standard_fields() -> None:
    with pytest.raises(ValueError, match="filesystem"):
        ToolSpec(
            name="risk",
            description="Risky tool.",
            input_schema={"type": "object", "properties": {}},
            annotations={"risk": {"filesystem": ""}},
        )


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

    message = result.to_message(ToolCall(id="call-1", name="tool"))

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
