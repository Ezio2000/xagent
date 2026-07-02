from __future__ import annotations

from typing import Any, cast

import pytest

from agent_runtime import ContentPart, Message, ToolCall


def test_message_from_dict_rejects_missing_required_fields() -> None:
    with pytest.raises(KeyError):
        Message.from_dict({"role": "user"})

    with pytest.raises(TypeError, match="role"):
        Message.from_dict({"role": 123, "parts": []})


def test_tool_call_from_dict_rejects_missing_required_arguments() -> None:
    with pytest.raises(KeyError):
        ToolCall.from_dict({"id": "call-1", "name": "tool"})

    with pytest.raises(TypeError, match="id"):
        ToolCall.from_dict({"id": 1, "name": "tool", "arguments": {}})

    with pytest.raises(TypeError, match="arguments"):
        ToolCall.from_dict({"id": "call-1", "name": "tool", "arguments": None})


def test_assistant_tool_call_ids_must_be_unique() -> None:
    calls = [
        ToolCall(id="call-1", name="tool", arguments={}),
        ToolCall(id="call-1", name="tool", arguments={}),
    ]

    with pytest.raises(ValueError, match="unique"):
        Message.assistant([], calls)

    with pytest.raises(ValueError, match="unique"):
        Message.from_dict(
            {
                "role": "assistant",
                "parts": [],
                "tool_calls": [call.to_dict() for call in calls],
            }
        )


def test_message_constructors_reject_invalid_core_types() -> None:
    with pytest.raises(TypeError, match="content part text"):
        ContentPart.text_part(cast(Any, 123))

    with pytest.raises(TypeError, match="tool call id"):
        ToolCall(id=cast(Any, 1), name="tool")

    with pytest.raises(TypeError, match="message role"):
        Message(role=cast(Any, 1), parts=[])


def test_message_from_dict_rejects_unknown_wire_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ContentPart.from_dict({"type": "text", "text": "hello", "provider": {}})

    with pytest.raises(ValueError, match="unknown"):
        ToolCall.from_dict({"id": "call-1", "name": "tool", "arguments": {}, "provider": {}})

    with pytest.raises(ValueError, match="unknown"):
        Message.from_dict({"role": "user", "parts": [], "provider": {}})


def test_content_part_from_dict_rejects_schema_invalid_optional_fields() -> None:
    with pytest.raises(TypeError, match="type"):
        ContentPart.from_dict({"type": 1})

    with pytest.raises(TypeError, match="metadata"):
        ContentPart.from_dict({"type": "text", "text": "hello", "metadata": None})
