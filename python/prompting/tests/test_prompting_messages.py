from __future__ import annotations

from kernel import ToolCall
from prompting import assistant_text, external_text, system_text, tool_text, user_text


def test_prompting_text_helpers_create_expected_roles() -> None:
    assert system_text("policy").role == "system"
    assert user_text("hello").role == "user"
    assert tool_text("ok", "call-1").role == "tool"


def test_assistant_text_preserves_tool_calls() -> None:
    message = assistant_text(
        "using tool",
        [ToolCall(id="call-1", name="search", arguments={"q": "x"})],
    )

    assert message.role == "assistant"
    assert message.text == "using tool"
    assert message.tool_calls[0].id == "call-1"


def test_external_text_sets_insert_metadata() -> None:
    message = external_text(
        "callback",
        insert_id="insert-1",
        source="webhook",
        correlation_id="corr-1",
    )

    assert message.role == "external"
    assert message.metadata["insert_id"] == "insert-1"
    assert message.metadata["source"] == "webhook"
    assert message.metadata["correlation_id"] == "corr-1"
