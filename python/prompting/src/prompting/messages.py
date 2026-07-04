"""Message construction helpers built on kernel DTOs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kernel import ContentPart, Message, ToolCall


def system_text(text: str, *, metadata: Mapping[str, Any] | None = None) -> Message:
    return Message.system([ContentPart.text_part(text)], metadata=metadata)


def user_text(text: str, *, metadata: Mapping[str, Any] | None = None) -> Message:
    return Message.user([ContentPart.text_part(text)], metadata=metadata)


def assistant_text(
    text: str,
    tool_calls: Sequence[ToolCall] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Message:
    return Message.assistant([ContentPart.text_part(text)], tool_calls, metadata=metadata)


def tool_text(
    text: str,
    tool_call_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Message:
    return Message.tool([ContentPart.text_part(text)], tool_call_id, metadata=metadata)


def external_text(
    text: str,
    *,
    insert_id: str,
    source: str,
    correlation_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Message:
    return Message.external(
        [ContentPart.text_part(text)],
        insert_id=insert_id,
        source=source,
        correlation_id=correlation_id,
        metadata=metadata,
    )
