"""Message fixtures for controlled runtime tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kernel import Message
from prompting import assistant_text, external_text, system_text, tool_text, user_text


def user_message(text: str, *, metadata: Mapping[str, Any] | None = None) -> Message:
    """Return a user text message fixture."""

    return user_text(text, metadata=metadata)


__all__ = [
    "assistant_text",
    "external_text",
    "system_text",
    "tool_text",
    "user_message",
    "user_text",
]
