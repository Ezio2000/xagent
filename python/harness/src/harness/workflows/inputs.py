"""Input normalization for high-level agent harness workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

from kernel import Message
from prompting import user_text

HarnessInput: TypeAlias = str | Message | Sequence[Message]
OptionalHarnessInput: TypeAlias = HarnessInput | None


def normalize_messages(value: object) -> tuple[Message, ...]:
    """Normalize common host inputs into defensive-copy kernel messages."""

    if isinstance(value, str):
        return (user_text(value),)
    if isinstance(value, Message):
        return (Message.from_dict(value.to_dict()),)
    if isinstance(value, bytes):
        raise TypeError("harness input must be text or Message values, not bytes")
    if isinstance(value, Sequence):
        messages: list[Message] = []
        for message in cast(Sequence[object], value):
            if not isinstance(message, Message):
                raise TypeError("harness input sequence items must be Message")
            messages.append(Message.from_dict(message.to_dict()))
        return tuple(messages)
    raise TypeError("harness input must be str, Message, or a sequence of Message")


def normalize_optional_messages(value: object | None) -> tuple[Message, ...]:
    """Normalize optional input used by resume workflows."""

    if value is None:
        return ()
    return normalize_messages(value)
