"""Conversation history ports and public trust-boundary validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jharness.kernel._history import analyze_history
from jharness.kernel._validation import expect_instance_tuple, expect_non_empty_str, freeze_mapping
from jharness.kernel.messages import Message
from jharness.kernel.state import RunState

if TYPE_CHECKING:
    from jharness.kernel.snapshot import RunSnapshot


@dataclass(frozen=True, slots=True)
class HistoryRewrite:
    """A reducer proposal that may replace history at a plan boundary."""

    messages: tuple[Message, ...]
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        messages = expect_instance_tuple(self.messages, Message, "history rewrite messages")
        if not messages:
            raise ValueError("history rewrite requires messages")
        expect_non_empty_str(self.reason, "history rewrite reason")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "history rewrite metadata"),
        )


@runtime_checkable
class HistoryReducer(Protocol):
    """Propose a valid history with no more messages at a planning boundary."""

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None: ...


def validate_history(history: Sequence[Message], state: RunState) -> None:
    """Fully validate ordered tool linkage against one lifecycle state."""

    analyze_history(history, state)
