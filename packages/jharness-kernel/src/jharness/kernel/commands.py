"""Immutable start, continue, and resume request values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias

from jharness.kernel._history import analyze_history
from jharness.kernel._validation import (
    expect_instance,
    expect_instance_tuple,
    expect_optional_str,
    freeze_mapping,
)
from jharness.kernel.checkpoint import Checkpoint
from jharness.kernel.context import RunContext
from jharness.kernel.errors import RequestError
from jharness.kernel.history import RunHistory
from jharness.kernel.messages import Message
from jharness.kernel.state import Planning, Suspended, ToolsPending


@dataclass(frozen=True, slots=True)
class SuspensionSelector:
    """Optional optimistic guard for the suspension being resumed."""

    reason: str | None = None
    source: str | None = None
    wait_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        for value, label in (
            (self.reason, "selector reason"),
            (self.source, "selector source"),
            (self.wait_id, "selector wait_id"),
        ):
            if expect_optional_str(value, label) == "":
                raise ValueError(f"{label} must not be empty")
        metadata = freeze_mapping(self.metadata, "selector metadata")
        if self.reason is None and self.source is None and self.wait_id is None and not metadata:
            raise ValueError("suspension selector must set at least one field")
        object.__setattr__(self, "metadata", metadata)

    def matches(self, suspended: Suspended) -> bool:
        suspension = suspended.suspension
        return (
            (self.reason is None or self.reason == suspension.reason)
            and (self.source is None or self.source == suspension.source)
            and (self.wait_id is None or self.wait_id == suspension.wait_id)
            and all(
                key in suspension.metadata and suspension.metadata[key] == value
                for key, value in self.metadata.items()
            )
        )


@dataclass(frozen=True, slots=True)
class StartRequest:
    messages: RunHistory
    context: RunContext | None = None

    kind: ClassVar[Literal["start"]] = "start"

    def __post_init__(self) -> None:
        expect_instance(self.messages, RunHistory, "start messages")
        messages, _ = analyze_history(
            self.messages,
            Planning(),
            label="start messages",
            empty_message="start request requires messages",
        )
        if self.context is not None:
            expect_instance(self.context, RunContext, "start context")
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class ContinueRequest:
    checkpoint: Checkpoint

    kind: ClassVar[Literal["continue"]] = "continue"

    def __post_init__(self) -> None:
        checkpoint = expect_instance(self.checkpoint, Checkpoint, "continue checkpoint")
        if not isinstance(checkpoint.snapshot.state, Planning | ToolsPending):
            raise ValueError("continue request requires an active checkpoint")


@dataclass(frozen=True, slots=True)
class ResumeRequest:
    checkpoint: Checkpoint
    selector: SuspensionSelector | None = None
    append_messages: tuple[Message, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    kind: ClassVar[Literal["resume"]] = "resume"

    def __post_init__(self) -> None:
        checkpoint = expect_instance(self.checkpoint, Checkpoint, "resume checkpoint")
        state = checkpoint.snapshot.state
        if not isinstance(state, Suspended):
            raise ValueError("resume request requires a suspended checkpoint")
        if self.selector is not None:
            expect_instance(self.selector, SuspensionSelector, "resume selector")
            if not self.selector.matches(state):
                raise RequestError(
                    "suspension_mismatch",
                    "resume selector does not match the suspended checkpoint",
                )
        messages = expect_instance_tuple(self.append_messages, Message, "resume append_messages")
        if messages and isinstance(state.resume_to, ToolsPending):
            raise RequestError(
                "messages_require_planning",
                "resume messages require a planning continuation",
            )
        if any(message.role not in {"system", "user", "external"} for message in messages):
            raise ValueError("resume messages must use a regular role")
        object.__setattr__(self, "append_messages", messages)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "resume metadata"))


RunRequest: TypeAlias = StartRequest | ContinueRequest | ResumeRequest


__all__ = [
    "ContinueRequest",
    "ResumeRequest",
    "RunRequest",
    "StartRequest",
    "SuspensionSelector",
]
