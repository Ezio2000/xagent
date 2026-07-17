"""Immutable provider-neutral conversation values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jharness.kernel._validation import (
    expect_instance,
    expect_instance_tuple,
    expect_non_empty_str,
    expect_optional_nonnegative_int,
    expect_optional_str,
    expect_str,
    freeze_mapping,
)

if TYPE_CHECKING:
    from jharness.kernel.tools import ToolOutcome


_REGULAR_ROLES = frozenset({"system", "user", "external"})
_ROLES = frozenset({*_REGULAR_ROLES, "assistant", "tool"})


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """One authoritative host-owned artifact reference."""

    ref: str
    media_type: str | None = None
    name: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.ref, "artifact ref")
        for value, label in (
            (self.media_type, "artifact media_type"),
            (self.name, "artifact name"),
            (self.sha256, "artifact sha256"),
        ):
            if value is not None:
                expect_non_empty_str(value, label)
        expect_optional_nonnegative_int(self.size_bytes, "artifact size_bytes")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "artifact metadata"))


@dataclass(frozen=True, slots=True)
class ContentPart:
    """One immutable text, artifact, or provider-opaque content part."""

    type: str
    text: str | None = None
    uri: str | None = None
    artifact: ArtifactRef | None = None
    media_type: str | None = None
    name: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        part_type = expect_non_empty_str(self.type, "content part type")
        text = expect_optional_str(self.text, "content part text")
        for value, label in (
            (self.uri, "content part uri"),
            (self.media_type, "content part media_type"),
            (self.name, "content part name"),
        ):
            if value is not None:
                expect_non_empty_str(value, label)
        if self.artifact is not None:
            expect_instance(self.artifact, ArtifactRef, "content part artifact")
        data = freeze_mapping(self.data, "content part data")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "content metadata"))
        if part_type == "text":
            self._validate_text(text, data)
        elif part_type == "artifact":
            self._validate_artifact(data)
        elif self.artifact is not None:
            raise ValueError("only artifact content may carry an artifact reference")

    def _validate_text(self, text: str | None, data: Mapping[str, Any]) -> None:
        if text is None:
            raise ValueError("text content part requires text")
        if (
            any(
                value is not None for value in (self.uri, self.artifact, self.media_type, self.name)
            )
            or data
        ):
            raise ValueError("text content part cannot carry opaque or artifact fields")

    def _validate_artifact(self, data: Mapping[str, Any]) -> None:
        if self.artifact is None:
            raise ValueError("artifact content part requires artifact")
        if (
            any(value is not None for value in (self.text, self.uri, self.media_type, self.name))
            or data
        ):
            raise ValueError("artifact content part cannot carry duplicate or opaque fields")

    @classmethod
    def text_part(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(type="text", text=text, metadata={} if metadata is None else metadata)

    @classmethod
    def artifact_part(
        cls,
        artifact: ArtifactRef,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContentPart:
        return cls(
            type="artifact",
            artifact=artifact,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-requested tool invocation."""

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.id, "tool call id")
        expect_non_empty_str(self.name, "tool call name")
        object.__setattr__(self, "arguments", freeze_mapping(self.arguments, "tool arguments"))


@dataclass(frozen=True, slots=True)
class TaskRef:
    """Host-owned task reference included in a model-visible tool outcome."""

    id: str
    status: str
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.id, "task id")
        expect_non_empty_str(self.status, "task status")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "task metadata"))


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """Stable portable error details."""

    code: str
    message: str

    def __post_init__(self) -> None:
        expect_non_empty_str(self.code, "error code")
        expect_non_empty_str(self.message, "error message")


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable conversation message with role-specific invariants."""

    role: str
    parts: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    outcome: ToolOutcome | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        role = expect_str(self.role, "message role")
        if role not in _ROLES:
            raise ValueError(f"unsupported message role: {role}")
        parts = expect_instance_tuple(self.parts, ContentPart, "message parts")
        calls = expect_instance_tuple(self.tool_calls, ToolCall, "message tool_calls")
        tool_call_id = expect_optional_str(self.tool_call_id, "tool_call_id")
        outcome = None if self.outcome is None else _expect_tool_outcome(self.outcome)
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "message metadata"))
        if role in _REGULAR_ROLES:
            _validate_regular_message(role, parts, calls, tool_call_id, outcome)
        elif role == "assistant":
            _validate_assistant_message(parts, calls, tool_call_id, outcome)
        else:
            _validate_tool_message(parts, calls, tool_call_id, outcome)

    @classmethod
    def system(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            "system",
            (ContentPart.text_part(text),),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def user(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            "user",
            (ContentPart.text_part(text),),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def external(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            "external",
            (ContentPart.text_part(text),),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def assistant(
        cls,
        parts: Sequence[ContentPart] = (),
        *,
        tool_calls: Sequence[ToolCall] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            "assistant",
            tuple(parts),
            tuple(tool_calls),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        outcome: ToolOutcome,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Message:
        return cls(
            "tool",
            tool_call_id=tool_call_id,
            outcome=outcome,
            metadata={} if metadata is None else metadata,
        )


def _expect_tool_outcome(value: object) -> ToolOutcome:
    from jharness.kernel.tools import ToolOutcome

    if not isinstance(value, ToolOutcome):
        raise TypeError("message outcome must be a ToolOutcome")
    return value


def _validate_regular_message(
    role: str,
    parts: tuple[ContentPart, ...],
    calls: tuple[ToolCall, ...],
    tool_call_id: str | None,
    outcome: ToolOutcome | None,
) -> None:
    if not parts:
        raise ValueError(f"{role} message requires at least one part")
    if calls or tool_call_id is not None or outcome is not None:
        raise ValueError(f"{role} message cannot carry tool fields")


def _validate_assistant_message(
    parts: tuple[ContentPart, ...],
    calls: tuple[ToolCall, ...],
    tool_call_id: str | None,
    outcome: ToolOutcome | None,
) -> None:
    if not parts and not calls:
        raise ValueError("assistant message requires parts or tool calls")
    if tool_call_id is not None or outcome is not None:
        raise ValueError("assistant message cannot carry tool outcome fields")
    ids = [call.id for call in calls]
    if len(ids) != len(set(ids)):
        raise ValueError("assistant tool call ids must be unique")


def _validate_tool_message(
    parts: tuple[ContentPart, ...],
    calls: tuple[ToolCall, ...],
    tool_call_id: str | None,
    outcome: ToolOutcome | None,
) -> None:
    if parts or calls:
        raise ValueError("tool message content is owned by its outcome")
    if tool_call_id is None:
        raise ValueError("tool message requires tool_call_id")
    expect_non_empty_str(tool_call_id, "tool_call_id")
    if outcome is None:
        raise ValueError("tool message requires outcome")
