"""Strict resume input contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from agent_runtime.messages import Message
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import RESUMABLE_STATUSES, AgentState, AgentStatus, PauseState


def _empty_messages() -> tuple[Message, ...]:
    return ()


def _empty_metadata() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[object], value)


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_optional_non_empty_str(value: object, label: str) -> str | None:
    text = _expect_optional_str(value, label)
    if text == "":
        raise ValueError(f"{label} must not be empty")
    return text


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _expect_run_snapshot(value: object, label: str) -> RunSnapshot:
    if not isinstance(value, RunSnapshot):
        raise TypeError(f"{label} must be a RunSnapshot")
    return value


def _expect_pause_selector(value: object, label: str) -> PauseSelector:
    if not isinstance(value, PauseSelector):
        raise TypeError(f"{label} must be a PauseSelector or None")
    return value


def _validate_tool_call_context(state: AgentState) -> None:
    pending = [call.to_dict() for call in state.pending_tool_calls]
    pending_matched = False
    covered_tool_message_indexes: set[int] = set()

    for assistant_index, assistant_message in enumerate(state.messages):
        if assistant_message.role != "assistant" or not assistant_message.tool_calls:
            continue

        expected_ids = [call.id for call in assistant_message.tool_calls]
        completed_ids: list[str] = []
        completed_indexes: list[int] = []
        scan_index = assistant_index + 1
        while scan_index < len(state.messages):
            message = state.messages[scan_index]
            if message.role != "tool":
                break
            if message.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            completed_ids.append(message.tool_call_id)
            completed_indexes.append(scan_index)
            scan_index += 1

        completed_count = len(completed_ids)
        if completed_ids != expected_ids[:completed_count]:
            raise ValueError("completed tool messages must match assistant tool call order")
        covered_tool_message_indexes.update(completed_indexes)
        if completed_count == len(expected_ids):
            continue

        if scan_index < len(state.messages):
            if pending:
                raise ValueError("pending tool calls require contiguous tool messages")
            raise ValueError("assistant tool_calls require contiguous tool messages")

        unresolved = list(assistant_message.tool_calls[completed_count:])
        if [call.to_dict() for call in unresolved] != pending:
            if pending:
                raise ValueError("pending tool calls must match unresolved assistant tool calls")
            raise ValueError("assistant tool_calls require matching tool messages")
        pending_matched = True

    if pending and not pending_matched:
        raise ValueError("pending tool calls require assistant tool_calls history")
    for index, message in enumerate(state.messages):
        if message.role == "tool" and index not in covered_tool_message_indexes:
            raise ValueError("tool messages require preceding assistant tool_calls")


@dataclass(slots=True, frozen=True)
class PauseSelector:
    """Optional guard that ensures a resume targets the expected pause."""

    reason: str | None = None
    source: str | None = None
    wait_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _expect_optional_non_empty_str(self.reason, "pause selector reason")
        _expect_optional_non_empty_str(self.source, "pause selector source")
        _expect_optional_str(self.wait_id, "pause selector wait_id")
        metadata = _copy_mapping(_expect_mapping(self.metadata, "pause selector metadata"))
        if self.reason is None and self.source is None and self.wait_id is None and not metadata:
            raise ValueError("pause selector must set at least one field")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PauseSelector:
        known = {"reason", "source", "wait_id", "metadata"}
        _reject_unknown_keys(value, known, "pause selector")
        return cls(
            reason=_expect_optional_non_empty_str(value["reason"], "pause selector reason"),
            source=_expect_optional_non_empty_str(value["source"], "pause selector source"),
            wait_id=_expect_optional_str(value["wait_id"], "pause selector wait_id"),
            metadata=_expect_mapping(value["metadata"], "pause selector metadata"),
        )

    def matches(self, pause: PauseState) -> bool:
        if self.reason is not None and pause.reason != self.reason:
            return False
        if self.source is not None and pause.source != self.source:
            return False
        if self.wait_id is not None and pause.wait_id != self.wait_id:
            return False
        return all(
            key in pause.metadata and pause.metadata[key] == expected
            for key, expected in self.metadata.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "source": self.source,
            "wait_id": self.wait_id,
            "metadata": _copy_mapping(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class ResumeInput:
    """Typed input for resuming a durable runtime snapshot."""

    snapshot: RunSnapshot
    append_messages: Sequence[Message] = field(default_factory=_empty_messages)
    expected_pause: PauseSelector | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        raw_snapshot = _expect_run_snapshot(self.snapshot, "resume snapshot")
        snapshot = RunSnapshot.from_dict(raw_snapshot.to_dict())
        snapshot_state = snapshot.state
        raw_messages = _expect_sequence(self.append_messages, "resume append_messages")
        messages: list[Message] = []
        for message in raw_messages:
            if not isinstance(message, Message):
                raise TypeError("resume append_messages items must be Message")
            messages.append(Message.from_dict(message.to_dict()))
        if snapshot_state.status in {
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.LIMIT_EXCEEDED,
        }:
            raise ValueError("resume snapshot must not be terminal")
        if (
            snapshot_state.status is not AgentStatus.PAUSED
            and snapshot_state.status not in RESUMABLE_STATUSES
        ):
            raise ValueError("resume snapshot status must be paused, planning, or executing_tools")
        if snapshot_state.status is AgentStatus.PLANNING and snapshot_state.pending_tool_calls:
            raise ValueError("planning resume snapshot must not have pending tool calls")
        if (
            snapshot_state.status is AgentStatus.EXECUTING_TOOLS
            and not snapshot_state.pending_tool_calls
        ):
            raise ValueError("executing_tools resume snapshot requires pending tool calls")
        if snapshot_state.status is not AgentStatus.PAUSED:
            if messages:
                raise ValueError("append_messages are only valid when resuming a paused snapshot")
            if self.expected_pause is not None:
                raise ValueError("expected_pause is only valid when resuming a paused snapshot")
        elif snapshot_state.pause is None:
            raise ValueError("paused resume snapshot requires pause metadata")
        else:
            pause = snapshot_state.pause
            if pause.resume_status is AgentStatus.PLANNING and snapshot_state.pending_tool_calls:
                raise ValueError(
                    "paused resume snapshot that resumes to planning must not have pending "
                    "tool calls"
                )
            if (
                pause.resume_status is AgentStatus.EXECUTING_TOOLS
                and not snapshot_state.pending_tool_calls
            ):
                raise ValueError(
                    "paused resume snapshot that resumes to executing_tools requires pending "
                    "tool calls"
                )
            if messages and pause.resume_status is AgentStatus.EXECUTING_TOOLS:
                raise ValueError(
                    "append_messages are only valid when a paused snapshot resumes to planning"
                )
            if self.expected_pause is not None:
                expected_pause = _expect_pause_selector(
                    cast(object, self.expected_pause), "resume expected_pause"
                )
                if not expected_pause.matches(pause):
                    raise ValueError("resume input does not match paused snapshot")

        context_state = AgentState.from_dict(snapshot_state.to_dict())
        if snapshot_state.status is AgentStatus.PAUSED and snapshot_state.pause is not None:
            context_state.status = snapshot_state.pause.resume_status
            context_state.pause = None
            context_state.messages.extend(messages)
        _validate_tool_call_context(context_state)

        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "append_messages", tuple(messages))
        object.__setattr__(
            self,
            "metadata",
            _copy_mapping(_expect_mapping(self.metadata, "resume input metadata")),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResumeInput:
        known = {"snapshot", "append_messages", "expected_pause", "metadata"}
        _reject_unknown_keys(value, known, "resume input")
        raw_expected = value["expected_pause"]
        return cls(
            snapshot=RunSnapshot.from_dict(_expect_mapping(value["snapshot"], "resume snapshot")),
            append_messages=[
                Message.from_dict(_expect_mapping(message, "resume append message"))
                for message in _expect_sequence(value["append_messages"], "resume append_messages")
            ],
            expected_pause=None
            if raw_expected is None
            else PauseSelector.from_dict(_expect_mapping(raw_expected, "resume expected_pause")),
            metadata=_expect_mapping(value["metadata"], "resume metadata"),
        )

    def apply(self) -> tuple[AgentState, RunSnapshot]:
        """Return the working state and canonical snapshot for this resume."""

        snapshot = RunSnapshot.from_dict(self.snapshot.to_dict())
        state = AgentState.from_dict(snapshot.state.to_dict())
        if state.status is AgentStatus.PAUSED:
            if state.pause is None:
                raise ValueError("paused resume snapshot requires pause metadata")
            state.status = state.pause.resume_status
            state.pause = None
            state.error = None
            state.messages.extend(
                Message.from_dict(message.to_dict()) for message in self.append_messages
            )
        return state, snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "append_messages": [message.to_dict() for message in self.append_messages],
            "expected_pause": None
            if self.expected_pause is None
            else self.expected_pause.to_dict(),
            "metadata": _copy_mapping(self.metadata),
        }
