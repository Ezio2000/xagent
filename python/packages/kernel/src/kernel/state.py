"""Agent state types."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast

from kernel.messages import ContentPart, Message, ToolCall
from kernel.models import ModelUsage
from kernel.status import RESUMABLE_STATUSES, TERMINAL_STATUSES, AgentStatus


def _empty_pending_tool_calls() -> list[ToolCall]:
    return []


def _empty_final_parts() -> list[ContentPart]:
    return []


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return cast(list[object], value)


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _usage_to_dict(usage: ModelUsage | None) -> dict[str, Any] | None:
    return None if usage is None else usage.to_dict()


def _usage_from_value(value: object, label: str) -> ModelUsage | None:
    if value is None:
        return None
    return ModelUsage.from_dict(_expect_mapping(value, label))


@dataclass(slots=True)
class PauseState:
    """Serializable pause metadata for a resumable run boundary."""

    reason: str
    resume_status: AgentStatus
    source: str = "host"
    wait_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        _expect_str(self.reason, "pause reason")
        _expect_optional_str(self.wait_id, "pause wait_id")
        _expect_str(self.source, "pause source")
        if not isinstance(cast(object, self.resume_status), AgentStatus):
            raise TypeError("pause resume_status must be an AgentStatus")
        if not self.reason:
            raise ValueError("pause reason must not be empty")
        if not self.source:
            raise ValueError("pause source must not be empty")
        if self.resume_status not in RESUMABLE_STATUSES:
            raise ValueError("pause resume_status must be planning or executing_tools")
        self.metadata = _copy_mapping(_expect_mapping(self.metadata, "pause metadata"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PauseState:
        known = {"reason", "resume_status", "source", "wait_id", "metadata"}
        _reject_unknown_keys(value, known, "pause state")
        raw_wait_id = value["wait_id"]
        return cls(
            reason=_expect_str(value["reason"], "pause reason"),
            resume_status=AgentStatus(_expect_str(value["resume_status"], "pause resume_status")),
            source=_expect_str(value["source"], "pause source"),
            wait_id=_expect_optional_str(raw_wait_id, "pause wait_id"),
            metadata=_expect_mapping(value["metadata"], "pause metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "resume_status": self.resume_status.value,
            "source": self.source,
            "wait_id": self.wait_id,
            "metadata": _copy_mapping(self.metadata),
        }


@dataclass(slots=True)
class AgentState:
    """Mutable working state owned by AgentLoop."""

    status: AgentStatus
    messages: list[Message]
    pending_tool_calls: list[ToolCall] = field(default_factory=_empty_pending_tool_calls)
    iterations: int = 0
    total_tool_calls: int = 0
    total_usage: ModelUsage | None = None
    final_parts: list[ContentPart] = field(default_factory=_empty_final_parts)
    error: str | None = None
    pause: PauseState | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.status), AgentStatus):
            raise TypeError("agent state status must be an AgentStatus")
        self.messages = [
            Message.from_dict(message.to_dict())
            if isinstance(cast(object, message), Message)
            else _raise_type("agent state messages items must be Message")
            for message in self.messages
        ]
        self.pending_tool_calls = [
            ToolCall.from_dict(call.to_dict())
            if isinstance(cast(object, call), ToolCall)
            else _raise_type("agent state pending_tool_calls items must be ToolCall")
            for call in self.pending_tool_calls
        ]
        self.iterations = _expect_int(self.iterations, "agent state iterations")
        self.total_tool_calls = _expect_int(self.total_tool_calls, "agent state total_tool_calls")
        if self.total_usage is not None and not isinstance(
            cast(object, self.total_usage), ModelUsage
        ):
            raise TypeError("agent state total_usage must be ModelUsage or None")
        if self.total_usage is not None:
            self.total_usage = ModelUsage.from_dict(self.total_usage.to_dict())
        self.final_parts = [
            ContentPart.from_dict(part.to_dict())
            if isinstance(cast(object, part), ContentPart)
            else _raise_type("agent state final_parts items must be ContentPart")
            for part in self.final_parts
        ]
        self.error = _expect_optional_str(self.error, "agent state error")
        if self.status is AgentStatus.PAUSED and self.pause is None:
            raise ValueError("paused state requires pause metadata")
        if self.status is not AgentStatus.PAUSED and self.pause is not None:
            raise ValueError("pause metadata is only valid for paused state")
        if self.pause is not None:
            self.pause = PauseState.from_dict(self.pause.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentState:
        known = {
            "status",
            "messages",
            "pending_tool_calls",
            "iterations",
            "total_tool_calls",
            "total_usage",
            "final_parts",
            "error",
            "pause",
        }
        _reject_unknown_keys(value, known, "agent state")
        raw_pause = value["pause"]
        return cls(
            status=AgentStatus(_expect_str(value["status"], "agent state status")),
            messages=[
                Message.from_dict(_expect_mapping(message, "agent state message"))
                for message in _expect_sequence(value["messages"], "agent state messages")
            ],
            pending_tool_calls=[
                ToolCall.from_dict(_expect_mapping(call, "agent state pending tool call"))
                for call in _expect_sequence(
                    value["pending_tool_calls"], "agent state pending_tool_calls"
                )
            ],
            iterations=_expect_int(value["iterations"], "agent state iterations"),
            total_tool_calls=_expect_int(value["total_tool_calls"], "agent state total_tool_calls"),
            total_usage=_usage_from_value(value["total_usage"], "agent state total_usage"),
            final_parts=[
                ContentPart.from_dict(_expect_mapping(part, "agent state final part"))
                for part in _expect_sequence(value["final_parts"], "agent state final_parts")
            ],
            error=_expect_optional_str(value["error"], "agent state error"),
            pause=None
            if raw_pause is None
            else PauseState.from_dict(_expect_mapping(raw_pause, "agent state pause")),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "message_count": len(self.messages),
            "message_roles": [message.role for message in self.messages],
            "pending_tool_call_count": len(self.pending_tool_calls),
            "pending_tool_call_ids": [call.id for call in self.pending_tool_calls],
            "iterations": self.iterations,
            "total_tool_calls": self.total_tool_calls,
            "total_usage": _usage_to_dict(self.total_usage),
            "has_final": bool(self.final_parts),
            "final_part_count": len(self.final_parts),
            "error": self.error,
            "pause": None if self.pause is None else self.pause.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status.value,
            "messages": [message.to_dict() for message in self.messages],
            "pending_tool_calls": [call.to_dict() for call in self.pending_tool_calls],
            "iterations": self.iterations,
            "total_tool_calls": self.total_tool_calls,
            "total_usage": _usage_to_dict(self.total_usage),
            "final_parts": [part.to_dict() for part in self.final_parts],
            "error": self.error,
            "pause": None if self.pause is None else self.pause.to_dict(),
        }
        return data


def _raise_type(message: str) -> NoReturn:
    raise TypeError(message)
