"""Immutable values shared by the Agent preset tools and their Host backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from jharness.kernel import ErrorInfo

AgentStatus: TypeAlias = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]

_AGENT_STATUSES = frozenset[AgentStatus]({"queued", "running", "completed", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """One validated request to create or recover a child Agent."""

    description: str
    prompt: str
    background: bool = False

    def __post_init__(self) -> None:
        _non_empty_string(self.description, "description")
        _non_empty_string(self.prompt, "prompt")
        _boolean(self.background, "background")


@dataclass(frozen=True, slots=True)
class AgentSnapshot:
    """One immutable Host observation of an Agent's current state."""

    agent_id: str
    description: str
    status: AgentStatus
    background: bool
    result: str | None = None
    error: ErrorInfo | None = None
    cancellation_requested: bool = False

    def __post_init__(self) -> None:
        _non_empty_string(self.agent_id, "agent_id")
        _non_empty_string(self.description, "description")
        status = _agent_status(self.status)
        _boolean(self.background, "background")
        result = cast(object, self.result)
        if result is not None and not isinstance(result, str):
            raise TypeError("result must be a string or None")
        error = cast(object, self.error)
        if error is not None and not isinstance(error, ErrorInfo):
            raise TypeError("error must be an ErrorInfo or None")
        cancellation_requested = _boolean(
            self.cancellation_requested,
            "cancellation_requested",
        )
        _validate_snapshot_state(status, result, error, cancellation_requested)


class AgentBackendError(Exception):
    """A stable recoverable failure reported by the Host Agent backend."""

    __slots__ = ("code", "message")

    code: str
    message: str

    def __init__(self, code: str, message: str) -> None:
        self.code = _non_empty_string(code, "code")
        self.message = _non_empty_string(message, "message")
        super().__init__(message)


def _agent_status(value: object) -> AgentStatus:
    if not isinstance(value, str):
        raise TypeError("status must be a string")
    if value not in _AGENT_STATUSES:
        raise ValueError(f"unsupported Agent status: {value}")
    return value


def _validate_snapshot_state(
    status: AgentStatus,
    result: str | None,
    error: ErrorInfo | None,
    cancellation_requested: bool,
) -> None:
    if status == "completed":
        _validate_completed(result, error, cancellation_requested)
        return
    if status == "failed":
        _validate_failed(result, error, cancellation_requested)
        return
    if result is not None:
        raise ValueError(f"{status} Agent snapshot cannot include result")
    if error is not None:
        raise ValueError(f"{status} Agent snapshot cannot include error")
    if status == "cancelled" and not cancellation_requested:
        raise ValueError("cancelled Agent snapshot requires cancellation_requested=true")


def _validate_completed(
    result: str | None,
    error: ErrorInfo | None,
    cancellation_requested: bool,
) -> None:
    if result is None:
        raise ValueError("completed Agent snapshot requires result")
    if error is not None:
        raise ValueError("completed Agent snapshot cannot include error")
    if cancellation_requested:
        raise ValueError("completed Agent snapshot cannot have cancellation_requested=true")


def _validate_failed(
    result: str | None,
    error: ErrorInfo | None,
    cancellation_requested: bool,
) -> None:
    if error is None:
        raise ValueError("failed Agent snapshot requires error")
    if result is not None:
        raise ValueError("failed Agent snapshot cannot include result")
    if cancellation_requested:
        raise ValueError("failed Agent snapshot cannot have cancellation_requested=true")


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be bool")
    return value


__all__ = [
    "AgentBackendError",
    "AgentRequest",
    "AgentSnapshot",
    "AgentStatus",
]
