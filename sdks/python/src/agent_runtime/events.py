"""Runtime events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time
from typing import Any, cast

from agent_runtime._frozen import freeze_value, thaw_value


def _empty_event_data() -> Mapping[str, Any]:
    return {}


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _expect_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    return float(value)


def _freeze_event_value(value: object) -> object:
    return freeze_value(value, error_message="event data is immutable")


def _thaw_event_value(value: object) -> object:
    return thaw_value(value)


class EventTypes:
    """Known core event type constants.

    Event type strings are intentionally open so integrations can add
    domain-specific events without changing the SDK core.
    """

    RUN_STARTED = "run_started"
    STATE_CHANGED = "state_changed"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"
    MODEL_ERROR = "model_error"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_CANCEL_REQUESTED = "tool_cancel_requested"
    TOOL_COMPLETED = "tool_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_COMPLETED = "approval_completed"
    BACKGROUND_TASK_STARTED = "background_task_started"
    BACKGROUND_TASK_UPDATED = "background_task_updated"
    BACKGROUND_TASK_COMPLETED = "background_task_completed"
    CHILD_RUN_STARTED = "child_run_started"
    CHILD_RUN_COMPLETED = "child_run_completed"
    CONVERSATION_INSERTED = "conversation_inserted"
    PAUSE_REQUESTED = "pause_requested"
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ERROR = "error"
    RUN_PAUSED = "run_paused"
    RUN_COMPLETED = "run_completed"


CORE_EVENT_TYPES = frozenset(
    {
        EventTypes.RUN_STARTED,
        EventTypes.STATE_CHANGED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_DELTA,
        EventTypes.MODEL_ERROR,
        EventTypes.MODEL_COMPLETED,
        EventTypes.TOOL_STARTED,
        EventTypes.TOOL_PROGRESS,
        EventTypes.TOOL_CANCEL_REQUESTED,
        EventTypes.TOOL_COMPLETED,
        EventTypes.APPROVAL_REQUESTED,
        EventTypes.APPROVAL_COMPLETED,
        EventTypes.BACKGROUND_TASK_STARTED,
        EventTypes.BACKGROUND_TASK_UPDATED,
        EventTypes.BACKGROUND_TASK_COMPLETED,
        EventTypes.CHILD_RUN_STARTED,
        EventTypes.CHILD_RUN_COMPLETED,
        EventTypes.CONVERSATION_INSERTED,
        EventTypes.PAUSE_REQUESTED,
        EventTypes.CHECKPOINT,
        EventTypes.FINAL,
        EventTypes.ERROR,
        EventTypes.RUN_PAUSED,
        EventTypes.RUN_COMPLETED,
    }
)

EventType = str


@dataclass(slots=True, frozen=True)
class AgentEvent:
    """JSON-serializable runtime event envelope."""

    type: EventType
    data: Mapping[str, Any] = field(default_factory=_empty_event_data)
    run_id: str = ""
    sequence: int = 0
    created_at: float = field(default_factory=time)
    schema_version: str = "v0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _expect_str(self.type, "event type"))
        object.__setattr__(self, "run_id", _expect_str(self.run_id, "event run_id"))
        object.__setattr__(
            self,
            "sequence",
            _expect_int(self.sequence, "event sequence"),
        )
        object.__setattr__(self, "created_at", _expect_number(self.created_at, "event created_at"))
        object.__setattr__(
            self,
            "schema_version",
            _expect_str(self.schema_version, "event schema_version"),
        )
        if not self.type:
            raise ValueError("event type must not be empty")
        if not self.run_id:
            raise ValueError("event run_id must not be empty")
        if self.sequence < 0:
            raise ValueError("event sequence must be >= 0")
        if not self.schema_version:
            raise ValueError("event schema_version must not be empty")
        object.__setattr__(
            self,
            "data",
            _freeze_event_value(_expect_mapping(self.data, "event data")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": cast(dict[str, Any], _thaw_event_value(self.data)),
            "run_id": self.run_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


@dataclass(slots=True, frozen=True)
class QueuedEvent:
    """Event requested by RuntimeHook.on_event."""

    type: EventType
    data: Mapping[str, Any] = field(default_factory=_empty_event_data)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _expect_str(self.type, "queued event type"))
        if not self.type:
            raise ValueError("queued event type must not be empty")
        object.__setattr__(
            self,
            "data",
            _freeze_event_value(_expect_mapping(self.data, "queued event data")),
        )


class EventEmitter:
    """Append custom events from RuntimeHook.on_event."""

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: list[QueuedEvent] = []

    def emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        if event_type in CORE_EVENT_TYPES:
            raise ValueError(f"core event type is runtime-owned: {event_type}")
        self._queue.append(QueuedEvent(event_type, {} if data is None else data))

    def drain(self) -> tuple[QueuedEvent, ...]:
        events = tuple(self._queue)
        self._queue.clear()
        return events
