"""Runtime events."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any


def _empty_event_data() -> Mapping[str, Any]:
    return {}


class EventTypes:
    """Known core event type constants.

    Event type strings are intentionally open so integrations can add
    domain-specific events without changing the SDK core.
    """

    RUN_STARTED = "run_started"
    STATE_CHANGED = "state_changed"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"
    MODEL_COMPLETED = "model_completed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ERROR = "error"
    RUN_COMPLETED = "run_completed"


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
        if not self.type:
            raise ValueError("event type must not be empty")
        if self.sequence < 0:
            raise ValueError("event sequence must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "data": deepcopy(dict(self.data)),
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
        if not self.type:
            raise ValueError("queued event type must not be empty")


class EventEmitter:
    """Append custom events from RuntimeHook.on_event."""

    __slots__ = ("_queue",)

    def __init__(self) -> None:
        self._queue: list[QueuedEvent] = []

    def emit(self, event_type: EventType, data: Mapping[str, Any] | None = None) -> None:
        self._queue.append(QueuedEvent(event_type, deepcopy(dict(data or {}))))

    def drain(self) -> tuple[QueuedEvent, ...]:
        events = tuple(self._queue)
        self._queue.clear()
        return events
