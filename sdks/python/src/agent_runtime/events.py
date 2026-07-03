"""Runtime events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, NoReturn, SupportsIndex, cast


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


class _FrozenList(list[Any]):
    def _readonly(self) -> NoReturn:
        raise TypeError("event data is immutable")

    def __setitem__(self, key: SupportsIndex | slice, value: Any) -> None:
        _ = key, value
        self._readonly()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        _ = key
        self._readonly()

    def __iadd__(self, value: Iterable[Any]) -> _FrozenList:
        _ = value
        self._readonly()

    def __imul__(self, value: SupportsIndex) -> _FrozenList:
        _ = value
        self._readonly()

    def append(self, item: Any) -> None:
        _ = item
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def extend(self, items: Iterable[Any]) -> None:
        _ = items
        self._readonly()

    def insert(self, index: SupportsIndex, item: Any) -> None:
        _ = index, item
        self._readonly()

    def pop(self, index: SupportsIndex = -1) -> Any:
        _ = index
        self._readonly()

    def remove(self, item: Any) -> None:
        _ = item
        self._readonly()

    def reverse(self) -> None:
        self._readonly()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        _ = key, reverse
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> list[Any]:
        return [deepcopy(item, memo) for item in self]


class _FrozenDict(dict[str, Any]):
    def _readonly(self) -> NoReturn:
        raise TypeError("event data is immutable")

    def __setitem__(self, key: str, value: Any) -> None:
        _ = key, value
        self._readonly()

    def __delitem__(self, key: str) -> None:
        _ = key
        self._readonly()

    def clear(self) -> None:
        self._readonly()

    def pop(self, key: str, default: Any = None) -> Any:
        _ = key, default
        self._readonly()

    def popitem(self) -> tuple[str, Any]:
        self._readonly()

    def setdefault(self, key: str, default: Any = None) -> Any:
        _ = key, default
        self._readonly()

    def update(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        self._readonly()

    def __ior__(self, value: object) -> _FrozenDict:
        _ = value
        self._readonly()

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[str, Any]:
        return {deepcopy(key, memo): deepcopy(value, memo) for key, value in self.items()}


def _freeze_event_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict(
            {key: _freeze_event_value(item) for key, item in cast(Mapping[str, Any], value).items()}
        )
    if isinstance(value, list | tuple):
        return _FrozenList([_freeze_event_value(item) for item in cast(Sequence[object], value)])
    return deepcopy(value)


def _thaw_event_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thaw_event_value(item) for key, item in cast(Mapping[str, Any], value).items()
        }
    if isinstance(value, list | tuple):
        return [_thaw_event_value(item) for item in cast(Sequence[object], value)]
    return deepcopy(value)


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
    TOOL_COMPLETED = "tool_completed"
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
        EventTypes.TOOL_COMPLETED,
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
