"""Compact immutable traces built from one invocation's events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite
from typing import Any, Literal, TypeAlias, cast

from jharness.kernel.events import Event, EventKind
from jharness.kernel.json_values import FrozenJsonDict, freeze_json_value

RequestKind: TypeAlias = Literal["start", "continue", "resume"]

_REQUEST_KINDS = frozenset({"start", "continue", "resume"})


def _freeze_data(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("trace entry data must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise TypeError("trace entry data keys must be strings")
    if isinstance(value, FrozenJsonDict):
        return cast(Mapping[str, Any], value)
    string_mapping = cast(Mapping[str, Any], mapping)
    return cast(
        Mapping[str, Any],
        freeze_json_value(
            string_mapping,
            label="trace entry data",
            error_message="trace entry data is immutable",
        ),
    )


def _metadata_keys(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("trace metadata_keys must be a sequence")
    keys = tuple(cast(Sequence[object], value))
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("trace metadata_keys must contain strings")
    result = cast(tuple[str, ...], keys)
    if len(result) != len(set(result)):
        raise ValueError("trace metadata_keys must be unique")
    return result


@dataclass(frozen=True, slots=True)
class TraceHeader:
    """Invocation identity stored once for every trace entry."""

    run_id: str
    invocation_id: str
    request_kind: RequestKind
    metadata_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.run_id), str) or not self.run_id:
            raise ValueError("trace run_id must be a non-empty string")
        if not isinstance(cast(object, self.invocation_id), str) or not self.invocation_id:
            raise ValueError("trace invocation_id must be a non-empty string")
        if self.request_kind not in _REQUEST_KINDS:
            raise ValueError(f"unsupported trace request kind: {self.request_kind}")
        object.__setattr__(self, "metadata_keys", _metadata_keys(self.metadata_keys))


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One event without the trace header's repeated identity."""

    sequence: int
    kind: EventKind
    created_at: float
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        sequence = cast(object, self.sequence)
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise TypeError("trace entry sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("trace entry sequence must be >= 1")
        if not isinstance(cast(object, self.kind), EventKind):
            raise TypeError("trace entry kind must be EventKind")
        created = cast(object, self.created_at)
        if isinstance(created, bool) or not isinstance(created, int | float):
            raise TypeError("trace entry created_at must be a number")
        created_at = float(self.created_at)
        if not isfinite(created_at) or created_at < 0:
            raise ValueError("trace entry created_at must be finite and >= 0")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "data", _freeze_data(self.data))


@dataclass(frozen=True, slots=True)
class RunTrace:
    """One compact diagnostic artifact; it is not recovery state."""

    header: TraceHeader
    entries: tuple[TraceEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.header), TraceHeader):
            raise TypeError("trace header must be TraceHeader")
        raw_entries = cast(object, self.entries)
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, str | bytes):
            raise TypeError("trace entries must be a sequence")
        entries = tuple(cast(Sequence[object], raw_entries))
        if len(entries) < 2:
            raise ValueError("trace requires at least two entries")
        if any(not isinstance(entry, TraceEntry) for entry in entries):
            raise TypeError("trace entries must contain TraceEntry values")
        object.__setattr__(self, "entries", entries)


def build_trace(
    events: Iterable[Event],
    request_kind: RequestKind,
    metadata_keys: Sequence[str] = (),
) -> RunTrace:
    """Compact one complete invocation event sequence without copying payloads."""

    if request_kind not in _REQUEST_KINDS:
        raise ValueError(f"unsupported trace request kind: {request_kind}")
    observed = tuple(events)
    if len(observed) < 2:
        raise ValueError("trace construction requires at least two events")
    if any(not isinstance(cast(object, event), Event) for event in observed):
        raise TypeError("trace construction requires Event values")
    first = observed[0]
    if first.kind is not EventKind.INVOCATION_STARTED:
        raise ValueError("trace events must start with invocation_started")
    if observed[-1].kind is not EventKind.INVOCATION_STOPPED:
        raise ValueError("trace events must end with invocation_stopped")
    if any(event.run_id != first.run_id for event in observed):
        raise ValueError("trace events must share one run_id")
    if any(event.invocation_id != first.invocation_id for event in observed):
        raise ValueError("trace events must share one invocation_id")
    if any(right.sequence <= left.sequence for left, right in pairwise(observed)):
        raise ValueError("trace event sequences must strictly increase")
    if first.data.get("request_kind") != request_kind:
        raise ValueError("trace request_kind must match invocation_started")
    header = TraceHeader(
        run_id=first.run_id,
        invocation_id=first.invocation_id,
        request_kind=request_kind,
        metadata_keys=_metadata_keys(metadata_keys),
    )
    entries = tuple(
        TraceEntry(event.sequence, event.kind, event.created_at, event.data) for event in observed
    )
    return RunTrace(header, entries)


__all__ = [
    "RequestKind",
    "RunTrace",
    "TraceEntry",
    "TraceHeader",
    "build_trace",
]
