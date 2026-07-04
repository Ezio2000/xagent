"""Durable runtime event journal protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from agent_runtime._frozen import freeze_value, thaw_value
from agent_runtime.events import AgentEvent


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer or null")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _expect_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    return float(value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _freeze_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        freeze_value(_expect_mapping(value, label), error_message=f"{label} is immutable"),
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], thaw_value(value))


@dataclass(slots=True, frozen=True)
class JournalRecord:
    """Append-only runtime event journal record."""

    event: AgentEvent
    checkpoint_id: str | None = None
    trace_step_id: int | None = None  # Host-filled correlation; runtime leaves it unset.
    payload_ref: str | None = None
    payload_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.event), AgentEvent):
            raise TypeError("journal record event must be an AgentEvent")
        checkpoint_id = _expect_optional_str(self.checkpoint_id, "journal checkpoint_id")
        trace_step_id = _expect_optional_int(self.trace_step_id, "journal trace_step_id")
        payload_ref = _expect_optional_str(self.payload_ref, "journal payload_ref")
        payload_hash = _expect_optional_str(self.payload_hash, "journal payload_hash")
        if checkpoint_id == "":
            raise ValueError("journal checkpoint_id must not be empty")
        if payload_ref == "":
            raise ValueError("journal payload_ref must not be empty")
        if payload_hash == "":
            raise ValueError("journal payload_hash must not be empty")
        object.__setattr__(self, "event", AgentEvent(**self.event.to_dict()))
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "trace_step_id", trace_step_id)
        object.__setattr__(self, "payload_ref", payload_ref)
        object.__setattr__(self, "payload_hash", payload_hash)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def run_id(self) -> str:
        return self.event.run_id

    @property
    def sequence(self) -> int:
        return self.event.sequence

    @property
    def event_type(self) -> str:
        return self.event.type

    @property
    def created_at(self) -> float:
        return self.event.created_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "event": self.event.to_dict(),
            "checkpoint_id": self.checkpoint_id,
            "trace_step_id": self.trace_step_id,
            "payload_ref": self.payload_ref,
            "payload_hash": self.payload_hash,
            "created_at": self.created_at,
            "metadata": _copy_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JournalRecord:
        _reject_unknown_keys(
            value,
            {
                "run_id",
                "sequence",
                "event_type",
                "event",
                "checkpoint_id",
                "trace_step_id",
                "payload_ref",
                "payload_hash",
                "created_at",
                "metadata",
            },
            "journal record",
        )
        event = AgentEvent(**_expect_mapping(value["event"], "journal record event"))
        record = cls(
            event=event,
            checkpoint_id=_expect_optional_str(
                value["checkpoint_id"], "journal record checkpoint_id"
            ),
            trace_step_id=_expect_optional_int(
                value["trace_step_id"], "journal record trace_step_id"
            ),
            payload_ref=_expect_optional_str(value["payload_ref"], "journal record payload_ref"),
            payload_hash=_expect_optional_str(value["payload_hash"], "journal record payload_hash"),
            metadata=_expect_mapping(value["metadata"], "journal record metadata"),
        )
        if _expect_str(value["run_id"], "journal record run_id") != record.run_id:
            raise ValueError("journal record run_id must match event run_id")
        if _expect_int(value["sequence"], "journal record sequence") != record.sequence:
            raise ValueError("journal record sequence must match event sequence")
        if _expect_str(value["event_type"], "journal record event_type") != record.event_type:
            raise ValueError("journal record event_type must match event type")
        if _expect_number(value["created_at"], "journal record created_at") != record.created_at:
            raise ValueError("journal record created_at must match event created_at")
        return record


class RunJournal(Protocol):
    """Host-owned append-only runtime event journal."""

    async def append(self, record: JournalRecord) -> None:
        """Append one event record."""
        ...

    def read(
        self, run_id: str, *, after_sequence: int | None = None
    ) -> AsyncIterator[JournalRecord]:
        """Read event records for a run."""
        ...
