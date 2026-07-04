"""Durable checkpoint store protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from agent_runtime._frozen import freeze_value, thaw_value
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import AgentStatus


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


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


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _expect_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    return float(value)


def _freeze_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        freeze_value(_expect_mapping(value, label), error_message=f"{label} is immutable"),
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], thaw_value(value))


def _expect_status(value: object, label: str) -> AgentStatus:
    try:
        return AgentStatus(_expect_str(value, label))
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid agent status") from exc


@dataclass(slots=True, frozen=True)
class CheckpointSummary:
    """Compact checkpoint metadata for host listing."""

    run_id: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    sequence: int
    status: AgentStatus
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        run_id = _expect_str(self.run_id, "checkpoint run_id")
        checkpoint_id = _expect_str(self.checkpoint_id, "checkpoint checkpoint_id")
        if not run_id:
            raise ValueError("checkpoint run_id must not be empty")
        if not checkpoint_id:
            raise ValueError("checkpoint checkpoint_id must not be empty")
        parent = _expect_optional_str(self.parent_checkpoint_id, "checkpoint parent_checkpoint_id")
        if parent == "":
            raise ValueError("checkpoint parent_checkpoint_id must not be empty")
        sequence = _expect_int(self.sequence, "checkpoint sequence")
        if sequence < 0:
            raise ValueError("checkpoint sequence must be >= 0")
        if not isinstance(cast(object, self.status), AgentStatus):
            raise TypeError("checkpoint status must be an AgentStatus")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "parent_checkpoint_id", parent)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "created_at", _expect_number(self.created_at, "created_at"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointSummary:
        _reject_unknown_keys(
            value,
            {
                "run_id",
                "checkpoint_id",
                "parent_checkpoint_id",
                "sequence",
                "status",
                "created_at",
                "metadata",
            },
            "checkpoint summary",
        )
        return cls(
            run_id=_expect_str(value["run_id"], "checkpoint run_id"),
            checkpoint_id=_expect_str(value["checkpoint_id"], "checkpoint checkpoint_id"),
            parent_checkpoint_id=_expect_optional_str(
                value["parent_checkpoint_id"], "checkpoint parent_checkpoint_id"
            ),
            sequence=_expect_int(value["sequence"], "checkpoint sequence"),
            status=_expect_status(value["status"], "checkpoint status"),
            created_at=_expect_number(value["created_at"], "checkpoint created_at"),
            metadata=_expect_mapping(value["metadata"], "checkpoint metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "sequence": self.sequence,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": _copy_mapping(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class StoredCheckpoint:
    """Durable checkpoint payload passed to a RunStore."""

    run_id: str
    checkpoint_id: str
    parent_checkpoint_id: str | None
    sequence: int
    status: AgentStatus
    snapshot: RunSnapshot
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        summary = CheckpointSummary(
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            parent_checkpoint_id=self.parent_checkpoint_id,
            sequence=self.sequence,
            status=self.status,
            created_at=self.created_at,
            metadata=self.metadata,
        )
        if not isinstance(cast(object, self.snapshot), RunSnapshot):
            raise TypeError("stored checkpoint snapshot must be a RunSnapshot")
        snapshot = RunSnapshot.from_dict(self.snapshot.to_dict())
        if summary.run_id != snapshot.context.run_id:
            raise ValueError("stored checkpoint run_id must match snapshot context run_id")
        if summary.sequence != snapshot.context.sequence:
            raise ValueError("stored checkpoint sequence must match snapshot context sequence")
        if summary.status is not snapshot.state.status:
            raise ValueError("stored checkpoint status must match snapshot state status")
        object.__setattr__(self, "run_id", summary.run_id)
        object.__setattr__(self, "checkpoint_id", summary.checkpoint_id)
        object.__setattr__(self, "parent_checkpoint_id", summary.parent_checkpoint_id)
        object.__setattr__(self, "sequence", summary.sequence)
        object.__setattr__(self, "status", summary.status)
        object.__setattr__(self, "created_at", summary.created_at)
        object.__setattr__(self, "metadata", summary.metadata)
        object.__setattr__(self, "snapshot", snapshot)

    def summary(self) -> CheckpointSummary:
        return CheckpointSummary(
            run_id=self.run_id,
            checkpoint_id=self.checkpoint_id,
            parent_checkpoint_id=self.parent_checkpoint_id,
            sequence=self.sequence,
            status=self.status,
            created_at=self.created_at,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data = self.summary().to_dict()
        data["snapshot"] = self.snapshot.to_dict()
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StoredCheckpoint:
        _reject_unknown_keys(
            value,
            {
                "run_id",
                "checkpoint_id",
                "parent_checkpoint_id",
                "sequence",
                "status",
                "snapshot",
                "created_at",
                "metadata",
            },
            "stored checkpoint",
        )
        summary = CheckpointSummary.from_dict(
            {
                "run_id": value["run_id"],
                "checkpoint_id": value["checkpoint_id"],
                "parent_checkpoint_id": value["parent_checkpoint_id"],
                "sequence": value["sequence"],
                "status": value["status"],
                "created_at": value["created_at"],
                "metadata": value["metadata"],
            }
        )
        return cls(
            run_id=summary.run_id,
            checkpoint_id=summary.checkpoint_id,
            parent_checkpoint_id=summary.parent_checkpoint_id,
            sequence=summary.sequence,
            status=summary.status,
            snapshot=RunSnapshot.from_dict(
                _expect_mapping(value["snapshot"], "stored checkpoint snapshot")
            ),
            created_at=summary.created_at,
            metadata=summary.metadata,
        )


class RunStore(Protocol):
    """Host-owned durable checkpoint store."""

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        """Persist a checkpoint before the runtime treats it as durable."""
        ...

    async def load_checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> RunSnapshot:
        """Load one checkpoint, defaulting to the latest sequence when checkpoint_id is null."""
        ...

    async def list_checkpoints(self, run_id: str) -> Sequence[CheckpointSummary]:
        """Return stored checkpoints for a run in host-defined order."""
        ...
