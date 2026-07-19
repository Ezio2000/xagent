"""Immutable invocation observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from jharness.kernel._validation import (
    expect_instance,
    expect_int,
    expect_non_empty_str,
    expect_number,
    freeze_mapping,
)


class EventKind(StrEnum):
    """Closed portable event vocabulary."""

    INVOCATION_STARTED = "invocation_started"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"
    MODEL_FINISHED = "model_finished"
    TOOL_BATCH_SELECTED = "tool_batch_selected"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DECIDED = "approval_decided"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_FINISHED = "tool_finished"
    TOOL_CANCEL_REQUESTED = "tool_cancel_requested"
    CHECKPOINT_COMMITTED = "checkpoint_committed"
    INVOCATION_STOPPED = "invocation_stopped"


@dataclass(frozen=True, slots=True)
class Event:
    """One read-only live or committed observation from an invocation."""

    run_id: str
    invocation_id: str
    sequence: int
    kind: EventKind
    created_at: float
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.run_id, "event run_id")
        expect_non_empty_str(self.invocation_id, "event invocation_id")
        if expect_int(self.sequence, "event sequence") < 1:
            raise ValueError("event sequence must be >= 1")
        expect_instance(self.kind, EventKind, "event kind")
        if expect_number(self.created_at, "event created_at") < 0:
            raise ValueError("event created_at must be >= 0")
        object.__setattr__(
            self,
            "data",
            freeze_mapping(self.data, "event data"),
        )
