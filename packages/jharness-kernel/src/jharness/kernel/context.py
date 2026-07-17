"""Immutable logical run context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from jharness.kernel._validation import (
    expect_non_empty_str,
    expect_number,
    expect_optional_number,
    expect_optional_str,
    freeze_mapping,
)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Stable correlation and deadline context shared by a logical run."""

    run_id: str
    started_at: float
    deadline: float | None = None
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    run_kind: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.run_id, "run_id")
        started_at = expect_number(self.started_at, "started_at")
        deadline = expect_optional_number(self.deadline, "deadline")
        parent_run_id = expect_optional_str(self.parent_run_id, "parent_run_id")
        parent_tool_call_id = expect_optional_str(
            self.parent_tool_call_id,
            "parent_tool_call_id",
        )
        run_kind = expect_optional_str(self.run_kind, "run_kind")
        if started_at < 0:
            raise ValueError("started_at must be >= 0")
        if deadline is not None and deadline < 0:
            raise ValueError("deadline must be >= 0")
        if parent_run_id is None:
            if parent_tool_call_id is not None or run_kind is not None:
                raise ValueError("parent_tool_call_id and run_kind require parent_run_id")
        else:
            expect_non_empty_str(parent_run_id, "parent_run_id")
            if run_kind is None or not run_kind:
                raise ValueError("child run requires non-empty run_kind")
            if parent_tool_call_id == "":
                raise ValueError("parent_tool_call_id must not be empty")
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "run metadata"))
