"""Runtime context shared across model calls, tools, and hooks."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4

from kernel._validation import (
    expect_mapping as _expect_mapping,
)
from kernel._validation import (
    expect_nonnegative_int as _expect_int,
)
from kernel._validation import (
    expect_number as _expect_number,
)
from kernel._validation import (
    expect_optional_number as _expect_optional_number,
)
from kernel._validation import (
    expect_optional_str as _expect_optional_str,
)
from kernel._validation import (
    expect_str as _expect_str,
)
from kernel._validation import (
    reject_unknown_keys as _reject_unknown_keys,
)


def _empty_metadata() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


@dataclass(slots=True)
class RuntimeContext:
    """Per-run context passed through runtime extension points.

    `started_at` and `deadline` are wall-clock epoch seconds so serialized
    contexts can be used as durable checkpoint data. The loop keeps monotonic
    timeout bookkeeping separately.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time)
    deadline: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    run_kind: str | None = None
    _sequence: int = 0

    def __post_init__(self) -> None:
        self.run_id = _expect_str(self.run_id, "runtime context run_id")
        self.started_at = _expect_number(self.started_at, "runtime context started_at")
        self.deadline = _expect_optional_number(self.deadline, "runtime context deadline")
        self.parent_run_id = _expect_optional_str(
            self.parent_run_id, "runtime context parent_run_id"
        )
        self.parent_tool_call_id = _expect_optional_str(
            self.parent_tool_call_id, "runtime context parent_tool_call_id"
        )
        self.run_kind = _expect_optional_str(self.run_kind, "runtime context run_kind")
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.deadline is not None and self.deadline <= self.started_at:
            raise ValueError("deadline must be after started_at")
        if self.parent_run_id == "":
            raise ValueError("parent_run_id must not be empty")
        if self.parent_tool_call_id == "":
            raise ValueError("parent_tool_call_id must not be empty")
        if self.run_kind == "":
            raise ValueError("run_kind must not be empty")
        if self.parent_run_id is None and (
            self.parent_tool_call_id is not None or self.run_kind is not None
        ):
            raise ValueError("parent_run_id is required for child run fields")
        self.metadata = _copy_mapping(self.metadata)
        self._sequence = _expect_int(self._sequence, "runtime context sequence")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RuntimeContext:
        known = {
            "run_id",
            "started_at",
            "deadline",
            "metadata",
            "parent_run_id",
            "parent_tool_call_id",
            "run_kind",
            "sequence",
        }
        _reject_unknown_keys(value, known, "runtime context")
        context = cls(
            run_id=_expect_str(value["run_id"], "runtime context run_id"),
            started_at=_expect_number(value["started_at"], "runtime context started_at"),
            deadline=_expect_optional_number(value["deadline"], "runtime context deadline"),
            metadata=_expect_mapping(value["metadata"], "runtime context metadata"),
            parent_run_id=_expect_optional_str(
                value.get("parent_run_id"), "runtime context parent_run_id"
            ),
            parent_tool_call_id=_expect_optional_str(
                value.get("parent_tool_call_id"), "runtime context parent_tool_call_id"
            ),
            run_kind=_expect_optional_str(value.get("run_kind"), "runtime context run_kind"),
        )
        context._sequence = _expect_int(value["sequence"], "runtime context sequence")
        return context

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    @property
    def sequence(self) -> int:
        return self._sequence

    @sequence.setter
    def sequence(self, value: int) -> None:
        self._sequence = _expect_int(value, "runtime context sequence")

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "metadata": _copy_mapping(self.metadata),
            **({} if self.parent_run_id is None else {"parent_run_id": self.parent_run_id}),
            **(
                {}
                if self.parent_tool_call_id is None
                else {"parent_tool_call_id": self.parent_tool_call_id}
            ),
            **({} if self.run_kind is None else {"run_kind": self.run_kind}),
            "sequence": self._sequence,
        }
