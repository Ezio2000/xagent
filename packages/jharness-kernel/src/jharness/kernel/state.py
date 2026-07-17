"""Flat immutable run lifecycle and committed metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias, cast

from jharness.kernel._validation import (
    expect_instance,
    expect_instance_tuple,
    expect_int,
    expect_non_empty_str,
    expect_optional_str,
    freeze_mapping,
)
from jharness.kernel.limits import LimitReason
from jharness.kernel.messages import ContentPart, ErrorInfo, ToolCall
from jharness.kernel.models import ModelUsage


@dataclass(frozen=True, slots=True)
class Planning:
    """The next durable step is one model invocation."""

    kind: ClassVar[str] = "planning"


@dataclass(frozen=True, slots=True)
class ToolsPending:
    """A non-empty model-ordered suffix of tool calls remains."""

    pending: tuple[ToolCall, ...]

    kind: ClassVar[str] = "tools_pending"

    def __post_init__(self) -> None:
        pending = expect_instance_tuple(self.pending, ToolCall, "pending")
        if not pending:
            raise ValueError("tools pending state requires at least one call")
        ids = tuple(call.id for call in pending)
        if len(ids) != len(set(ids)):
            raise ValueError("pending tool call ids must be unique")
        object.__setattr__(self, "pending", pending)


ActiveState: TypeAlias = Planning | ToolsPending


@dataclass(frozen=True, slots=True)
class Suspension:
    """Host-visible reason why active execution stopped."""

    reason: str
    source: str
    wait_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_non_empty_str(self.reason, "suspension reason")
        expect_non_empty_str(self.source, "suspension source")
        wait_id = expect_optional_str(self.wait_id, "suspension wait_id")
        if wait_id == "":
            raise ValueError("suspension wait_id must not be empty")
        object.__setattr__(
            self,
            "metadata",
            freeze_mapping(self.metadata, "suspension metadata"),
        )


@dataclass(frozen=True, slots=True)
class Suspended:
    """Execution stopped with its exact next active state preserved."""

    resume_to: ActiveState
    suspension: Suspension

    kind: ClassVar[str] = "suspended"

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.resume_to), ActiveState):
            raise TypeError("resume_to must be Planning or ToolsPending")
        expect_instance(self.suspension, Suspension, "suspension")


@dataclass(frozen=True, slots=True)
class Completed:
    """Successful final output."""

    parts: tuple[ContentPart, ...]

    kind: ClassVar[str] = "completed"

    def __post_init__(self) -> None:
        parts = expect_instance_tuple(self.parts, ContentPart, "completed parts")
        if not parts:
            raise ValueError("completed state requires at least one content part")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class Failed:
    """Unrecoverable model, protocol, or infrastructure failure."""

    error: ErrorInfo

    kind: ClassVar[str] = "failed"

    def __post_init__(self) -> None:
        expect_instance(self.error, ErrorInfo, "failure error")


@dataclass(frozen=True, slots=True)
class Limited:
    """Portable run-budget terminal state."""

    reason: LimitReason

    kind: ClassVar[str] = "limited"

    def __post_init__(self) -> None:
        expect_instance(self.reason, LimitReason, "limit reason")


RunState: TypeAlias = Planning | ToolsPending | Suspended | Completed | Failed | Limited


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Monotonic counters advanced only by a committed checkpoint."""

    planning_steps: int = 0
    tool_calls: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)

    def __post_init__(self) -> None:
        for value, label in (
            (self.planning_steps, "planning_steps"),
            (self.tool_calls, "tool_calls"),
        ):
            if expect_int(value, label) < 0:
                raise ValueError(f"{label} must be >= 0")
        expect_instance(self.usage, ModelUsage, "usage")

    def advance(
        self,
        *,
        planning_steps: int = 0,
        tool_calls: int = 0,
        usage: ModelUsage | None = None,
    ) -> RunMetrics:
        """Return metrics after applying one non-negative durable delta."""

        planning_delta = expect_int(planning_steps, "planning_steps delta")
        tool_delta = expect_int(tool_calls, "tool_calls delta")
        if planning_delta < 0 or tool_delta < 0:
            raise ValueError("metric deltas must be >= 0")
        if usage is not None:
            expect_instance(usage, ModelUsage, "usage delta")
        return RunMetrics(
            planning_steps=self.planning_steps + planning_delta,
            tool_calls=self.tool_calls + tool_delta,
            usage=self.usage.add(usage),
        )
