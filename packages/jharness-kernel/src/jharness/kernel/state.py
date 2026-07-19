"""Flat immutable run lifecycle and committed metrics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeAlias, cast, overload

from jharness.kernel._digest import (
    empty_call_id_suffix_digest,
    empty_tool_call_suffix_digest,
    prepend_call_id_digest,
    prepend_tool_call_digest,
)
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


@dataclass(frozen=True, slots=True, init=False, repr=False)
class PendingToolCalls(Sequence[ToolCall]):
    """Immutable pending-call cursor with constant-time suffix advancement."""

    _calls: tuple[ToolCall, ...]
    _offset: int
    _digests: tuple[bytes, ...]
    _call_id_digests: tuple[bytes, ...]

    def __init__(self, calls: Sequence[ToolCall]) -> None:
        normalized = expect_instance_tuple(calls, ToolCall, "pending tool calls")
        if not normalized:
            raise ValueError("pending tool calls require at least one call")
        ids = tuple(call.id for call in normalized)
        if len(ids) != len(set(ids)):
            raise ValueError("pending tool call ids must be unique")
        digests = [b""] * (len(normalized) + 1)
        call_id_digests = [b""] * (len(normalized) + 1)
        digests[-1] = empty_tool_call_suffix_digest()
        call_id_digests[-1] = empty_call_id_suffix_digest()
        for index in range(len(normalized) - 1, -1, -1):
            call = normalized[index]
            digests[index] = prepend_tool_call_digest(digests[index + 1], call)
            call_id_digests[index] = prepend_call_id_digest(call_id_digests[index + 1], call.id)
        object.__setattr__(self, "_calls", normalized)
        object.__setattr__(self, "_offset", 0)
        object.__setattr__(self, "_digests", tuple(digests))
        object.__setattr__(self, "_call_id_digests", tuple(call_id_digests))

    def __len__(self) -> int:
        return len(self._calls) - self._offset

    def __iter__(self) -> Iterator[ToolCall]:
        return (self._calls[index] for index in range(self._offset, len(self._calls)))

    @overload
    def __getitem__(self, index: int) -> ToolCall: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ToolCall, ...]: ...

    def __getitem__(self, index: int | slice) -> ToolCall | tuple[ToolCall, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self._calls[self._offset + start : self._offset + stop]
            return tuple(self)[index]
        index = expect_int(index, "pending tool call index")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("pending tool call index out of range")
        return self._calls[self._offset + index]

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, PendingToolCalls):
            return False
        if self._calls is other._calls and self._offset == other._offset:
            return True
        if len(self) != len(other) or self.digest != other.digest:
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"PendingToolCalls({tuple(self)!r})"

    @property
    def pending_count(self) -> int:
        return len(self)

    @property
    def digest(self) -> bytes:
        """Return the semantic digest of the remaining full calls."""

        return self._digests[self._offset]

    @property
    def call_id_digest(self) -> bytes:
        """Return the composable ordered digest of remaining call ids."""

        return self._call_id_digests[self._offset]

    def prefix(self, count: int) -> tuple[ToolCall, ...]:
        """Materialize at most ``count`` leading calls."""

        count = expect_int(count, "pending tool call prefix count")
        if count < 0:
            raise ValueError("pending tool call prefix count must be >= 0")
        return self._calls[self._offset : self._offset + min(count, len(self))]

    def limit(self, count: int) -> Sequence[ToolCall]:
        """Return an O(1) sequence view bounded to at most ``count`` calls."""

        count = expect_int(count, "pending tool call limit")
        if count < 0:
            raise ValueError("pending tool call limit must be >= 0")
        return _PendingToolCallWindow(self, min(count, len(self)))

    def advance(self, count: int) -> PendingToolCalls | None:
        """Advance by ``count`` calls while sharing all immutable backing data."""

        count = expect_int(count, "pending tool call advance count")
        if count < 0 or count > len(self):
            raise ValueError("pending tool call advance count is out of range")
        if count == 0:
            return self
        offset = self._offset + count
        if offset == len(self._calls):
            return None
        pending = object.__new__(PendingToolCalls)
        object.__setattr__(pending, "_calls", self._calls)
        object.__setattr__(pending, "_offset", offset)
        object.__setattr__(pending, "_digests", self._digests)
        object.__setattr__(pending, "_call_id_digests", self._call_id_digests)
        return pending


@dataclass(frozen=True, slots=True, init=False, repr=False)
class _PendingToolCallWindow(Sequence[ToolCall]):
    _pending: PendingToolCalls
    _count: int

    def __init__(self, pending: PendingToolCalls, count: int) -> None:
        object.__setattr__(self, "_pending", pending)
        object.__setattr__(self, "_count", count)

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[ToolCall]:
        calls = self._pending._calls  # pyright: ignore[reportPrivateUsage]
        offset = self._pending._offset  # pyright: ignore[reportPrivateUsage]
        return (calls[index] for index in range(offset, offset + self._count))

    @overload
    def __getitem__(self, index: int) -> ToolCall: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ToolCall, ...]: ...

    def __getitem__(self, index: int | slice) -> ToolCall | tuple[ToolCall, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step != 1:
                return tuple(self)[index]
            calls = self._pending._calls  # pyright: ignore[reportPrivateUsage]
            offset = self._pending._offset  # pyright: ignore[reportPrivateUsage]
            return calls[offset + start : offset + stop]
        index = expect_int(index, "pending tool call window index")
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("pending tool call window index out of range")
        return self._pending[index]


@dataclass(frozen=True, slots=True)
class ToolsPending:
    """A non-empty model-ordered suffix of tool calls remains."""

    pending: PendingToolCalls

    kind: ClassVar[str] = "tools_pending"

    def __post_init__(self) -> None:
        expect_instance(self.pending, PendingToolCalls, "pending")


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
