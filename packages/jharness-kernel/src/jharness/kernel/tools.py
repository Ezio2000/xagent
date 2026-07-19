"""Single-invocation tool contracts and bounded batch selection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, cast, runtime_checkable

from jharness.kernel._validation import (
    expect_bool,
    expect_instance,
    expect_instance_tuple,
    expect_non_empty_str,
    expect_optional_str,
    expect_str,
    freeze_mapping,
)
from jharness.kernel.context import RunContext
from jharness.kernel.errors import ToolError
from jharness.kernel.json_values import FrozenJsonValue, freeze_json_value
from jharness.kernel.limits import RunLimits
from jharness.kernel.messages import ContentPart, ErrorInfo, Message, TaskRef, ToolCall

if TYPE_CHECKING:
    from jharness.kernel.state import Suspension


def _freeze_schema(value: object, label: str) -> Mapping[str, Any] | bool:
    if isinstance(value, bool):
        return value
    return freeze_mapping(value, label)


def _parts(value: Sequence[ContentPart], label: str) -> tuple[ContentPart, ...]:
    parts = expect_instance_tuple(value, ContentPart, f"{label} parts")
    if not parts:
        raise ValueError(f"{label} requires at least one content part")
    return parts


def _structured(value: object) -> FrozenJsonValue:
    return freeze_json_value(
        value,
        label="tool structured_content",
        error_message="tool structured_content is immutable",
    )


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """Portable scheduling facts for one logical tool."""

    concurrency: Literal["serial", "parallel"] = "serial"
    read_only: bool = False
    idempotent: bool = False

    def __post_init__(self) -> None:
        concurrency = expect_str(self.concurrency, "tool concurrency")
        read_only = expect_bool(self.read_only, "tool read_only")
        idempotent = expect_bool(self.idempotent, "tool idempotent")
        if concurrency not in {"serial", "parallel"}:
            raise ValueError(f"unsupported tool concurrency: {concurrency}")
        if concurrency == "parallel" and not (read_only and idempotent):
            raise ValueError("parallel tool must be read-only and idempotent")


@dataclass(frozen=True, slots=True)
class ToolRisk:
    """Standard approval facts plus immutable host-specific fields."""

    filesystem: str | None = None
    network: str | None = None
    subprocess: bool | None = None
    destructive: bool | None = None
    requires_approval: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        for value, label in (
            (self.filesystem, "risk filesystem"),
            (self.network, "risk network"),
        ):
            text = expect_optional_str(value, label)
            if text == "":
                raise ValueError(f"{label} must not be empty")
        for value, label in (
            (self.subprocess, "risk subprocess"),
            (self.destructive, "risk destructive"),
            (self.requires_approval, "risk requires_approval"),
        ):
            if value is not None:
                expect_instance(value, bool, label)
        extra = freeze_mapping(self.extra, "risk extra")
        reserved = {
            "filesystem",
            "network",
            "subprocess",
            "destructive",
            "requires_approval",
        }
        if reserved.intersection(extra):
            raise ValueError("risk extra cannot replace standardized fields")
        object.__setattr__(self, "extra", extra)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Immutable model-neutral tool declaration."""

    name: str
    description: str
    input_schema: Mapping[str, Any] | bool
    output_schema: Mapping[str, Any] | bool | None = None
    execution: ToolExecution = field(default_factory=ToolExecution)
    risk: ToolRisk = field(default_factory=ToolRisk)

    def __post_init__(self) -> None:
        expect_non_empty_str(self.name, "tool name")
        expect_str(self.description, "tool description")
        object.__setattr__(self, "input_schema", _freeze_schema(self.input_schema, "input schema"))
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _freeze_schema(self.output_schema, "output schema"),
            )
        expect_instance(self.execution, ToolExecution, "tool execution")
        expect_instance(self.risk, ToolRisk, "tool risk")

    @property
    def parallel_safe(self) -> bool:
        return self.execution.concurrency == "parallel"


@dataclass(frozen=True, slots=True, init=False)
class ToolSuccess:
    """A completed model-visible tool observation."""

    parts: tuple[ContentPart, ...]
    structured_content: FrozenJsonValue = None

    def __init__(
        self,
        parts: Sequence[ContentPart],
        structured_content: object = None,
    ) -> None:
        object.__setattr__(self, "parts", _parts(parts, "tool success"))
        object.__setattr__(self, "structured_content", _structured(structured_content))

    @property
    def kind(self) -> Literal["success"]:
        return "success"


@dataclass(frozen=True, slots=True, init=False)
class ToolFailure:
    """A recoverable model-visible tool error."""

    parts: tuple[ContentPart, ...]
    error: ErrorInfo
    structured_content: FrozenJsonValue = None

    def __init__(
        self,
        parts: Sequence[ContentPart],
        error: ErrorInfo,
        structured_content: object = None,
    ) -> None:
        object.__setattr__(self, "parts", _parts(parts, "tool failure"))
        object.__setattr__(self, "error", expect_instance(error, ErrorInfo, "tool failure error"))
        object.__setattr__(self, "structured_content", _structured(structured_content))

    @property
    def kind(self) -> Literal["failure"]:
        return "failure"

    @classmethod
    def from_error(
        cls,
        code: str,
        message: str,
        *,
        structured_content: object = None,
    ) -> ToolFailure:
        return cls(
            (ContentPart.text_part(message),),
            ErrorInfo(code, message),
            structured_content,
        )


@dataclass(frozen=True, slots=True, init=False)
class ToolAccepted:
    """Acknowledgement of host-owned background work."""

    parts: tuple[ContentPart, ...]
    correlation_id: str
    task: TaskRef | None = None
    structured_content: FrozenJsonValue = None

    def __init__(
        self,
        parts: Sequence[ContentPart],
        correlation_id: str,
        task: TaskRef | None = None,
        structured_content: object = None,
    ) -> None:
        object.__setattr__(self, "parts", _parts(parts, "tool accepted"))
        correlation_id = expect_str(correlation_id, "tool accepted correlation_id")
        if not correlation_id:
            raise ValueError("tool accepted requires correlation_id")
        if task is not None:
            expect_instance(task, TaskRef, "tool accepted task")
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "structured_content", _structured(structured_content))

    @property
    def kind(self) -> Literal["accepted"]:
        return "accepted"


@dataclass(frozen=True, slots=True, init=False)
class ToolWaiting:
    """A model-visible observation that external work is pending."""

    parts: tuple[ContentPart, ...]
    task: TaskRef | None = None
    structured_content: FrozenJsonValue = None

    def __init__(
        self,
        parts: Sequence[ContentPart],
        task: TaskRef | None = None,
        structured_content: object = None,
    ) -> None:
        object.__setattr__(self, "parts", _parts(parts, "tool waiting"))
        if task is not None:
            expect_instance(task, TaskRef, "tool waiting task")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "structured_content", _structured(structured_content))

    @property
    def kind(self) -> Literal["waiting"]:
        return "waiting"


ToolOutcome: TypeAlias = ToolSuccess | ToolFailure | ToolAccepted | ToolWaiting
SettledOutcome: TypeAlias = ToolSuccess | ToolFailure | ToolAccepted


@dataclass(frozen=True, slots=True)
class SettledResult:
    """A completed tool invocation."""

    outcome: SettledOutcome

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.outcome), SettledOutcome):
            raise TypeError("settled result requires a settled ToolOutcome")


@dataclass(frozen=True, slots=True)
class WaitingResult:
    """A waiting outcome paired with host-only suspension data."""

    outcome: ToolWaiting
    suspension: Suspension

    def __post_init__(self) -> None:
        from jharness.kernel.state import Suspension

        expect_instance(self.outcome, ToolWaiting, "waiting result outcome")
        expect_instance(self.suspension, Suspension, "waiting result suspension")


ToolResult: TypeAlias = SettledResult | WaitingResult


def tool_message(call: ToolCall, result: ToolResult) -> Message:
    """Construct the durable model-visible message for one invocation result."""

    expect_instance(call, ToolCall, "tool message call")
    if not isinstance(cast(object, result), ToolResult):
        raise TypeError("tool message result must be a ToolResult")
    return Message.tool(call.id, result.outcome)


ProgressEmitter: TypeAlias = Callable[[Mapping[str, Any]], Awaitable[None]]
CancelChecker: TypeAlias = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Restricted tool-facing view of one active invocation."""

    run: RunContext
    _emit: ProgressEmitter = field(repr=False)
    _cancelled: CancelChecker = field(repr=False)

    def __post_init__(self) -> None:
        expect_instance(self.run, RunContext, "tool context run")
        if not callable(self._emit):
            raise TypeError("tool context emit callback must be callable")
        if not callable(self._cancelled):
            raise TypeError("tool context cancellation callback must be callable")

    async def emit_progress(self, progress: Mapping[str, Any]) -> None:
        await self._emit(freeze_mapping(progress, "tool progress"))

    @property
    def cancel_requested(self) -> bool:
        return self._cancelled()


@runtime_checkable
class ToolBinding(Protocol):
    """Validated immutable call-to-implementation binding."""

    @property
    def call(self) -> ToolCall: ...

    @property
    def spec(self) -> ToolSpec: ...

    async def invoke(self, context: ToolContext) -> ToolResult: ...


@runtime_checkable
class ToolCatalog(Protocol):
    """Immutable invocation-local tool catalog."""

    def specs(self) -> tuple[ToolSpec, ...]: ...

    def spec(self, name: str) -> ToolSpec | None: ...

    def bind(self, call: ToolCall) -> ToolBinding: ...


@runtime_checkable
class ToolCatalogProvider(Protocol):
    """Open one immutable catalog per invocation."""

    async def open_catalog(self) -> ToolCatalog: ...


@dataclass(frozen=True, slots=True)
class ToolBatch:
    """One selected pending prefix and atomic commit unit."""

    id: str
    calls: tuple[ToolCall, ...]
    parallel: bool = False

    def __post_init__(self) -> None:
        expect_non_empty_str(self.id, "tool batch id")
        calls = expect_instance_tuple(self.calls, ToolCall, "tool batch calls")
        parallel = expect_bool(self.parallel, "tool batch parallel")
        if not calls:
            raise ValueError("tool batch requires calls")
        ids = [call.id for call in calls]
        if len(ids) != len(set(ids)):
            raise ValueError("tool batch call ids must be unique")
        if not parallel and len(calls) != 1:
            raise ValueError("serial tool batch must contain exactly one call")
        object.__setattr__(self, "calls", calls)


@runtime_checkable
class BatchPolicy(Protocol):
    """Pure strategy that selects only a pending prefix."""

    def select(
        self,
        pending: Sequence[ToolCall],
        catalog: ToolCatalog,
        limits: RunLimits,
    ) -> ToolBatch: ...


class DefaultBatchPolicy:
    """Group consecutive explicitly parallel-safe calls; otherwise serialize."""

    __slots__ = ()

    def select(
        self,
        pending: Sequence[ToolCall],
        catalog: ToolCatalog,
        limits: RunLimits,
    ) -> ToolBatch:
        if not pending:
            raise ToolError("cannot select an empty pending set")
        first = pending[0]
        first_spec = catalog.spec(first.name)
        batch_id = f"tool-batch-{first.id}"
        if first_spec is None or not first_spec.parallel_safe:
            return ToolBatch(batch_id, (first,))
        selected = [first]
        capacity = min(len(pending), limits.max_tool_batch_size)
        for index in range(1, capacity):
            call = pending[index]
            spec = catalog.spec(call.name)
            if spec is None or not spec.parallel_safe:
                break
            selected.append(call)
        return ToolBatch(batch_id, tuple(selected), parallel=len(selected) > 1)


class EmptyToolCatalog:
    """Kernel-owned immutable empty catalog."""

    __slots__ = ()

    def specs(self) -> tuple[ToolSpec, ...]:
        return ()

    def spec(self, name: str) -> ToolSpec | None:
        return None

    def bind(self, call: ToolCall) -> ToolBinding:
        raise ToolError(f"unknown tool: {call.name}")


class EmptyToolCatalogProvider:
    """Kernel-owned default provider."""

    __slots__ = ()

    async def open_catalog(self) -> ToolCatalog:
        return EmptyToolCatalog()
