"""Private AgentLoop support types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from time import monotonic
from typing import Any, TypeAlias

from kernel._trace import TraceRecorder
from kernel.control import ConversationInsert, PauseRequest, RunController
from kernel.errors import InvalidToolCall
from kernel.journal import RunJournal
from kernel.limits import LoopLimits
from kernel.messages import ToolCall
from kernel.scheduler import ToolBatch, ToolCatalog, ToolScheduler, ToolSchedulerProtocol
from kernel.snapshot import RunSnapshot
from kernel.state import AgentState
from kernel.status import AgentStatus
from kernel.store import RunStore
from kernel.tools import RuntimeContextSnapshot, ToolOutput, ToolSpec

ToolSchedulerFactory: TypeAlias = Callable[[ToolCatalog, LoopLimits], ToolSchedulerProtocol]
TracePayload: TypeAlias = Mapping[str, Any]


class RuntimeTimeoutError(Exception):
    """Raised only when the runtime-owned deadline expires."""


class RuntimePauseInterrupt(Exception):
    """Raised when host code interrupts a model call before it can commit."""

    def __init__(self, request: PauseRequest) -> None:
        super().__init__(request.reason)
        self.request = request


class RuntimeConversationInsert(Exception):
    """Raised when external input preempts an in-flight model call."""

    def __init__(self, insert: ConversationInsert) -> None:
        super().__init__(insert.id)
        self.insert = insert


def default_tool_scheduler_factory(tools: ToolCatalog, limits: LoopLimits) -> ToolScheduler:
    return ToolScheduler(
        tools,
        max_parallel_tool_calls=(
            1 if limits.stop_on_tool_error else limits.max_parallel_tool_calls
        ),
    )


@dataclass(slots=True, frozen=True)
class PreparedToolBatch:
    """Tool batch after hook rewriting and approval decisions."""

    batch: ToolBatch
    precomputed_results: Mapping[str, ToolOutput]
    pause_request: PauseRequest | None = None


@dataclass(slots=True, frozen=True)
class ToolProgressRecord:
    """Live progress emitted by a tool implementation."""

    call: ToolCall
    batch_id: str
    index: int
    data: Mapping[str, Any]


class EmptyToolRegistry:
    """No-tool registry used when a run has no host-provided tools."""

    __slots__ = ()

    def specs(self) -> tuple[ToolSpec, ...]:
        return ()

    def spec_for(self, name: str) -> ToolSpec | None:
        _ = name
        return None

    def validate_call(self, call: ToolCall) -> None:
        raise InvalidToolCall(f"unknown tool: {call.name}")

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContextSnapshot,
        *,
        progress_emitter: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> ToolOutput:
        _ = context, progress_emitter, cancel_checker
        self.validate_call(call)
        raise RuntimeError("unreachable")


@dataclass(slots=True, frozen=True)
class AppliedTransition:
    """Runtime state transition that may be notified after a durable commit."""

    previous: AgentStatus
    current: AgentStatus
    state: AgentState
    data: Mapping[str, Any]


@dataclass(slots=True)
class RunControlState:
    """Runtime-owned control state that hooks cannot mutate."""

    run_id: str
    started_at: float
    deadline: float | None = None
    monotonic_deadline: float | None = None
    run_controller: RunController | None = None
    run_store: RunStore | None = None
    run_journal: RunJournal | None = None
    trace: TraceRecorder | None = None
    tool_scheduler: ToolSchedulerProtocol | None = None
    active_tool_call_ids: set[str] = dataclass_field(default_factory=lambda: set[str]())
    initial_snapshot: RunSnapshot | None = None
    last_checkpoint: RunSnapshot | None = None
    last_checkpoint_id: str | None = None
    post_journal_dispatch: dict[int, tuple[bool, bool]] = dataclass_field(
        default_factory=lambda: dict[int, tuple[bool, bool]]()
    )
    post_journal_transitions: dict[int, AppliedTransition] = dataclass_field(
        default_factory=lambda: dict[int, AppliedTransition]()
    )
    sequence: int = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def remaining_seconds(self) -> float | None:
        if self.monotonic_deadline is None:
            return None
        return max(0.0, self.monotonic_deadline - monotonic())
