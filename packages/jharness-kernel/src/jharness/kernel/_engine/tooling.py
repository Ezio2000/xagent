"""One validated, approved, bounded, atomic tool batch."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from time import time
from typing import Any, Protocol, TypeAlias, cast

from jharness.kernel._engine.change import Change, failed, limited, suspend
from jharness.kernel._engine.deadline import Deadline, WorkDeadlineReached
from jharness.kernel.approval import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalSuspend,
)
from jharness.kernel.checkpoint import SuspensionView, ToolBatchFact, ToolOutcomeKind
from jharness.kernel.control import CancelTool, ControlInbox, Insert, Pause, drain_pending_controls
from jharness.kernel.errors import ToolError
from jharness.kernel.events import EventKind
from jharness.kernel.limits import LimitReason, RunLimits
from jharness.kernel.messages import ToolCall
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import Planning, Suspended, Suspension, ToolsPending
from jharness.kernel.tools import (
    BatchPolicy,
    SettledResult,
    ToolBatch,
    ToolBinding,
    ToolCatalog,
    ToolContext,
    ToolFailure,
    ToolResult,
    WaitingResult,
    tool_message,
)


class Emit(Protocol):
    def __call__(self, kind: EventKind, data: Mapping[str, Any]) -> Awaitable[None]: ...


@dataclass(frozen=True, slots=True)
class Prepared:
    call: ToolCall
    binding: ToolBinding | None = None
    result: ToolResult | None = None


@dataclass(frozen=True, slots=True)
class ToolStepOutcome:
    change: Change
    deferred: tuple[Insert, ...] = ()


class ToolStep:
    __slots__ = ("_approval", "_catalog", "_emit", "_limits", "_policy")

    def __init__(
        self,
        *,
        catalog: ToolCatalog,
        limits: RunLimits,
        policy: BatchPolicy,
        approval: ApprovalPolicy | None,
        emit: Emit,
    ) -> None:
        self._catalog = catalog
        self._limits = limits
        self._policy = policy
        self._approval = approval
        self._emit = emit

    async def run(
        self,
        snapshot: RunSnapshot,
        pending: ToolsPending,
        *,
        deadline: Deadline,
        inbox: ControlInbox,
    ) -> ToolStepOutcome:
        capacity = self._limits.max_tool_calls - snapshot.metrics.tool_calls
        if capacity <= 0:
            return ToolStepOutcome(limited(LimitReason.MAX_TOOL_CALLS))
        try:
            batch = self._select(pending.pending[:capacity])
            prepared = self._bind(batch)
            approved = await self._approve(prepared, batch, deadline)
        except WorkDeadlineReached:
            return ToolStepOutcome(limited(LimitReason.DEADLINE))
        except Exception as exc:
            return ToolStepOutcome(
                failed("tool_protocol_error", str(exc) or exc.__class__.__name__)
            )
        if isinstance(approved, Pause):
            return ToolStepOutcome(suspend(pending, approved.suspension))

        boundary_pause, deferred = drain_pending_controls(inbox)
        try:
            results, observed_pause, inserts = await execute(
                approved,
                batch,
                snapshot,
                deadline,
                inbox,
                limits=self._limits,
                emit=self._emit,
            )
        except WorkDeadlineReached:
            return ToolStepOutcome(limited(LimitReason.DEADLINE))
        boundary_pause = boundary_pause or observed_pause
        deferred.extend(inserts)
        return _outcome(pending, batch, results, boundary_pause, deferred)

    def _select(self, selectable: tuple[ToolCall, ...]) -> ToolBatch:
        batch = self._policy.select(selectable, self._catalog, self._limits)
        if not isinstance(cast(object, batch), ToolBatch):
            raise ToolError("batch policy returned an invalid batch")
        if len(batch.calls) > self._limits.max_tool_batch_size:
            raise ToolError("batch exceeds max_tool_batch_size")
        if batch.calls != selectable[: len(batch.calls)]:
            raise ToolError("batch policy must select an exact pending prefix")
        if batch.parallel and any(
            (spec := self._catalog.spec(call.name)) is None or not spec.parallel_safe
            for call in batch.calls
        ):
            raise ToolError("parallel batch contains a non-parallel-safe tool")
        return batch

    def _bind(self, batch: ToolBatch) -> tuple[Prepared, ...]:
        prepared: list[Prepared] = []
        for call in batch.calls:
            try:
                binding = self._catalog.bind(call)
                if not isinstance(cast(object, binding), ToolBinding):
                    raise ToolError("catalog returned an invalid binding")
                if binding.call != call or binding.spec != self._catalog.spec(call.name):
                    raise ToolError("binding does not match the frozen catalog")
                prepared.append(Prepared(call, binding=binding))
            except Exception as exc:
                prepared.append(
                    Prepared(
                        call,
                        result=SettledResult(
                            ToolFailure.from_error(
                                "invalid_tool_call", str(exc) or exc.__class__.__name__
                            )
                        ),
                    )
                )
        return tuple(prepared)

    async def _approve(
        self,
        prepared: tuple[Prepared, ...],
        batch: ToolBatch,
        deadline: Deadline,
    ) -> tuple[Prepared, ...] | Pause:
        if self._approval is None:
            return prepared
        requests: list[ApprovalRequest] = []
        for index, item in enumerate(prepared):
            if item.binding is None:
                continue
            request = ApprovalRequest(batch.id, index, item.call, item.binding.spec.risk)
            requests.append(request)
            await self._emit(EventKind.APPROVAL_REQUESTED, approval_request_data(request))
        if not requests:
            return prepared
        decisions = await _await_deadline(self._approval.decide(tuple(requests)), deadline)
        if not isinstance(cast(object, decisions), tuple):
            raise ToolError("approval policy must return a tuple of decisions")
        if len(decisions) != len(requests):
            raise ToolError("approval policy must return one decision per request")
        return await _apply_decisions(prepared, tuple(requests), decisions, emit=self._emit)


async def _apply_decisions(
    prepared: tuple[Prepared, ...],
    requests: tuple[ApprovalRequest, ...],
    decisions: tuple[ApprovalDecision, ...],
    *,
    emit: Emit,
) -> tuple[Prepared, ...] | Pause:
    mutable = list(prepared)
    first_pause: Pause | None = None
    for request, decision in zip(requests, decisions, strict=True):
        if not isinstance(cast(object, decision), ApprovalDecision):
            raise ToolError("approval policy returned an invalid decision")
        if decision.call_id != request.call.id:
            raise ToolError("approval decision call_id mismatch")
        await emit(EventKind.APPROVAL_DECIDED, approval_decision_data(decision))
        if isinstance(decision, ApprovalSuspend) and first_pause is None:
            first_pause = Pause(decision.suspension)
        elif isinstance(decision, ApprovalDeny):
            mutable[request.index] = Prepared(
                request.call,
                result=SettledResult(ToolFailure.from_error("approval_denied", decision.reason)),
            )
    return first_pause or tuple(mutable)


Progress: TypeAlias = tuple[str, Mapping[str, Any]]
ToolTask: TypeAlias = asyncio.Task[tuple[int, ToolResult]]


@dataclass(slots=True)
class _Execution:
    results: list[ToolResult | None]
    progress: asyncio.Queue[Progress]
    overflowed: set[str]
    active: set[str]
    permits: asyncio.Semaphore
    completion_order: deque[int]
    tasks: dict[int, ToolTask]
    progress_task: asyncio.Task[Progress]
    control_task: asyncio.Task[Pause | Insert | CancelTool]
    pause: Pause | None
    inserts: list[Insert]


async def execute(
    prepared: tuple[Prepared, ...],
    batch: ToolBatch,
    snapshot: RunSnapshot,
    deadline: Deadline,
    inbox: ControlInbox,
    *,
    limits: RunLimits,
    emit: Emit,
) -> tuple[tuple[ToolResult, ...], Pause | None, tuple[Insert, ...]]:
    results: list[ToolResult | None] = [item.result for item in prepared]
    actual = [(index, item) for index, item in enumerate(prepared) if item.binding is not None]
    if not actual:
        return _complete(results), None, ()
    state = _start(actual, results, batch, snapshot, inbox, limits, emit)
    try:
        while state.tasks:
            await _advance(state, batch, deadline, inbox, emit)
        await _drain_progress(state.progress, emit)
    finally:
        await _cancel_and_settle({state.progress_task, state.control_task, *state.tasks.values()})
    return _complete(state.results), state.pause, tuple(state.inserts)


def _start(
    actual: list[tuple[int, Prepared]],
    results: list[ToolResult | None],
    batch: ToolBatch,
    snapshot: RunSnapshot,
    inbox: ControlInbox,
    limits: RunLimits,
    emit: Emit,
) -> _Execution:
    progress: asyncio.Queue[Progress] = asyncio.Queue(limits.max_buffered_progress)
    overflowed: set[str] = set()
    active: set[str] = set()
    permits = asyncio.Semaphore(limits.max_tool_concurrency if batch.parallel else 1)
    completion_order: deque[int] = deque()
    tasks: dict[int, ToolTask] = {}
    for index, item in actual:
        tasks[index] = asyncio.create_task(
            _invoke(
                index,
                item,
                batch,
                snapshot,
                inbox,
                progress,
                overflowed,
                active,
                permits,
                completion_order,
                emit,
            )
        )
    return _Execution(
        results,
        progress,
        overflowed,
        active,
        permits,
        completion_order,
        tasks,
        asyncio.create_task(progress.get()),
        asyncio.create_task(inbox.next()),
        None,
        [],
    )


async def _invoke(
    index: int,
    item: Prepared,
    batch: ToolBatch,
    snapshot: RunSnapshot,
    inbox: ControlInbox,
    progress: asyncio.Queue[Progress],
    overflowed: set[str],
    active: set[str],
    permits: asyncio.Semaphore,
    completion_order: deque[int],
    emit: Emit,
) -> tuple[int, ToolResult]:
    binding = cast(ToolBinding, item.binding)
    await permits.acquire()
    completed = False
    try:
        active.add(item.call.id)
        await emit(
            EventKind.TOOL_STARTED,
            {
                "batch_id": batch.id,
                "index": index,
                "call": call_data(item.call),
                "parallel": batch.parallel,
            },
        )
        context = ToolContext(
            snapshot.context,
            _progress(item.call.id, progress, overflowed),
            lambda: inbox.cancellation_requested(item.call.id),
        )
        try:
            result = _tool_result(await binding.invoke(context))
            if item.call.id in overflowed:
                result = SettledResult(
                    ToolFailure.from_error("progress_overflow", "tool progress buffer exceeded")
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result = SettledResult(
                ToolFailure.from_error("tool_error", str(exc) or exc.__class__.__name__)
            )
        finally:
            active.discard(item.call.id)
            inbox.clear_cancellation(item.call.id)
        completed = True
        completion_order.append(index)
        return index, result
    finally:
        if not completed:
            permits.release()


def _progress(
    call_id: str,
    queue: asyncio.Queue[Progress],
    overflowed: set[str],
) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
    async def emit_progress(value: Mapping[str, Any]) -> None:
        try:
            queue.put_nowait((call_id, value))
        except asyncio.QueueFull as exc:
            overflowed.add(call_id)
            raise ToolError("tool progress buffer exceeded") from exc

    return emit_progress


async def _advance(
    state: _Execution,
    batch: ToolBatch,
    deadline: Deadline,
    inbox: ControlInbox,
    emit: Emit,
) -> None:
    remaining = deadline.remaining()
    if remaining is not None and remaining <= 0:
        raise WorkDeadlineReached
    done, _ = await asyncio.wait(
        {*state.tasks.values(), state.progress_task, state.control_task},
        timeout=remaining,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if not done:
        raise WorkDeadlineReached
    if state.progress_task in done:
        call_id, value = state.progress_task.result()
        await emit(EventKind.TOOL_PROGRESS, {"tool_call_id": call_id, "progress": value})
        state.progress_task = asyncio.create_task(state.progress.get())
    if state.control_task in done:
        control = state.control_task.result()
        if isinstance(control, CancelTool):
            if control.call_id in state.active:
                await emit(EventKind.TOOL_CANCEL_REQUESTED, {"tool_call_id": control.call_id})
                if control.call_id in state.active:
                    inbox.request_cancellation(control.call_id)
            else:
                inbox.clear_cancellation(control.call_id)
        elif isinstance(control, Pause) and state.pause is None:
            state.pause = control
        elif isinstance(control, Insert):
            state.inserts.append(control)
        state.control_task = asyncio.create_task(inbox.next())
    await _finish_ready(state, batch, emit)


async def _finish_ready(state: _Execution, batch: ToolBatch, emit: Emit) -> None:
    while state.completion_order:
        index = state.completion_order[0]
        task = state.tasks[index]
        if not task.done():
            break
        state.completion_order.popleft()
        try:
            observed_index, result = task.result()
            if observed_index != index:
                raise ToolError("tool completion index does not match its task")
            await _drain_progress(state.progress, emit)
            state.results[index] = result
            del state.tasks[index]
            await emit(
                EventKind.TOOL_FINISHED,
                {
                    "batch_id": batch.id,
                    "index": index,
                    "tool_call_id": batch.calls[index].id,
                    "outcome_kind": result.outcome.kind,
                },
            )
        finally:
            state.permits.release()
    for task in state.tasks.values():
        if task.done():
            task.result()
            raise ToolError("tool task completed without recording its settlement")


async def _drain_progress(queue: asyncio.Queue[Progress], emit: Emit) -> None:
    while not queue.empty():
        call_id, value = queue.get_nowait()
        await emit(EventKind.TOOL_PROGRESS, {"tool_call_id": call_id, "progress": value})


def _complete(results: list[ToolResult | None]) -> tuple[ToolResult, ...]:
    if any(result is None for result in results):
        raise ToolError("tool batch completed without every result")
    return cast(tuple[ToolResult, ...], tuple(results))


def _tool_result(value: object) -> ToolResult:
    if not isinstance(value, ToolResult):
        raise TypeError("tool binding returned an invalid result")
    return value


async def _cancel_and_settle(tasks: set[asyncio.Task[Any]]) -> None:
    for task in tasks:
        task.cancel()
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=0.1)
    for task in pending:
        task.cancel()
        asyncio.get_running_loop().call_exception_handler(
            {
                "message": "JHarness abandoned a tool task that ignored cancellation",
                "task": task,
            }
        )
        task.add_done_callback(_consume)


def _consume(task: asyncio.Task[Any]) -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.exception()


def _outcome(
    pending: ToolsPending,
    batch: ToolBatch,
    results: tuple[ToolResult, ...],
    pause: Pause | None,
    deferred: list[Insert],
) -> ToolStepOutcome:
    remaining = pending.pending[len(batch.calls) :]
    next_active = ToolsPending(remaining) if remaining else Planning()
    waiting = next((result for result in results if isinstance(result, WaitingResult)), None)
    suspension = (
        waiting.suspension if waiting is not None else (pause.suspension if pause else None)
    )
    state = next_active if suspension is None else Suspended(next_active, suspension)
    view = None if suspension is None else suspension_view(suspension)
    change = Change(
        ToolBatchFact(
            time(),
            batch.id,
            tuple(call.id for call in batch.calls),
            batch.parallel,
            tuple(ToolOutcomeKind(result.outcome.kind) for result in results),
            view,
        ),
        state,
        append=tuple(
            tool_message(call, result) for call, result in zip(batch.calls, results, strict=True)
        ),
        tool_calls=len(batch.calls),
    )
    return ToolStepOutcome(change, tuple(deferred))


def suspension_view(suspension: Suspension) -> SuspensionView:
    return SuspensionView(
        suspension.reason,
        suspension.source,
        suspension.wait_id,
        tuple(sorted(suspension.metadata)),
    )


async def _await_deadline(awaitable: Awaitable[Any], deadline: Deadline) -> Any:
    remaining = deadline.remaining()
    if remaining is not None and remaining <= 0:
        raise WorkDeadlineReached
    try:
        async with asyncio.timeout(remaining):
            return await awaitable
    except TimeoutError as exc:
        raise WorkDeadlineReached from exc


def call_data(call: ToolCall) -> Mapping[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def approval_request_data(request: ApprovalRequest) -> Mapping[str, Any]:
    risk: dict[str, Any] = dict(request.risk.extra)
    for key in (
        "filesystem",
        "network",
        "subprocess",
        "destructive",
        "requires_approval",
    ):
        value = getattr(request.risk, key)
        if value is not None:
            risk[key] = value
    return {
        "batch_id": request.batch_id,
        "index": request.index,
        "call": call_data(request.call),
        "risk": risk,
    }


def approval_decision_data(decision: ApprovalDecision) -> Mapping[str, Any]:
    if isinstance(decision, ApprovalAllow):
        return {"call_id": decision.call_id, "kind": "allow"}
    if isinstance(decision, ApprovalDeny):
        return {"call_id": decision.call_id, "kind": "deny", "reason": decision.reason}
    suspension = decision.suspension
    return {
        "call_id": decision.call_id,
        "kind": "suspend",
        "suspension": {
            "reason": suspension.reason,
            "source": suspension.source,
            "wait_id": suspension.wait_id,
            "metadata": suspension.metadata,
        },
    }
