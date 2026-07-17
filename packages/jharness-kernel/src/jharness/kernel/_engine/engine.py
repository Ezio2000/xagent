"""Small state dispatcher around effects, pure reduction, and atomic commit."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import time
from typing import Any, TypeAlias, cast
from uuid import uuid4

from jharness.kernel._engine.change import Change, failed, insert, limited, suspend
from jharness.kernel._engine.commit import Committer, WorkCommitDeadlineReached
from jharness.kernel._engine.deadline import (
    Deadline,
    EffectInterrupted,
    WorkDeadlineReached,
    await_effect,
)
from jharness.kernel._engine.planning import PlanningStep
from jharness.kernel._engine.tooling import ToolStep
from jharness.kernel._engine.verification import fact_data, run_view, verify_change
from jharness.kernel._validation import expect_instance
from jharness.kernel.approval import ApprovalPolicy
from jharness.kernel.checkpoint import (
    Checkpoint,
    HistoryRewriteFact,
    ResumedFact,
    StartedFact,
)
from jharness.kernel.commands import ContinueRequest, RunRequest, StartRequest
from jharness.kernel.context import RunContext
from jharness.kernel.control import (
    ControlInbox,
    ControlSource,
    Insert,
    Pause,
    drain_pending_controls,
)
from jharness.kernel.errors import CommitError
from jharness.kernel.events import EventKind
from jharness.kernel.history import HistoryReducer, HistoryRewrite, validate_history
from jharness.kernel.limits import LimitReason, RunLimits
from jharness.kernel.models import (
    Model,
    ModelCapabilities,
    ModelOptions,
    ResponseFormat,
    ToolChoice,
)
from jharness.kernel.repository import EphemeralRepository, RunRepository
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import (
    Limited,
    Planning,
    RunMetrics,
    Suspended,
    ToolsPending,
)
from jharness.kernel.tools import (
    BatchPolicy,
    DefaultBatchPolicy,
    EmptyToolCatalogProvider,
    ToolCatalog,
    ToolCatalogProvider,
)

Emit: TypeAlias = Callable[[EventKind, Mapping[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model: Model
    tools: ToolCatalogProvider = field(default_factory=EmptyToolCatalogProvider)
    limits: RunLimits = field(default_factory=RunLimits)
    model_options: ModelOptions = field(default_factory=ModelOptions)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    response_format: ResponseFormat | None = None
    approval: ApprovalPolicy | None = None
    history_reducer: HistoryReducer | None = None
    batch_policy: BatchPolicy = field(default_factory=DefaultBatchPolicy)
    repository: RunRepository | None = None
    repository_timeout: float = 5.0


class Engine:
    __slots__ = (
        "_catalog",
        "_committer",
        "_config",
        "_deadline",
        "_deferred",
        "_emit",
        "_inbox",
        "_last",
        "_request",
        "_stream",
    )

    def __init__(self, config: EngineConfig, request: RunRequest, *, stream: bool) -> None:
        self._config = config
        self._request = request
        self._stream = stream
        self._emit: Emit | None = None
        self._inbox: ControlInbox | None = None
        self._deadline = Deadline(None)
        self._committer: Committer | None = None
        self._catalog: ToolCatalog | None = None
        self._last: Checkpoint | None = None
        self._deferred: deque[Insert] = deque()

    async def run(self, emit: Emit, controls: ControlSource) -> Checkpoint:
        self._emit = emit
        loop = asyncio.get_running_loop()
        inbox = controls.attach(loop)
        self._inbox = inbox
        try:
            checkpoint = await self._initialize()
            self._last = checkpoint
            if isinstance(checkpoint.snapshot.state, Planning | ToolsPending):
                checkpoint = await self._prepare(checkpoint)
            await self._stop(checkpoint)
            return checkpoint
        except CommitError as exc:
            last = exc.last_checkpoint
            self._last = last
            await emit(
                EventKind.INVOCATION_STOPPED,
                {
                    "reason": "repository_error",
                    "last_checkpoint_id": None if last is None else last.id,
                },
            )
            raise
        except asyncio.CancelledError:
            last = self._last
            await emit(
                EventKind.INVOCATION_STOPPED,
                {
                    "reason": "consumer_closed",
                    "last_checkpoint_id": None if last is None else last.id,
                },
            )
            if last is None:
                raise
            return last
        finally:
            controls.detach(inbox)

    async def _initialize(self) -> Checkpoint:
        request = self._request
        starting = None if isinstance(request, StartRequest) else request.checkpoint
        repository = self._config.repository or EphemeralRepository(starting)
        self._committer = Committer(repository, timeout_seconds=self._config.repository_timeout)
        context = (
            expect_instance(request.context, RunContext, "start context")
            if isinstance(request, StartRequest)
            else request.checkpoint.snapshot.context
        )
        wall_deadline = context.deadline
        if (
            not isinstance(request, StartRequest)
            and self._config.limits.timeout_seconds is not None
        ):
            invocation_deadline = time() + self._config.limits.timeout_seconds
            wall_deadline = (
                invocation_deadline
                if wall_deadline is None
                else min(wall_deadline, invocation_deadline)
            )
        self._deadline = Deadline.from_wall_time(wall_deadline, asyncio.get_running_loop())
        await self._event(
            EventKind.INVOCATION_STARTED,
            {
                "request_kind": request.kind,
                "starting_checkpoint_id": None if starting is None else starting.id,
                "starting": None if starting is None else run_view(starting.snapshot),
            },
        )
        if isinstance(request, StartRequest):
            snapshot = RunSnapshot(
                0,
                context,
                request.messages,
                RunMetrics(),
                Planning(),
            )
            checkpoint = Checkpoint(
                str(uuid4()),
                snapshot,
                StartedFact(time(), tuple(message.role for message in request.messages)),
            )
            try:
                committed = await self._committer.persist_start(
                    checkpoint,
                    work_timeout_seconds=self._deadline.remaining(),
                )
            except WorkCommitDeadlineReached as exc:
                raise CommitError(
                    "start checkpoint commit exceeded work deadline",
                    last_checkpoint=None,
                ) from exc
            await self._committed(None, committed)
            return committed
        if isinstance(request, ContinueRequest):
            return request.checkpoint
        state = cast(Suspended, request.checkpoint.snapshot.state)
        change = Change(
            ResumedFact(
                time(),
                tuple(message.role for message in request.append_messages),
                tuple(sorted(request.metadata)),
            ),
            state.resume_to,
            append=request.append_messages,
        )
        return await self._commit(request.checkpoint, change)

    async def _prepare(self, checkpoint: Checkpoint) -> Checkpoint:
        try:
            self._catalog = await self._open_catalog()
        except EffectInterrupted as interrupted:
            if isinstance(interrupted.control, Pause):
                return await self._commit(
                    checkpoint,
                    suspend(
                        cast(Planning | ToolsPending, checkpoint.snapshot.state),
                        interrupted.control.suspension,
                    ),
                )
            return await self._commit(
                checkpoint,
                failed(
                    "tool_catalog_error",
                    str(interrupted) or interrupted.__class__.__name__,
                ),
            )
        except WorkDeadlineReached:
            return await self._commit(checkpoint, limited(LimitReason.DEADLINE))
        except Exception as exc:
            return await self._commit(
                checkpoint,
                failed("tool_catalog_error", str(exc) or exc.__class__.__name__),
            )
        try:
            capabilities = expect_instance(
                self._config.model.capabilities,
                ModelCapabilities,
                "model capabilities",
            )
        except Exception as exc:
            return await self._commit(
                checkpoint,
                failed("model_capabilities_error", str(exc) or exc.__class__.__name__),
            )
        return await self._run_active(checkpoint, capabilities)

    async def _run_active(
        self, checkpoint: Checkpoint, capabilities: ModelCapabilities
    ) -> Checkpoint:
        skip_reducer = False
        while isinstance(checkpoint.snapshot.state, Planning | ToolsPending):
            boundary = self._boundary(checkpoint)
            if boundary is not None:
                checkpoint = await self._commit(checkpoint, boundary)
                continue
            state = checkpoint.snapshot.state
            if isinstance(state, Planning):
                checkpoint, skip_reducer = await self._plan(checkpoint, capabilities, skip_reducer)
            else:
                checkpoint = await self._tools(checkpoint, state)
        return checkpoint

    async def _plan(
        self,
        checkpoint: Checkpoint,
        capabilities: ModelCapabilities,
        skip_reducer: bool,
    ) -> tuple[Checkpoint, bool]:
        if self._config.history_reducer is not None and not skip_reducer:
            rewrite = await self._reduce_history(checkpoint)
            if rewrite is not None:
                rewritten = isinstance(rewrite.fact, HistoryRewriteFact)
                return await self._commit(checkpoint, rewrite), rewritten
        if checkpoint.snapshot.metrics.planning_steps >= self._config.limits.max_planning_steps:
            return await self._commit(checkpoint, limited(LimitReason.MAX_PLANNING_STEPS)), False
        step = PlanningStep(
            model=self._config.model,
            capabilities=capabilities,
            catalog=cast(ToolCatalog, self._catalog),
            limits=self._config.limits,
            options=self._config.model_options,
            tool_choice=self._config.tool_choice,
            response_format=self._config.response_format,
            stream=self._stream,
            emit=self._event,
        )
        change = await step.run(
            checkpoint.snapshot,
            deadline=self._deadline,
            inbox=cast(ControlInbox, self._inbox),
        )
        return await self._commit(checkpoint, change), False

    async def _tools(self, checkpoint: Checkpoint, state: ToolsPending) -> Checkpoint:
        step = ToolStep(
            catalog=cast(ToolCatalog, self._catalog),
            limits=self._config.limits,
            policy=self._config.batch_policy,
            approval=self._config.approval,
            emit=self._event,
        )
        outcome = await step.run(
            checkpoint.snapshot,
            state,
            deadline=self._deadline,
            inbox=cast(ControlInbox, self._inbox),
        )
        self._deferred.extend(outcome.deferred)
        return await self._commit(checkpoint, outcome.change)

    async def _open_catalog(self) -> ToolCatalog:
        catalog = await await_effect(
            self._config.tools.open_catalog(),
            deadline=self._deadline,
            inbox=cast(ControlInbox, self._inbox),
            defer_insert=self._deferred.append,
        )
        if not isinstance(cast(object, catalog), ToolCatalog):
            raise TypeError("tool provider returned an invalid catalog")
        return catalog

    def _boundary(self, checkpoint: Checkpoint) -> Change | None:
        if self._deadline.expired():
            return limited(LimitReason.DEADLINE)
        inbox = cast(ControlInbox, self._inbox)
        pause, inserts = drain_pending_controls(inbox)
        self._deferred.extend(inserts)
        state = cast(Planning | ToolsPending, checkpoint.snapshot.state)
        if pause is not None:
            return suspend(state, pause.suspension)
        if isinstance(state, Planning) and self._deferred:
            return insert(self._deferred.popleft())
        return None

    async def _reduce_history(self, checkpoint: Checkpoint) -> Change | None:
        reducer = cast(HistoryReducer, self._config.history_reducer)
        try:
            rewrite = await await_effect(
                reducer.reduce(checkpoint.snapshot),
                deadline=self._deadline,
                inbox=cast(ControlInbox, self._inbox),
            )
        except EffectInterrupted as interrupted:
            if isinstance(interrupted.control, Pause):
                return suspend(Planning(), interrupted.control.suspension)
            if isinstance(interrupted.control, Insert):
                return insert(interrupted.control)
            raise
        except WorkDeadlineReached:
            return limited(LimitReason.DEADLINE)
        except Exception as exc:
            return failed("history_reducer_error", str(exc) or exc.__class__.__name__)
        if rewrite is None:
            return None
        if not isinstance(cast(object, rewrite), HistoryRewrite):
            return failed("history_reducer_error", "history reducer returned an invalid value")
        if len(rewrite.messages) > len(checkpoint.snapshot.history):
            return failed("history_reducer_error", "history rewrite cannot increase message count")
        try:
            validate_history(rewrite.messages, Planning())
        except Exception as exc:
            return failed("history_reducer_error", str(exc) or exc.__class__.__name__)
        return Change(
            HistoryRewriteFact(
                time(),
                len(checkpoint.snapshot.history),
                tuple(message.role for message in rewrite.messages),
                rewrite.reason,
                tuple(sorted(rewrite.metadata)),
            ),
            Planning(),
            replace=rewrite.messages,
        )

    async def _commit(self, previous: Checkpoint, change: Change) -> Checkpoint:
        committer = cast(Committer, self._committer)
        cleanup = _deadline_change(change)
        remaining = self._deadline.remaining()
        if not cleanup and remaining is not None and remaining <= 0:
            change = limited(LimitReason.DEADLINE)
            cleanup = True
        try:
            checkpoint = await committer.apply(
                previous,
                change,
                work_timeout_seconds=None if cleanup else remaining,
            )
        except WorkCommitDeadlineReached:
            checkpoint = await committer.apply(previous, limited(LimitReason.DEADLINE))
        self._last = checkpoint
        await self._committed(previous, checkpoint)
        return checkpoint

    async def _committed(self, previous: Checkpoint | None, checkpoint: Checkpoint) -> None:
        before = None if previous is None else run_view(previous.snapshot)
        fact = fact_data(checkpoint.fact)
        after = run_view(checkpoint.snapshot)
        verify_change(before, fact, after)
        await self._event(
            EventKind.CHECKPOINT_COMMITTED,
            {"checkpoint_id": checkpoint.id, "fact": fact, "after": after},
        )

    async def _stop(self, checkpoint: Checkpoint) -> None:
        reason = "suspended" if isinstance(checkpoint.snapshot.state, Suspended) else "terminal"
        await self._event(
            EventKind.INVOCATION_STOPPED,
            {"reason": reason, "last_checkpoint_id": checkpoint.id},
        )

    async def _event(self, kind: EventKind, data: Mapping[str, Any]) -> None:
        await cast(Emit, self._emit)(kind, data)


def _deadline_change(change: Change) -> bool:
    return isinstance(change.state, Limited) and change.state.reason is LimitReason.DEADLINE
