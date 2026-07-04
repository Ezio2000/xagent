"""Agent loop implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from inspect import isawaitable, iscoroutinefunction
from time import monotonic, time
from typing import Any, TypeAlias, TypeVar, cast

from agent_runtime.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from agent_runtime.control import ConversationInsert, PauseRequest, RunController
from agent_runtime.errors import (
    AgentError,
    InvalidToolCall,
    LimitExceeded,
    ModelProviderError,
    ToolError,
)
from agent_runtime.events import CORE_EVENT_TYPES, AgentEvent, EventEmitter, EventTypes
from agent_runtime.hooks import ModelErrorDecision, RuntimeHook
from agent_runtime.journal import JournalRecord, RunJournal
from agent_runtime.limits import LimitReasons, LoopLimits
from agent_runtime.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_part_without_metadata,
    content_parts_summary,
)
from agent_runtime.models import (
    ModelClient,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelStreamAccumulator,
    ModelUsage,
    ResponseFormat,
    ToolChoice,
    model_capabilities,
    stream_event_to_delta_payload,
)
from agent_runtime.resume import ResumeInput
from agent_runtime.runtime import RuntimeContext
from agent_runtime.scheduler import (
    ToolBatch,
    ToolCatalog,
    ToolCompleted,
    ToolScheduler,
    ToolSchedulerProtocol,
    ToolStarted,
)
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import AgentState, AgentStatus, PauseState
from agent_runtime.store import RunStore, StoredCheckpoint
from agent_runtime.tools import (
    Tool,
    ToolAcceptance,
    ToolInvocation,
    ToolObservation,
    ToolOutput,
    ToolRegistry,
    ToolRejection,
)
from agent_runtime.trace import RunTrace, TraceRecorder

T = TypeVar("T")
ToolSchedulerFactory: TypeAlias = Callable[[ToolCatalog, LoopLimits], ToolSchedulerProtocol]
_MAX_DISPATCHED_EVENTS_PER_RUNTIME_EVENT = 1000
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


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


def _default_tool_scheduler_factory(tools: ToolCatalog, limits: LoopLimits) -> ToolScheduler:
    return ToolScheduler(
        tools,
        max_parallel_tool_calls=(
            1 if limits.stop_on_tool_error else limits.max_parallel_tool_calls
        ),
    )


@dataclass(slots=True, frozen=True)
class AgentResult:
    """Final result returned by AgentLoop.run."""

    status: AgentStatus
    final_parts: tuple[ContentPart, ...]
    messages: tuple[Message, ...]
    iterations: int
    total_tool_calls: int
    total_usage: ModelUsage | None = None
    error: str | None = None
    run_id: str = ""
    snapshot: RunSnapshot | None = None
    trace: RunTrace | None = None


@dataclass(slots=True, frozen=True)
class PreparedToolBatch:
    """Tool batch after hook rewriting and approval decisions."""

    batch: ToolBatch
    precomputed_results: Mapping[str, ToolOutput]
    pause_request: PauseRequest | None = None


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


class AgentLoop:
    """Lightweight state-machine-driven agent loop."""

    __slots__ = (
        "_hooks",
        "_limits",
        "_model",
        "_model_options",
        "_approval_policy",
        "_response_format",
        "_run_journal",
        "_run_store",
        "_tool_scheduler_factory",
        "_tool_choice",
        "_trace_enabled",
        "_tools",
    )

    _hooks: tuple[RuntimeHook, ...]
    _limits: LoopLimits
    _model: ModelClient
    _model_options: ModelOptions
    _approval_policy: ApprovalPolicy | None
    _response_format: ResponseFormat | None
    _run_journal: RunJournal | None
    _run_store: RunStore | None
    _tool_scheduler_factory: ToolSchedulerFactory
    _tool_choice: ToolChoice
    _trace_enabled: bool
    _tools: ToolRegistry

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: Sequence[Tool] | ToolRegistry | None = None,
        limits: LoopLimits | None = None,
        model_options: ModelOptions | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        hooks: Sequence[RuntimeHook] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        run_store: RunStore | None = None,
        run_journal: RunJournal | None = None,
        trace: bool = True,
        tool_scheduler_factory: ToolSchedulerFactory | None = None,
    ) -> None:
        if not isinstance(cast(object, trace), bool):
            raise TypeError("trace must be a boolean")
        self._model = model
        self._tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self._limits = limits or LoopLimits()
        self._limits.validate()
        self._model_options = ModelOptions.from_dict((model_options or ModelOptions()).to_dict())
        self._tool_choice = ToolChoice.from_dict((tool_choice or ToolChoice()).to_dict())
        self._response_format = (
            None if response_format is None else ResponseFormat.from_dict(response_format.to_dict())
        )
        self._hooks = tuple(hooks or ())
        self._approval_policy = approval_policy
        self._run_store = run_store
        self._run_journal = run_journal
        self._trace_enabled = trace
        self._tool_scheduler_factory = tool_scheduler_factory or _default_tool_scheduler_factory

    async def run(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AgentResult:
        state = AgentState(status=AgentStatus.PLANNING, messages=list(messages))
        runtime_context, control = self._prepare_run(context, controller=controller)
        return await self._run_prepared_state(state, runtime_context, control, stream=stream)

    async def run_events(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = AgentState(status=AgentStatus.PLANNING, messages=list(messages))
        runtime_context, control = self._prepare_run(context, controller=controller)
        iterator = self._run_prepared_state_events(
            state, runtime_context, control, stream=stream
        ).__aiter__()
        try:
            async with aclosing(self._pump_events(iterator)) as events:
                async for event in events:
                    yield event
        finally:
            runtime_context.sequence = control.sequence

    async def run_snapshot(
        self,
        resume_input: ResumeInput,
        *,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AgentResult:
        if type(resume_input) is not ResumeInput:
            raise TypeError("run_snapshot requires ResumeInput")
        resume_input = ResumeInput.from_dict(resume_input.to_dict())
        working_state, resume_snapshot = resume_input.apply()
        runtime_context, control = self._prepare_run(resume_snapshot.context, controller=controller)
        self._seed_resume_checkpoint(control, resume_snapshot)
        if control.trace is not None:
            control.trace.record_resume(resume_input, working_state)
        return await self._run_prepared_state(
            working_state, runtime_context, control, stream=stream
        )

    async def run_snapshot_events(
        self,
        resume_input: ResumeInput,
        *,
        stream: bool = False,
        controller: RunController | None = None,
    ) -> AsyncIterator[AgentEvent]:
        if type(resume_input) is not ResumeInput:
            raise TypeError("run_snapshot_events requires ResumeInput")
        resume_input = ResumeInput.from_dict(resume_input.to_dict())
        working_state, resume_snapshot = resume_input.apply()
        runtime_context, control = self._prepare_run(resume_snapshot.context, controller=controller)
        self._seed_resume_checkpoint(control, resume_snapshot)
        if control.trace is not None:
            control.trace.record_resume(resume_input, working_state)
        iterator = self._run_prepared_state_events(
            working_state, runtime_context, control, stream=stream
        ).__aiter__()
        try:
            async with aclosing(self._pump_events(iterator)) as events:
                async for event in events:
                    yield event
        finally:
            runtime_context.sequence = control.sequence

    async def _run_prepared_state(
        self,
        working_state: AgentState,
        runtime_context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AgentResult:
        iterator = self._run_prepared_state_events(
            working_state, runtime_context, control, stream=stream
        ).__aiter__()
        try:
            await self._drain_events(iterator)
        except RuntimeTimeoutError as exc:
            raise LimitExceeded(LimitReasons.TIMEOUT_SECONDS) from exc
        finally:
            runtime_context.sequence = control.sequence
        return self._result(working_state, runtime_context, control)

    async def _run_prepared_state_events(
        self,
        working_state: AgentState,
        runtime_context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        iterator = self._run_state_events(
            working_state, runtime_context, control, stream=stream
        ).__aiter__()
        try:
            async with aclosing(self._pump_events(iterator)) as events:
                async for event in events:
                    dispatch_options = self._post_journal_dispatch_options(control, event)
                    if dispatch_options is not None:
                        await self._append_journal_event(control, event)
                        transition = control.post_journal_transitions.pop(event.sequence, None)
                        if transition is not None:
                            await self._notify_transition(transition, runtime_context, control)
                        trace_before_hooks, mark_trace_durable_before_hooks = dispatch_options
                        dispatched_events = await self._dispatch_events(
                            event,
                            runtime_context,
                            control,
                            trace_before_hooks=trace_before_hooks,
                            mark_trace_durable_before_hooks=mark_trace_durable_before_hooks,
                        )
                        for index, dispatched_event in enumerate(dispatched_events):
                            if index > 0:
                                await self._append_journal_event(control, dispatched_event)
                            yield dispatched_event
                        continue
                    await self._append_journal_event(control, event)
                    yield event
        except RuntimeTimeoutError as exc:
            raise LimitExceeded(LimitReasons.TIMEOUT_SECONDS) from exc
        finally:
            runtime_context.sequence = control.sequence

    async def _run_state_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        terminal_checkpoint_committed = False
        control.initial_snapshot = self._snapshot(state, context, control)
        try:
            for event in await self._events(
                context,
                control,
                EventTypes.RUN_STARTED,
                {"state": state.summary()},
                trace_before_hooks=True,
            ):
                yield event

            drive_iterator = self._drive(state, context, control, stream=stream).__aiter__()
            async with aclosing(self._pump_events(drive_iterator)) as events:
                async for event in events:
                    yield event

            terminal_checkpoint_committed = state.is_terminal
            if state.status is not AgentStatus.PAUSED:
                self._clear_pause_request(control)

            if state.status is AgentStatus.COMPLETED:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.FINAL,
                    {
                        "parts": [part.to_dict() for part in state.final_parts],
                        "summary": content_parts_summary(state.final_parts),
                    },
                ):
                    yield event
            elif state.status is AgentStatus.PAUSED:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.RUN_PAUSED,
                    {"pause": None if state.pause is None else state.pause.to_dict()},
                ):
                    yield event
            else:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": state.error or state.status.value},
                ):
                    yield event

            for event in await self._events(
                context, control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
            ):
                yield event
            self._clear_pause_request(control)
        except RuntimeTimeoutError:
            if terminal_checkpoint_committed:
                self._clear_pause_request(control)
                yield await self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": LimitReasons.TIMEOUT_SECONDS},
                )
                yield await self._raw_event(
                    control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
                )
                return
            durable = control.last_checkpoint or control.initial_snapshot
            self._restore_state_from_snapshot(state, durable)
            self._rollback_trace_to_durable(control)
            if state.is_terminal:
                self._clear_pause_request(control)
                yield await self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": LimitReasons.TIMEOUT_SECONDS},
                )
                yield await self._raw_event(
                    control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
                )
                return
            previous = state.status
            state.status = AgentStatus.LIMIT_EXCEEDED
            state.error = LimitReasons.TIMEOUT_SECONDS
            state.pause = None
            self._clear_pause_request(control)
            yield await self._raw_event(
                control,
                EventTypes.STATE_CHANGED,
                {
                    "from": previous.value,
                    "to": state.status.value,
                    "iterations": state.iterations,
                    "total_tool_calls": state.total_tool_calls,
                    "total_usage": None
                    if state.total_usage is None
                    else state.total_usage.to_dict(),
                    "error": state.error,
                    "pause": None,
                },
            )
            yield await self._raw_checkpoint_event(state, context, control)
            yield await self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield await self._raw_event(
                control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            if terminal_checkpoint_committed:
                self._clear_pause_request(control)
                yield await self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": str(exc) or exc.__class__.__name__},
                )
                yield await self._raw_event(
                    control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
                )
                return
            durable = control.last_checkpoint or control.initial_snapshot
            self._restore_state_from_snapshot(state, durable)
            self._rollback_trace_to_durable(control)
            if state.is_terminal:
                self._clear_pause_request(control)
                yield await self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": str(exc) or exc.__class__.__name__},
                )
                yield await self._raw_event(
                    control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
                )
                return
            previous = state.status
            state.status = AgentStatus.FAILED
            state.error = str(exc) or exc.__class__.__name__
            state.pause = None
            self._clear_pause_request(control)
            yield await self._raw_event(
                control,
                EventTypes.STATE_CHANGED,
                {
                    "from": previous.value,
                    "to": state.status.value,
                    "iterations": state.iterations,
                    "total_tool_calls": state.total_tool_calls,
                    "total_usage": None
                    if state.total_usage is None
                    else state.total_usage.to_dict(),
                    "error": state.error,
                    "pause": None,
                },
            )
            yield await self._raw_checkpoint_event(state, context, control)
            yield await self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield await self._raw_event(
                control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
            )

    async def _drive(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        while not state.is_terminal:
            reason = self._active_limit_reason(state, control)
            if reason is not None:
                for event in await self._limit(state, context, control, reason):
                    yield event
                return

            insert_events = await self._insert_if_requested(state, context, control)
            if insert_events:
                for event in insert_events:
                    yield event
                continue

            pause_events = await self._pause_if_requested(state, context, control)
            if pause_events:
                for event in pause_events:
                    yield event
                return

            if state.status is AgentStatus.PLANNING:
                iterator = self._planning_step(state, context, control, stream=stream).__aiter__()
                async with aclosing(self._pump_events(iterator)) as events:
                    async for event in events:
                        yield event
                continue

            if state.status is AgentStatus.EXECUTING_TOOLS:
                iterator = self._tool_step(state, context, control).__aiter__()
                async with aclosing(self._pump_events(iterator)) as events:
                    async for event in events:
                        yield event
                continue

            for event in await self._transition(
                state,
                AgentStatus.FAILED,
                context,
                control,
                error=f"unsupported state: {state.status.value}",
            ):
                yield event
            return

    async def _planning_step(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        reason = self._active_limit_reason(state, control)
        if reason is not None:
            for event in await self._limit(state, context, control, reason):
                yield event
            return

        state.iterations += 1
        model_error_attempts = 0
        while True:
            using_stream = stream and self._model_supports_streaming()
            request = ModelRequest(
                messages=tuple(state.messages),
                tools=self._tools.specs(),
                options=self._model_options,
                tool_choice=self._tool_choice,
                response_format=self._response_format,
            )
            try:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.MODEL_STARTED,
                    {"iteration": state.iterations},
                ):
                    yield event

                request = await self._before_model(request, context, control)
                if using_stream:
                    response_holder: list[ModelResponse] = []
                    iterator = self._stream_model(
                        request, context, control, response_holder
                    ).__aiter__()
                    async with aclosing(self._pump_events(iterator)) as events:
                        async for event in events:
                            yield event
                    response = response_holder[0]
                else:
                    response = await self._await_model_with_interrupt(
                        self._model.complete(request, context),
                        control,
                    )
                response = await self._after_model(
                    ModelResponse.from_dict(response.to_dict()), context, control
                )
                assistant_message = response.to_assistant_message()
                break
            except RuntimePauseInterrupt as exc:
                state.iterations -= 1
                for event in await self._pause(
                    state,
                    context,
                    control,
                    exc.request,
                    resume_status=AgentStatus.PLANNING,
                    origin="control",
                ):
                    yield event
                return
            except RuntimeConversationInsert as exc:
                state.iterations -= 1
                for event in await self._apply_conversation_insert(
                    state, context, control, exc.insert
                ):
                    yield event
                return
            except RuntimeTimeoutError:
                for event in await self._limit(
                    state, context, control, LimitReasons.TIMEOUT_SECONDS
                ):
                    yield event
                return
            except ModelProviderError as exc:
                decision = await self._on_model_error(exc, request, context, control)
                retry = (
                    decision.retry
                    and not using_stream
                    and model_error_attempts < self._limits.max_model_retries
                )
                for event in await self._events(
                    context,
                    control,
                    EventTypes.MODEL_ERROR,
                    {"error": exc.info.to_dict(), "retry": retry},
                ):
                    yield event
                if retry:
                    model_error_attempts += 1
                    continue
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error=decision.message or exc.info.message,
                ):
                    yield event
                return
            except AgentError as exc:
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error=str(exc) or exc.__class__.__name__,
                ):
                    yield event
                return
            except Exception as exc:
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error=str(exc) or exc.__class__.__name__,
                ):
                    yield event
                return

        state.messages.append(assistant_message)
        self._record_model_usage(state, response.usage)
        limit_reason = None
        transition_data: Mapping[str, Any] | None = None
        transition: AppliedTransition | None = None
        if assistant_message.tool_calls:
            state.pending_tool_calls = list(assistant_message.tool_calls)
            limit_reason = self._limits.tool_call_reason(state) or self._limits.usage_reason(
                state.total_usage
            )
            if limit_reason is None:
                transition = self._apply_transition_state(
                    state,
                    AgentStatus.EXECUTING_TOOLS,
                    context,
                    control,
                )
                transition_data = transition.data
        else:
            limit_reason = self._limits.usage_reason(state.total_usage)
            if limit_reason is None:
                state.final_parts = [content_part_without_metadata(part) for part in response.parts]
                transition = self._apply_transition_state(
                    state,
                    AgentStatus.COMPLETED,
                    context,
                    control,
                )
                transition_data = transition.data

        for event in await self._events(
            context,
            control,
            EventTypes.MODEL_COMPLETED,
            response.summary(),
        ):
            yield event
        if limit_reason is not None:
            for event in await self._limit(state, context, control, limit_reason):
                yield event
            return
        if transition_data is None:
            raise RuntimeError("model response did not produce a state transition")
        transition_event: AgentEvent | None = None
        transition_to_notify: AppliedTransition | None = transition
        if self._defer_events_until_checkpoint_saved(control):
            transition_event = self._runtime_event(
                control, EventTypes.STATE_CHANGED, transition_data
            )
            transition_events: tuple[AgentEvent, ...] = ()
        else:
            if transition_to_notify is not None:
                await self._notify_transition(transition_to_notify, context, control)
                transition_to_notify = None
            transition_events = await self._events(
                context, control, EventTypes.STATE_CHANGED, transition_data
            )
        limit_reason = self._active_limit_reason(state, control)
        if limit_reason is not None:
            limit_events = await self._limit(state, context, control, limit_reason)
            if transition_event is not None:
                transition_events = await self._dispatch_deferred_transition_event(
                    transition_to_notify, transition_event, context, control
                )
            for event in self._merge_deferred_events(transition_events, limit_events):
                yield event
            return
        pause_events = await self._pause_if_requested(state, context, control)
        if pause_events:
            if transition_event is not None:
                transition_events = await self._dispatch_deferred_transition_event(
                    transition_to_notify, transition_event, context, control
                )
            for event in self._merge_deferred_events(transition_events, pause_events):
                yield event
            return
        if transition_event is not None:
            events = await self._checkpoint_events_after_deferred_events(
                transition_event, state, context, control, transition=transition_to_notify
            )
        else:
            checkpoint_events = await self._checkpoint_events(state, context, control)
            events = (*transition_events, *checkpoint_events)
        for event in events:
            yield event

    async def _stream_model(
        self,
        request: ModelRequest,
        context: RuntimeContext,
        control: RunControlState,
        response_holder: list[ModelResponse],
    ) -> AsyncIterator[AgentEvent]:
        stream_method = cast(Any, self._model).stream
        iterator = stream_method(request, context).__aiter__()
        accumulator = ModelStreamAccumulator()
        completed_response: ModelResponse | None = None

        try:
            while True:
                try:
                    stream_event = await self._anext_model_with_interrupt(iterator, control)
                except StopAsyncIteration:
                    break
                response = accumulator.apply(stream_event)
                if response is not None:
                    completed_response = response
                payload = stream_event_to_delta_payload(stream_event)
                if payload is None:
                    continue
                for event in await self._events(
                    context,
                    control,
                    EventTypes.MODEL_DELTA,
                    payload,
                ):
                    yield event
        finally:
            await self._close_async_iterator(iterator)

        response_holder.append(completed_response or accumulator.response())

    async def _tool_step(
        self, state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AsyncIterator[AgentEvent]:
        while state.pending_tool_calls:
            reason = self._active_limit_reason(state, control)
            if reason is not None:
                for event in await self._limit(state, context, control, reason):
                    yield event
                return

            remaining_slots = self._remaining_tool_call_slots(state)
            pending_calls = tuple(state.pending_tool_calls[:remaining_slots])
            pending_snapshots = self._tool_call_snapshots(pending_calls)
            batch = self._tool_scheduler(control).next_batch(
                self._tool_calls_from_snapshots(pending_snapshots)
            )
            if batch is None:
                for event in await self._limit(
                    state, context, control, LimitReasons.MAX_TOTAL_TOOL_CALLS
                ):
                    yield event
                return
            if not self._is_prefix_tool_batch(batch, pending_snapshots):
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error="tool scheduler must return a non-empty prefix batch",
                ):
                    yield event
                return
            batch = ToolBatch(
                id=batch.id,
                calls=self._tool_calls_from_snapshots(pending_snapshots[: len(batch.calls)]),
                parallel=batch.parallel,
            )

            prepared_holder: list[PreparedToolBatch | None] = []
            try:
                prepare_iterator = self._prepare_tool_batch_events(
                    batch, state, context, control, prepared_holder
                ).__aiter__()
                async with aclosing(self._pump_events(prepare_iterator)) as events:
                    async for event in events:
                        yield event
            except RuntimeTimeoutError:
                for event in await self._limit(
                    state, context, control, LimitReasons.TIMEOUT_SECONDS
                ):
                    yield event
                return
            prepared = prepared_holder[0] if prepared_holder else None
            if prepared is None:
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error="before_tool cannot change tool call id, name, or mode",
                ):
                    yield event
                return
            if prepared.pause_request is not None:
                for event in await self._pause(
                    state,
                    context,
                    control,
                    prepared.pause_request,
                    resume_status=AgentStatus.EXECUTING_TOOLS,
                    origin="control",
                ):
                    yield event
                return

            try:
                iterator = self._run_tool_batch(prepared, state, context, control).__aiter__()
                async with aclosing(self._pump_events(iterator)) as events:
                    async for event in events:
                        yield event
            except RuntimeTimeoutError:
                for event in await self._limit(
                    state, context, control, LimitReasons.TIMEOUT_SECONDS
                ):
                    yield event
                return

            if state.is_terminal:
                return

            reason = self._active_limit_reason(state, control)
            if reason is not None:
                for event in await self._limit(state, context, control, reason):
                    yield event
                return

            pause_events = await self._pause_if_requested(state, context, control)
            if pause_events:
                for event in pause_events:
                    yield event
                return

        if state.status is not AgentStatus.PLANNING:
            for event in await self._transition(state, AgentStatus.PLANNING, context, control):
                yield event

    async def _prepare_tool_batch_events(
        self,
        batch: ToolBatch,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        prepared_holder: list[PreparedToolBatch | None],
    ) -> AsyncIterator[AgentEvent]:
        prepared: list[ToolCall] = []
        precomputed_results: dict[str, ToolOutput] = {}
        raw_calls = batch.calls[:1] if self._approval_policy is not None else batch.calls
        prepared_parallel = batch.parallel and len(raw_calls) > 1
        for index, raw_call in enumerate(raw_calls):
            raw_snapshot = raw_call.to_dict()
            raw_tool_call = ToolCall.from_dict(raw_snapshot)
            call = await self._before_tool(raw_tool_call, context, control)
            if (
                call.id != raw_snapshot["id"]
                or call.name != raw_snapshot["name"]
                or call.mode != raw_snapshot["mode"]
            ):
                prepared_holder.append(None)
                return
            self._replace_tool_call_in_history(state, cast(str, raw_snapshot["id"]), call)
            state.pending_tool_calls[index] = call
            validation_error = self._tool_validation_error_output(call)
            if validation_error is not None:
                precomputed_results[call.id] = validation_error
                prepared.append(call)
                continue
            decision: ApprovalDecision | None = None
            if self._approval_policy is not None:
                request, request_events = await self._approval_request(call, context, control)
                for event in request_events:
                    yield event
                decision = await self._approval_decision(request, control)
                completed_events = await self._approval_completed_events(
                    call, decision, context, control
                )
                for event in completed_events:
                    yield event
                if decision.action == "pause":
                    prepared_holder.append(
                        PreparedToolBatch(
                            ToolBatch(batch.id, (call,), parallel=False),
                            precomputed_results,
                            pause_request=PauseRequest(
                                reason=decision.reason,
                                source="approval",
                                wait_id=call.id,
                                metadata=decision.metadata,
                            ),
                        )
                    )
                    return
            if decision is not None and decision.action == "deny":
                precomputed_results[call.id] = self._approval_denial_output(call, decision)
            prepared.append(call)
        prepared_holder.append(
            PreparedToolBatch(
                ToolBatch(batch.id, tuple(prepared), prepared_parallel),
                precomputed_results,
            )
        )

    async def _run_tool_batch(
        self,
        prepared: PreparedToolBatch,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> AsyncIterator[AgentEvent]:
        batch = prepared.batch
        precomputed_results = dict(prepared.precomputed_results)
        completed: dict[int, tuple[ToolCall, ToolOutput, Message]] = {}
        started_indices: set[int] = set()
        executing_indices: set[int] = set()
        completed_indices: set[int] = set()
        execute_results: dict[int, dict[str, Any]] = {}
        scheduler_error: AgentError | None = None
        batch_snapshots = self._tool_call_snapshots(batch.calls)
        scheduler_batch = ToolBatch(
            id=batch.id,
            calls=self._tool_calls_from_snapshots(batch_snapshots),
            parallel=batch.parallel,
        )

        def reject_scheduler(message: str) -> None:
            nonlocal scheduler_error
            error = AgentError(message)
            scheduler_error = error
            raise error

        async def execute(call: ToolCall) -> ToolOutput:
            call_index = self._batch_call_index(batch_snapshots, call)
            if call_index not in started_indices:
                reject_scheduler("tool scheduler must start a batch call before executing it")
            if call_index in executing_indices or call_index in execute_results:
                reject_scheduler("tool scheduler must execute each batch call at most once")
            executing_indices.add(call_index)
            try:
                canonical_call = ToolCall.from_dict(batch_snapshots[call_index])
                precomputed = precomputed_results.get(canonical_call.id)
                if precomputed is not None:
                    result = self._normalize_tool_output(precomputed)
                else:
                    result = await self._await_with_timeout(
                        self._execute_tool(canonical_call, context), control
                    )
                    result = self._normalize_tool_output(result)
                result_snapshot = self._tool_output_snapshot(result)
                execute_results[call_index] = result_snapshot
                return self._tool_output_from_snapshot(result_snapshot)
            finally:
                executing_indices.discard(call_index)

        async for progress in self._tool_scheduler(control).run_batch(
            scheduler_batch, execute, stop_on_error=self._limits.stop_on_tool_error
        ):
            if scheduler_error is not None:
                raise scheduler_error
            self._validate_scheduler_progress(batch, batch_snapshots, progress)
            canonical_call = ToolCall.from_dict(batch_snapshots[progress.index])
            if isinstance(progress, ToolStarted):
                if progress.index in started_indices:
                    raise AgentError("tool scheduler emitted duplicate tool_started")
                if progress.index in completed_indices:
                    raise AgentError("tool scheduler started a completed tool call")
                started_indices.add(progress.index)
                implementation_invoked = canonical_call.id not in precomputed_results
                for event in await self._events(
                    context,
                    control,
                    EventTypes.TOOL_STARTED,
                    {
                        "id": canonical_call.id,
                        "name": canonical_call.name,
                        "mode": canonical_call.mode,
                        "arguments": dict(canonical_call.arguments),
                        "batch_id": batch.id,
                        "parallel": batch.parallel,
                        "index": progress.index,
                        "implementation_invoked": implementation_invoked,
                    },
                ):
                    yield event
                continue

            if progress.index not in started_indices:
                raise AgentError("tool scheduler completed a tool call before starting it")
            if progress.index in completed_indices:
                raise AgentError("tool scheduler emitted duplicate tool_completed")
            expected_result = execute_results.get(progress.index)
            if expected_result is None:
                raise AgentError("tool scheduler must complete results produced by execute")
            progress_result_snapshot = self._tool_output_snapshot(progress.result)
            if progress_result_snapshot != expected_result:
                raise AgentError("tool scheduler must not replace execute results")
            completed_indices.add(progress.index)
            implementation_invoked = canonical_call.id not in precomputed_results
            result = await self._after_tool(
                self._tool_output_from_snapshot(expected_result), context, control
            )
            self._validate_tool_output_mode(canonical_call, result)
            if not implementation_invoked:
                self._validate_non_invoked_tool_output(canonical_call, result)
            invocation = ToolInvocation.from_tool_call(canonical_call)
            tool_message = result.to_message(invocation)
            completed[progress.index] = (canonical_call, result, tool_message)
            for event in await self._events(
                context,
                control,
                EventTypes.TOOL_COMPLETED,
                {
                    "id": canonical_call.id,
                    "name": canonical_call.name,
                    "mode": canonical_call.mode,
                    "batch_id": batch.id,
                    "parallel": batch.parallel,
                    "index": progress.index,
                    "implementation_invoked": implementation_invoked,
                    "result": result.summary(),
                },
            ):
                yield event

            if len(completed) < len(batch.calls):
                continue

            committed_results: list[ToolOutput] = []
            for index in range(len(batch.calls)):
                _call, committed_result, message = completed[index]
                committed_results.append(committed_result)
                state.total_tool_calls += 1
                state.messages.append(message)
                state.pending_tool_calls.pop(0)

            completed.clear()

            if self._limits.stop_on_tool_error:
                error_result = next(
                    (result for result in committed_results if result.is_error),
                    None,
                )
                if error_result is not None:
                    state.pending_tool_calls = []
                    for event in await self._transition(
                        state,
                        AgentStatus.FAILED,
                        context,
                        control,
                        error=error_result.text_content or "tool_error",
                    ):
                        yield event
                    return

            limit_reason = self._post_tool_commit_limit_reason(state, control)
            if limit_reason is not None:
                for event in await self._limit(state, context, control, limit_reason):
                    yield event
                return

            pause_result = next(
                (result for result in committed_results if result.pause is not None),
                None,
            )
            if pause_result is not None and pause_result.pause is not None:
                resume_status = (
                    AgentStatus.EXECUTING_TOOLS
                    if state.pending_tool_calls
                    else AgentStatus.PLANNING
                )
                for event in await self._pause(
                    state,
                    context,
                    control,
                    pause_result.pause,
                    resume_status=resume_status,
                    origin="tool_result",
                ):
                    yield event
                return

            pause_events = await self._pause_after_tool_commit_if_requested(state, context, control)
            if pause_events:
                for event in pause_events:
                    yield event
                return

            if not state.pending_tool_calls:
                transition = self._apply_transition_state(
                    state,
                    AgentStatus.PLANNING,
                    context,
                    control,
                )
                transition_data = transition.data
                transition_event: AgentEvent | None = None
                transition_to_notify: AppliedTransition | None = transition
                if self._defer_events_until_checkpoint_saved(control):
                    transition_event = self._runtime_event(
                        control, EventTypes.STATE_CHANGED, transition_data
                    )
                    transition_events: tuple[AgentEvent, ...] = ()
                else:
                    await self._notify_transition(transition_to_notify, context, control)
                    transition_to_notify = None
                    transition_events = await self._events(
                        context, control, EventTypes.STATE_CHANGED, transition_data
                    )
                limit_reason = self._active_limit_reason(state, control)
                if limit_reason is not None:
                    limit_events = await self._limit(state, context, control, limit_reason)
                    if transition_event is not None:
                        transition_events = await self._dispatch_deferred_transition_event(
                            transition_to_notify, transition_event, context, control
                        )
                    for event in self._merge_deferred_events(transition_events, limit_events):
                        yield event
                    return
                pause_events = await self._pause_if_requested(state, context, control)
                if pause_events:
                    if transition_event is not None:
                        transition_events = await self._dispatch_deferred_transition_event(
                            transition_to_notify, transition_event, context, control
                        )
                    for event in self._merge_deferred_events(transition_events, pause_events):
                        yield event
                    return
                if transition_event is not None:
                    events = await self._checkpoint_events_after_deferred_events(
                        transition_event,
                        state,
                        context,
                        control,
                        transition=transition_to_notify,
                    )
                else:
                    checkpoint_events = await self._checkpoint_events(state, context, control)
                    events = (*transition_events, *checkpoint_events)
                for event in events:
                    yield event
                return

            for event in await self._checkpoint_events(state, context, control):
                yield event

        if scheduler_error is not None:
            raise scheduler_error
        if completed_indices != set(range(len(batch.calls))):
            raise AgentError("tool scheduler ended before completing the selected batch")

    async def _approval_request(
        self, call: ToolCall, context: RuntimeContext, control: RunControlState
    ) -> tuple[ApprovalRequest, tuple[AgentEvent, ...]]:
        policy = self._approval_policy
        if policy is None:
            raise RuntimeError("approval request requires an approval policy")
        spec = self._tools.spec_for(call.name)
        risk = {} if spec is None else dict(spec.annotations)
        request = ApprovalRequest(
            tool_call=call,
            tool_spec=spec,
            context=context,
            risk=risk,
        )
        events = await self._events(
            context,
            control,
            EventTypes.APPROVAL_REQUESTED,
            {
                "id": call.id,
                "name": call.name,
                "mode": call.mode,
                "risk": risk,
                "metadata": request.metadata,
            },
        )
        return request, events

    async def _approval_decision(
        self, request: ApprovalRequest, control: RunControlState
    ) -> ApprovalDecision:
        policy = self._approval_policy
        if policy is None:
            raise RuntimeError("approval decision requires an approval policy")
        decision = await self._await_with_timeout(policy.decide(request), control)
        return ApprovalDecision.from_dict(decision.to_dict())

    async def _approval_completed_events(
        self,
        call: ToolCall,
        decision: ApprovalDecision,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        return await self._events(
            context,
            control,
            EventTypes.APPROVAL_COMPLETED,
            {
                "id": call.id,
                "name": call.name,
                "mode": call.mode,
                "action": decision.action,
                "reason": decision.reason,
                "metadata": decision.metadata,
            },
        )

    @staticmethod
    def _approval_denial_output(call: ToolCall, decision: ApprovalDecision) -> ToolOutput:
        text = f"tool call denied by approval policy: {decision.reason}"
        metadata = {"approval": "denied", **dict(decision.metadata)}
        if call.mode == "execute":
            return ToolObservation(
                parts=[ContentPart.text_part(text)],
                metadata=metadata,
                is_error=True,
            )
        if call.mode == "accept":
            return ToolRejection.text(text, metadata=metadata)
        return ToolOutput(
            kind="tool_error",
            parts=[ContentPart.text_part(text)],
            metadata=metadata,
            is_error=True,
        )

    def _tool_validation_error_output(self, call: ToolCall) -> ToolOutput | None:
        try:
            self._tools.validate_call(call)
        except InvalidToolCall as exc:
            return self._tool_error_output(call, exc)
        return None

    async def _execute_tool(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        try:
            tool_context = RuntimeContext.from_dict(context.to_dict())
            return await self._tools.invoke(call, tool_context)
        except (InvalidToolCall, ToolError) as exc:
            return self._tool_error_output(call, exc)

    @staticmethod
    def _tool_error_output(call: ToolCall, exc: Exception) -> ToolOutput:
        text = str(exc) or exc.__class__.__name__
        metadata = {"error_type": exc.__class__.__name__}
        if call.mode == "execute":
            return ToolObservation(
                parts=[ContentPart.text_part(text)],
                metadata=metadata,
                is_error=True,
            )
        if call.mode == "accept":
            return ToolRejection.text(text, metadata=metadata)
        return ToolOutput(
            kind="tool_error",
            parts=[ContentPart.text_part(text)],
            metadata=metadata,
            is_error=True,
        )

    def _remaining_tool_call_slots(self, state: AgentState) -> int:
        return max(0, self._limits.max_total_tool_calls - state.total_tool_calls)

    async def _await_with_timeout(self, awaitable: Awaitable[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(awaitable)
        if remaining is None:
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise
            return await task

        if remaining <= 0:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError

        try:
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        if not done:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError
        return await task

    async def _await_model_with_interrupt(
        self, awaitable: Awaitable[T], control: RunControlState
    ) -> T:
        controller = control.run_controller
        if controller is None:
            return await self._await_with_timeout(awaitable, control)

        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(awaitable)
        pending_insert = controller.pop_insert()
        if pending_insert is not None:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeConversationInsert(pending_insert)
        interrupt_task = asyncio.ensure_future(controller.wait_for_interrupt_or_insert())
        try:
            if remaining is None:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
                )
            elif remaining <= 0:
                raise RuntimeTimeoutError
            else:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise RuntimeTimeoutError

            if interrupt_task in done:
                request = await interrupt_task
                if isinstance(request, ConversationInsert):
                    task.add_done_callback(self._consume_background_task_exception)
                    task.cancel()
                    raise RuntimeConversationInsert(request)
                if task in done:
                    return await task
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise RuntimePauseInterrupt(request)

            if task in done:
                interrupt_task.cancel()
                return await task

            raise RuntimeError("model wait returned without a completed task")
        except RuntimeTimeoutError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        except asyncio.CancelledError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        finally:
            if not interrupt_task.done():
                interrupt_task.cancel()

    async def _anext_with_timeout(self, iterator: AsyncIterator[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(anext(iterator))
        if remaining is None:
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                await self._close_async_iterator(iterator)
                raise
            return await task

        if remaining <= 0:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError

        try:
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        if not done:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise RuntimeTimeoutError
        return await task

    async def _anext_model_with_interrupt(
        self, iterator: AsyncIterator[T], control: RunControlState
    ) -> T:
        controller = control.run_controller
        if controller is None:
            return await self._anext_with_timeout(iterator, control)

        remaining = control.remaining_seconds()
        pending_insert = controller.pop_insert()
        if pending_insert is not None:
            await self._close_async_iterator(iterator)
            raise RuntimeConversationInsert(pending_insert)
        pending_request = controller.pause_request
        if pending_request is not None and pending_request.interrupt:
            await self._close_async_iterator(iterator)
            raise RuntimePauseInterrupt(pending_request)
        task = asyncio.ensure_future(anext(iterator))
        interrupt_task = asyncio.ensure_future(controller.wait_for_interrupt_or_insert())
        try:
            if remaining is None:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
                )
            elif remaining <= 0:
                raise RuntimeTimeoutError
            else:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise RuntimeTimeoutError

            if interrupt_task in done:
                request = await interrupt_task
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                await self._close_async_iterator(iterator)
                if isinstance(request, ConversationInsert):
                    raise RuntimeConversationInsert(request)
                raise RuntimePauseInterrupt(request)

            if task in done:
                interrupt_task.cancel()
                return await task

            raise RuntimeError("model stream wait returned without a completed task")
        except RuntimeTimeoutError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        except asyncio.CancelledError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        finally:
            if not interrupt_task.done():
                interrupt_task.cancel()

    async def _pump_events(
        self, iterator: AsyncIterator[AgentEvent]
    ) -> AsyncGenerator[AgentEvent, None]:
        try:
            while True:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            await self._close_async_iterator(iterator)

    async def _drain_events(self, iterator: AsyncIterator[AgentEvent]) -> None:
        async with aclosing(self._pump_events(iterator)) as events:
            async for _event in events:
                pass

    async def _close_async_iterator(self, iterator: AsyncIterator[object]) -> None:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                close_result = aclose()
                if isawaitable(close_result):
                    await cast(Awaitable[object], close_result)

    def _prepare_run(
        self,
        context: RuntimeContext | None = None,
        *,
        controller: RunController | None = None,
    ) -> tuple[RuntimeContext, RunControlState]:
        now_monotonic = monotonic()
        now_wall = time()
        if context is None:
            deadline = None
            if self._limits.timeout_seconds is not None:
                deadline = now_wall + self._limits.timeout_seconds
            runtime_context = RuntimeContext(started_at=now_wall, deadline=deadline)
        else:
            runtime_context = RuntimeContext.from_dict(context.to_dict())
            if self._limits.timeout_seconds is not None:
                limit_deadline = now_wall + self._limits.timeout_seconds
                if runtime_context.deadline is None:
                    runtime_context.deadline = limit_deadline
                else:
                    runtime_context.deadline = min(runtime_context.deadline, limit_deadline)

        remaining = None
        if runtime_context.deadline is not None:
            remaining = max(0.0, runtime_context.deadline - now_wall)
        tool_scheduler = self._tool_scheduler_factory(
            ToolCatalog(self._tools.specs()), self._limits
        )
        if not isinstance(cast(object, tool_scheduler), ToolSchedulerProtocol):
            raise TypeError("tool_scheduler_factory must return ToolSchedulerProtocol")
        control = RunControlState(
            run_id=runtime_context.run_id,
            started_at=runtime_context.started_at,
            deadline=runtime_context.deadline,
            monotonic_deadline=None if remaining is None else now_monotonic + remaining,
            run_controller=controller,
            run_store=self._run_store,
            run_journal=self._run_journal,
            trace=TraceRecorder(runtime_context.run_id) if self._trace_enabled else None,
            tool_scheduler=tool_scheduler,
            sequence=runtime_context.sequence,
        )
        return runtime_context, control

    @staticmethod
    def _tool_scheduler(control: RunControlState) -> ToolSchedulerProtocol:
        if control.tool_scheduler is None:
            raise RuntimeError("run control is missing a tool scheduler")
        return control.tool_scheduler

    @staticmethod
    def _tool_call_snapshots(calls: tuple[ToolCall, ...]) -> tuple[dict[str, Any], ...]:
        return tuple(ToolCall.from_dict(call.to_dict()).to_dict() for call in calls)

    @staticmethod
    def _tool_calls_from_snapshots(
        snapshots: tuple[Mapping[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        return tuple(ToolCall.from_dict(snapshot) for snapshot in snapshots)

    @staticmethod
    def _tool_output_snapshot(output: ToolOutput) -> dict[str, Any]:
        return AgentLoop._normalize_tool_output(output).to_dict()

    @staticmethod
    def _tool_output_from_snapshot(snapshot: Mapping[str, Any]) -> ToolOutput:
        kind = snapshot.get("kind")
        if kind == "observation":
            return ToolObservation.from_dict(snapshot)
        if kind == "acceptance":
            return ToolAcceptance.from_dict(snapshot)
        if kind == "rejection":
            return ToolRejection.from_dict(snapshot)
        return ToolOutput.from_dict(snapshot)

    @staticmethod
    def _is_prefix_tool_batch(batch: ToolBatch, snapshots: tuple[Mapping[str, Any], ...]) -> bool:
        if not batch.calls or len(batch.calls) > len(snapshots):
            return False
        expected = snapshots[: len(batch.calls)]
        return all(
            actual.to_dict() == dict(pending)
            for actual, pending in zip(batch.calls, expected, strict=True)
        )

    @staticmethod
    def _batch_call_index(snapshots: tuple[Mapping[str, Any], ...], call: ToolCall) -> int:
        call_data = call.to_dict()
        for index, snapshot in enumerate(snapshots):
            if call_data == dict(snapshot):
                return index
        raise AgentError("tool scheduler attempted to execute a call outside the selected batch")

    @staticmethod
    def _validate_scheduler_progress(
        batch: ToolBatch,
        snapshots: tuple[Mapping[str, Any], ...],
        progress: ToolStarted | ToolCompleted,
    ) -> None:
        if progress.batch.id != batch.id or progress.batch.parallel != batch.parallel:
            raise AgentError("tool scheduler progress batch does not match selected batch")
        if len(progress.batch.calls) != len(snapshots):
            raise AgentError("tool scheduler progress batch calls do not match selected batch")
        if any(
            actual.to_dict() != dict(expected)
            for actual, expected in zip(progress.batch.calls, snapshots, strict=True)
        ):
            raise AgentError("tool scheduler progress batch calls do not match selected batch")
        if progress.index < 0 or progress.index >= len(snapshots):
            raise AgentError("tool scheduler progress index is outside the selected batch")
        if progress.call.to_dict() != dict(snapshots[progress.index]):
            raise AgentError("tool scheduler progress call does not match batch index")

    def _seed_resume_checkpoint(self, control: RunControlState, snapshot: RunSnapshot) -> None:
        control.last_checkpoint_id = self._checkpoint_id(snapshot.context.sequence)

    def _model_supports_streaming(self) -> bool:
        return model_capabilities(self._model).streaming

    @staticmethod
    def _record_model_usage(state: AgentState, usage: ModelUsage | None) -> None:
        existing = state.total_usage
        if usage is None:
            return
        values: dict[str, int] = {}
        for usage_field in _USAGE_FIELDS:
            current = None if existing is None else getattr(existing, usage_field)
            increment = getattr(usage, usage_field)
            if existing is None:
                if increment is not None:
                    values[usage_field] = increment
            elif current is not None and increment is not None:
                values[usage_field] = current + increment
            elif current is not None:
                values[usage_field] = current
            elif increment is not None:
                values[usage_field] = increment
        if not values:
            return
        state.total_usage = ModelUsage(
            input_tokens=values.get("input_tokens"),
            output_tokens=values.get("output_tokens"),
            total_tokens=values.get("total_tokens"),
            reasoning_tokens=values.get("reasoning_tokens"),
            cache_read_tokens=values.get("cache_read_tokens"),
            cache_write_tokens=values.get("cache_write_tokens"),
        )

    def _timeout_reason(self, control: RunControlState) -> str | None:
        remaining = control.remaining_seconds()
        if remaining is not None and remaining <= 0:
            return LimitReasons.TIMEOUT_SECONDS
        return None

    def _active_limit_reason(self, state: AgentState, control: RunControlState) -> str | None:
        if state.is_terminal:
            return None
        reason = self._timeout_reason(control)
        if reason is not None:
            return reason
        if state.status is AgentStatus.PLANNING:
            return self._limits.usage_reason(state.total_usage) or self._limits.iteration_reason(
                state
            )
        if state.status is AgentStatus.EXECUTING_TOOLS:
            return self._limits.usage_reason(state.total_usage) or self._limits.tool_call_reason(
                state
            )
        return None

    def _post_tool_commit_limit_reason(
        self, state: AgentState, control: RunControlState
    ) -> str | None:
        reason = self._active_limit_reason(state, control)
        if reason is not None:
            return reason
        if state.status is AgentStatus.EXECUTING_TOOLS and not state.pending_tool_calls:
            return self._limits.iteration_reason(state)
        return None

    async def _limit(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        reason: str,
    ) -> tuple[AgentEvent, ...]:
        return await self._transition(
            state, AgentStatus.LIMIT_EXCEEDED, context, control, error=reason
        )

    async def _insert_if_requested(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        controller = control.run_controller
        if controller is None or state.status is not AgentStatus.PLANNING or state.is_terminal:
            return ()
        insert = controller.pop_insert()
        if insert is None:
            return ()
        return await self._apply_conversation_insert(state, context, control, insert)

    async def _apply_conversation_insert(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        insert: ConversationInsert,
    ) -> tuple[AgentEvent, ...]:
        if state.is_terminal:
            return ()
        if state.status is not AgentStatus.PLANNING:
            controller = control.run_controller
            if controller is not None:
                controller.insert(insert)
            return ()
        message = insert.to_message()
        state.messages.append(message)
        event_data = {
            "insert": insert.to_dict(),
            "message": message.to_dict(),
        }
        if self._defer_events_until_checkpoint_saved(control):
            insert_event = self._runtime_event(
                control,
                EventTypes.CONVERSATION_INSERTED,
                event_data,
            )
            return await self._checkpoint_events_after_deferred_events(
                insert_event, state, context, control
            )
        events = list(
            await self._events(context, control, EventTypes.CONVERSATION_INSERTED, event_data)
        )
        events.extend(await self._checkpoint_events(state, context, control))
        return tuple(events)

    async def _pause_if_requested(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        controller = control.run_controller
        if controller is None or state.is_terminal:
            return ()
        request = controller.pause_request
        if request is None:
            return ()
        return await self._pause(
            state,
            context,
            control,
            request,
            resume_status=state.status,
            origin="control",
        )

    async def _pause_after_tool_commit_if_requested(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        controller = control.run_controller
        if controller is None or state.is_terminal:
            return ()
        request = controller.pause_request
        if request is None:
            return ()
        resume_status = (
            AgentStatus.EXECUTING_TOOLS if state.pending_tool_calls else AgentStatus.PLANNING
        )
        return await self._pause(
            state,
            context,
            control,
            request,
            resume_status=resume_status,
            origin="control",
        )

    async def _pause(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        request: PauseRequest,
        *,
        resume_status: AgentStatus,
        origin: str,
    ) -> tuple[AgentEvent, ...]:
        if state.is_terminal:
            return ()
        reason = self._active_limit_reason(state, control)
        if reason is not None:
            return await self._limit(state, context, control, reason)
        pause = PauseState(
            reason=request.reason,
            resume_status=resume_status,
            source=request.source,
            wait_id=request.wait_id,
            metadata=request.metadata,
        )
        event_data = {
            "request": request.to_dict(),
            "resume_status": resume_status.value,
            "origin": origin,
        }
        if self._defer_events_until_checkpoint_saved(control):
            pause_event = self._runtime_event(
                control,
                EventTypes.PAUSE_REQUESTED,
                event_data,
            )
            transition = self._apply_transition_state(
                state,
                AgentStatus.PAUSED,
                context,
                control,
                pause=pause,
            )
            transition_data = transition.data
            transition_event = self._runtime_event(
                control,
                EventTypes.STATE_CHANGED,
                transition_data,
            )
            events = list(
                await self._checkpoint_events_after_deferred_events(
                    (pause_event, transition_event),
                    state,
                    context,
                    control,
                    transition=transition,
                )
            )
        else:
            events = list(
                await self._events(
                    context,
                    control,
                    EventTypes.PAUSE_REQUESTED,
                    event_data,
                )
            )
            events.extend(
                await self._transition(
                    state,
                    AgentStatus.PAUSED,
                    context,
                    control,
                    pause=pause,
                )
            )
        controller = control.run_controller
        if controller is not None and controller.pause_request == request:
            controller.clear_pause()
        return tuple(events)

    @staticmethod
    def _clear_pause_request(control: RunControlState) -> None:
        controller = control.run_controller
        if controller is not None and controller.pause_request is not None:
            controller.clear_pause()

    @staticmethod
    def _rollback_trace_to_durable(control: RunControlState) -> None:
        if control.trace is not None:
            control.trace.rollback_to_durable()

    async def _transition(
        self,
        state: AgentState,
        status: AgentStatus,
        context: RuntimeContext,
        control: RunControlState,
        *,
        error: str | None = None,
        pause: PauseState | None = None,
        emit_checkpoint: bool = True,
    ) -> tuple[AgentEvent, ...]:
        transition = self._apply_transition_state(
            state, status, context, control, error=error, pause=pause
        )
        transition_data = transition.data
        if emit_checkpoint and self._defer_events_until_checkpoint_saved(control):
            transition_event = self._runtime_event(
                control,
                EventTypes.STATE_CHANGED,
                transition_data,
            )
            return await self._checkpoint_events_after_deferred_events(
                transition_event, state, context, control, transition=transition
            )
        await self._notify_transition(transition, context, control)
        events = list(
            await self._events(
                context,
                control,
                EventTypes.STATE_CHANGED,
                transition_data,
            )
        )
        if emit_checkpoint:
            events.extend(await self._checkpoint_events(state, context, control))
        return tuple(events)

    async def _apply_transition(
        self,
        state: AgentState,
        status: AgentStatus,
        context: RuntimeContext,
        control: RunControlState,
        *,
        error: str | None = None,
        pause: PauseState | None = None,
    ) -> dict[str, Any]:
        transition = self._apply_transition_state(
            state, status, context, control, error=error, pause=pause
        )
        await self._notify_transition(transition, context, control)
        return dict(transition.data)

    def _apply_transition_state(
        self,
        state: AgentState,
        status: AgentStatus,
        context: RuntimeContext,
        control: RunControlState,
        *,
        error: str | None = None,
        pause: PauseState | None = None,
    ) -> AppliedTransition:
        _ = context, control
        previous = state.status
        state.status = status
        state.error = None if status is AgentStatus.PAUSED else error
        state.pause = pause if status is AgentStatus.PAUSED else None
        transition_state = AgentState.from_dict(state.to_dict())
        data = {
            "from": previous.value,
            "to": status.value,
            "iterations": state.iterations,
            "total_tool_calls": state.total_tool_calls,
            "total_usage": None if state.total_usage is None else state.total_usage.to_dict(),
            "error": state.error,
            "pause": None if state.pause is None else state.pause.to_dict(),
        }
        return AppliedTransition(previous, status, transition_state, data)

    async def _notify_transition(
        self,
        transition: AppliedTransition,
        context: RuntimeContext,
        control: RunControlState,
    ) -> None:
        await self._notify_hooks(
            "on_transition",
            transition.previous,
            transition.current,
            transition.state,
            context,
            control=control,
        )

    async def _checkpoint_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        defer_dispatch: bool = False,
    ) -> tuple[AgentEvent, ...]:
        sequence = control.next_sequence()
        snapshot = self._snapshot(state, context, control, sequence=sequence)
        checkpoint_id = self._checkpoint_id(sequence)
        event = AgentEvent(
            EventTypes.CHECKPOINT,
            snapshot.to_dict(),
            run_id=control.run_id,
            sequence=sequence,
        )
        await self._save_checkpoint(
            control,
            checkpoint_id=checkpoint_id,
            snapshot=snapshot,
            parent_checkpoint_id=control.last_checkpoint_id,
        )
        control.last_checkpoint = RunSnapshot.from_dict(event.data)
        control.last_checkpoint_id = checkpoint_id
        if defer_dispatch or control.run_journal is not None:
            if control.run_journal is not None:
                control.post_journal_dispatch[event.sequence] = (True, True)
            return (event,)
        events = await self._dispatch_checkpoint_events((event,), context, control)
        return events

    async def _checkpoint_events_after_deferred_events(
        self,
        deferred_event: AgentEvent | tuple[AgentEvent, ...],
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        transition: AppliedTransition | None = None,
    ) -> tuple[AgentEvent, ...]:
        deferred_events = (
            (deferred_event,) if isinstance(deferred_event, AgentEvent) else deferred_event
        )
        checkpoint_events = await self._checkpoint_events(
            state, context, control, defer_dispatch=True
        )
        if control.run_journal is not None:
            for event in deferred_events:
                control.post_journal_dispatch[event.sequence] = (False, False)
                if transition is not None and event.type == EventTypes.STATE_CHANGED:
                    control.post_journal_transitions[event.sequence] = transition
                    transition = None
            return (*deferred_events, *checkpoint_events)

        dispatched_deferred_events: list[AgentEvent] = []
        for event in deferred_events:
            if transition is not None and event.type == EventTypes.STATE_CHANGED:
                await self._notify_transition(transition, context, control)
                transition = None
            dispatched_deferred_events.extend(await self._dispatch_events(event, context, control))
        checkpoint_events = await self._dispatch_checkpoint_events(
            checkpoint_events, context, control
        )
        return (*dispatched_deferred_events, *checkpoint_events)

    async def _dispatch_deferred_transition_event(
        self,
        transition: AppliedTransition | None,
        event: AgentEvent,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        if control.run_journal is not None:
            control.post_journal_dispatch[event.sequence] = (False, False)
            if transition is not None:
                control.post_journal_transitions[event.sequence] = transition
            return (event,)
        if transition is not None:
            await self._notify_transition(transition, context, control)
        return await self._dispatch_events(event, context, control)

    async def _dispatch_checkpoint_events(
        self,
        checkpoint_events: tuple[AgentEvent, ...],
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        events: list[AgentEvent] = []
        for event in checkpoint_events:
            events.extend(
                await self._dispatch_events(
                    event,
                    context,
                    control,
                    trace_before_hooks=True,
                    mark_trace_durable_before_hooks=True,
                )
            )
        return tuple(events)

    async def _events(
        self,
        context: RuntimeContext,
        control: RunControlState,
        event_type: str,
        data: Mapping[str, Any],
        *,
        trace_before_hooks: bool = False,
    ) -> tuple[AgentEvent, ...]:
        event = self._runtime_event(control, event_type, data)
        return await self._dispatch_events(
            event, context, control, trace_before_hooks=trace_before_hooks
        )

    @staticmethod
    def _runtime_event(
        control: RunControlState,
        event_type: str,
        data: Mapping[str, Any],
    ) -> AgentEvent:
        return AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )

    @staticmethod
    def _defer_events_until_checkpoint_saved(control: RunControlState) -> bool:
        return control.run_store is not None

    @staticmethod
    def _merge_deferred_events(
        deferred_events: tuple[AgentEvent, ...],
        committed_events: tuple[AgentEvent, ...],
    ) -> tuple[AgentEvent, ...]:
        if not deferred_events:
            return committed_events
        return (deferred_events[0], *committed_events, *deferred_events[1:])

    @staticmethod
    def _post_journal_dispatch_options(
        control: RunControlState,
        event: AgentEvent,
    ) -> tuple[bool, bool] | None:
        return control.post_journal_dispatch.pop(event.sequence, None)

    async def _dispatch_events(
        self,
        first_event: AgentEvent,
        context: RuntimeContext,
        control: RunControlState,
        *,
        trace_before_hooks: bool = False,
        mark_trace_durable_before_hooks: bool = False,
    ) -> tuple[AgentEvent, ...]:
        specs: list[tuple[AgentEvent, bool]] = [(first_event, True)]
        events: list[AgentEvent] = []
        pending_trace_events: list[AgentEvent] = []
        dispatched_count = 0
        while specs:
            dispatched_count += 1
            if dispatched_count > _MAX_DISPATCHED_EVENTS_PER_RUNTIME_EVENT:
                raise RuntimeError("custom event dispatch limit exceeded")
            event, runtime_owned = specs.pop(0)
            emitter = EventEmitter()
            runtime_event = event
            recorded_before_hooks = False
            if runtime_owned and trace_before_hooks and control.trace is not None:
                control.trace.record_event(runtime_event)
                if mark_trace_durable_before_hooks:
                    control.trace.mark_durable()
                recorded_before_hooks = True
            for hook in self._hooks:
                replacement = await self._call_hook(
                    hook.on_event, event, context, emitter, control=control
                )
                if replacement is not None:
                    if runtime_owned:
                        raise ValueError("core runtime events cannot be replaced")
                    replacement_event = cast(AgentEvent, replacement)
                    if replacement_event.type in CORE_EVENT_TYPES:
                        raise ValueError(
                            f"core event type is runtime-owned: {replacement_event.type}"
                        )
                    event = AgentEvent(
                        replacement_event.type,
                        replacement_event.data,
                        run_id=event.run_id,
                        sequence=event.sequence,
                        created_at=event.created_at,
                        schema_version=event.schema_version,
                    )
            events.append(event)
            if runtime_owned and not recorded_before_hooks and control.trace is not None:
                pending_trace_events.append(runtime_event)
            specs.extend(
                (
                    AgentEvent(
                        queued.type,
                        queued.data,
                        run_id=control.run_id,
                        sequence=control.next_sequence(),
                    ),
                    False,
                )
                for queued in emitter.drain()
            )
        if control.trace is not None:
            for pending in pending_trace_events:
                control.trace.record_event(pending)
        return tuple(events)

    async def _append_journal_event(
        self,
        control: RunControlState,
        event: AgentEvent,
    ) -> None:
        journal = control.run_journal
        if journal is None:
            return
        checkpoint_id = None
        if event.type == EventTypes.CHECKPOINT:
            checkpoint_id = self._checkpoint_id(event.sequence)
        await self._await_with_timeout(
            journal.append(JournalRecord(event=event, checkpoint_id=checkpoint_id)),
            control,
        )

    async def _save_checkpoint(
        self,
        control: RunControlState,
        *,
        checkpoint_id: str,
        snapshot: RunSnapshot,
        parent_checkpoint_id: str | None,
    ) -> None:
        store = control.run_store
        if store is None:
            return
        await self._await_with_timeout(
            store.save_checkpoint(
                StoredCheckpoint(
                    run_id=control.run_id,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    sequence=snapshot.context.sequence,
                    status=snapshot.state.status,
                    snapshot=snapshot,
                    created_at=time(),
                )
            ),
            control,
        )

    @staticmethod
    def _checkpoint_id(sequence: int) -> str:
        return f"checkpoint-{sequence}"

    async def _raw_event(
        self, control: RunControlState, event_type: str, data: Mapping[str, Any]
    ) -> AgentEvent:
        event = AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )
        if control.trace is not None:
            control.trace.record_event(event)
        return event

    async def _raw_checkpoint_event(
        self, state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentEvent:
        sequence = control.next_sequence()
        snapshot = self._snapshot(state, context, control, sequence=sequence)
        checkpoint_id = self._checkpoint_id(sequence)
        event = AgentEvent(
            EventTypes.CHECKPOINT,
            snapshot.to_dict(),
            run_id=control.run_id,
            sequence=sequence,
        )
        await self._save_checkpoint(
            control,
            checkpoint_id=checkpoint_id,
            snapshot=snapshot,
            parent_checkpoint_id=control.last_checkpoint_id,
        )
        control.last_checkpoint = RunSnapshot.from_dict(event.data)
        control.last_checkpoint_id = checkpoint_id
        if control.trace is not None:
            control.trace.record_event(event)
        if control.trace is not None:
            control.trace.mark_durable()
        return event

    @staticmethod
    def _consume_background_task_exception(task: asyncio.Future[Any]) -> None:
        with suppress(BaseException):
            task.exception()

    async def _before_model(
        self, request: ModelRequest, context: RuntimeContext, control: RunControlState
    ) -> ModelRequest:
        current = request
        for hook in self._hooks:
            replacement = await self._call_hook(
                hook.before_model, current, context, control=control
            )
            if replacement is not None:
                current = cast(ModelRequest, replacement)
            current = ModelRequest.from_dict(current.to_dict())
        return ModelRequest.from_dict(current.to_dict())

    async def _after_model(
        self, response: ModelResponse, context: RuntimeContext, control: RunControlState
    ) -> ModelResponse:
        current = response
        for hook in self._hooks:
            replacement = await self._call_hook(hook.after_model, current, context, control=control)
            if replacement is not None:
                current = cast(ModelResponse, replacement)
            current = ModelResponse.from_dict(current.to_dict())
        return ModelResponse.from_dict(current.to_dict())

    async def _on_model_error(
        self,
        error: ModelProviderError,
        request: ModelRequest,
        context: RuntimeContext,
        control: RunControlState,
    ) -> ModelErrorDecision:
        current = ModelErrorDecision()
        info = error.info
        for hook in self._hooks:
            replacement = await self._call_hook(
                hook.on_model_error, info, request, context, control=control
            )
            if replacement is not None:
                if not isinstance(cast(object, replacement), ModelErrorDecision):
                    raise TypeError("on_model_error must return ModelErrorDecision or None")
                current = ModelErrorDecision(
                    retry=replacement.retry,
                    message=replacement.message,
                )
        return current

    async def _before_tool(
        self, call: ToolCall, context: RuntimeContext, control: RunControlState
    ) -> ToolCall:
        current = call
        for hook in self._hooks:
            replacement = await self._call_hook(hook.before_tool, current, context, control=control)
            if replacement is not None:
                current = cast(ToolCall, replacement)
            current = ToolCall.from_dict(current.to_dict())
        return ToolCall.from_dict(current.to_dict())

    async def _after_tool(
        self, result: ToolOutput, context: RuntimeContext, control: RunControlState
    ) -> ToolOutput:
        current = result
        for hook in self._hooks:
            replacement = await self._call_hook(hook.after_tool, current, context, control=control)
            if replacement is not None:
                current = cast(ToolOutput, replacement)
            current = self._normalize_tool_output(current)
        return self._normalize_tool_output(current)

    async def _notify_hooks(self, name: str, *args: object, control: RunControlState) -> None:
        for hook in self._hooks:
            method = getattr(hook, name)
            await self._call_hook(method, *args, control=control)

    async def _call_hook(
        self,
        method: Callable[..., Any],
        *args: object,
        control: RunControlState,
    ) -> Any:
        if iscoroutinefunction(method):
            return await self._await_with_timeout(method(*args), control)
        value = await self._await_with_timeout(asyncio.to_thread(method, *args), control)
        if isawaitable(value):
            return await self._await_with_timeout(value, control)
        return value

    @staticmethod
    def _replace_tool_call_in_history(
        state: AgentState, original_call_id: str, replacement: ToolCall
    ) -> None:
        for message in reversed(state.messages):
            if message.role != "assistant":
                continue
            for index, call in enumerate(message.tool_calls):
                if call.id == original_call_id:
                    message.tool_calls[index] = ToolCall.from_dict(replacement.to_dict())
                    return

    @staticmethod
    def _normalize_tool_output(output: ToolOutput) -> ToolOutput:
        if isinstance(output, ToolObservation):
            return ToolObservation.from_dict(output.to_dict())
        if isinstance(output, ToolAcceptance):
            return ToolAcceptance.from_dict(output.to_dict())
        if isinstance(output, ToolRejection):
            return ToolRejection.from_dict(output.to_dict())
        return ToolOutput.from_dict(output.to_dict())

    @staticmethod
    def _validate_tool_output_mode(call: ToolCall, output: ToolOutput) -> None:
        if call.mode == "execute" and output.kind != "observation":
            raise AgentError("execute tool call must produce ToolObservation")
        if call.mode == "accept" and output.kind not in {"acceptance", "rejection"}:
            raise AgentError("accept tool call must produce ToolAcceptance or ToolRejection")
        if call.mode not in {"execute", "accept"} and output.kind in {
            "observation",
            "acceptance",
            "rejection",
        }:
            raise AgentError("custom tool call must produce an extension ToolOutput kind")

    @staticmethod
    def _validate_non_invoked_tool_output(call: ToolCall, output: ToolOutput) -> None:
        if not output.is_error:
            raise AgentError("non-invoked tool result must remain an error")
        if output.pause is not None:
            raise AgentError("non-invoked tool result must not request pause")
        if call.mode == "accept" and output.kind != "rejection":
            raise AgentError("non-invoked accept tool result must remain a rejection")

    @staticmethod
    def _restore_state_from_snapshot(state: AgentState, snapshot: RunSnapshot) -> None:
        restored = AgentState.from_dict(snapshot.state.to_dict())
        state.status = restored.status
        state.messages = restored.messages
        state.pending_tool_calls = restored.pending_tool_calls
        state.iterations = restored.iterations
        state.total_tool_calls = restored.total_tool_calls
        state.total_usage = restored.total_usage
        state.final_parts = restored.final_parts
        state.error = restored.error
        state.pause = restored.pause

    @staticmethod
    def _snapshot(
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        sequence: int | None = None,
    ) -> RunSnapshot:
        context_data = context.to_dict()
        context_data["run_id"] = control.run_id
        context_data["started_at"] = control.started_at
        context_data["deadline"] = control.deadline
        context_data["sequence"] = control.sequence if sequence is None else sequence
        return RunSnapshot(
            state=AgentState.from_dict(state.to_dict()),
            context=RuntimeContext.from_dict(context_data),
        )

    @staticmethod
    def _result(
        state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentResult:
        _ = context
        snapshot = control.last_checkpoint
        trace = None if control.trace is None else control.trace.to_trace()
        return AgentResult(
            status=state.status,
            final_parts=tuple(state.final_parts),
            messages=tuple(state.messages),
            iterations=state.iterations,
            total_tool_calls=state.total_tool_calls,
            total_usage=None
            if state.total_usage is None
            else ModelUsage.from_dict(state.total_usage.to_dict()),
            error=state.error,
            run_id=control.run_id,
            snapshot=snapshot,
            trace=trace,
        )
