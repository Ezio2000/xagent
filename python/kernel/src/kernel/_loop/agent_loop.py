"""Agent loop implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, suppress
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic, time
from typing import Any, cast

from kernel._frozen import freeze_value
from kernel._loop.helpers import (
    approval_denial_output,
    background_task_event_type,
    batch_call_index,
    begin_tool_execution,
    build_snapshot,
    child_run_event_data,
    clear_pause_request,
    clear_tool_cancel,
    finish_tool_execution,
    is_active_tool_cancel,
    is_prefix_tool_batch,
    normalize_tool_output,
    record_model_usage,
    replace_tool_call_in_history,
    restore_state_from_snapshot,
    rollback_trace_to_durable,
    tool_call_snapshots,
    tool_calls_from_snapshots,
    tool_error_output,
    tool_output_from_snapshot,
    tool_output_snapshot,
    validate_non_invoked_tool_output,
    validate_scheduler_progress,
    validate_tool_output_mode,
)
from kernel._loop.hooks import HookMixin
from kernel._loop.timeouts import TimeoutMixin
from kernel._loop.types import (
    AppliedTransition,
    EmptyToolRegistry,
    PreparedToolBatch,
    RunControlState,
    RuntimeConversationInsert,
    RuntimePauseInterrupt,
    RuntimeTimeoutError,
    ToolProgressRecord,
    ToolSchedulerFactory,
    TracePayload,
    default_tool_scheduler_factory,
)
from kernel._trace import TraceRecorder, TraceStepKinds
from kernel.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from kernel.context import RuntimeContext
from kernel.control import ConversationInsert, PauseRequest, RunController, ToolCancelRequest
from kernel.errors import (
    AgentError,
    InvalidToolCall,
    LimitExceeded,
    ModelProviderError,
    ToolError,
)
from kernel.events import CORE_EVENT_TYPES, AgentEvent, EventEmitter, EventTypes
from kernel.hooks import RuntimeHook
from kernel.journal import JournalRecord, RunJournal
from kernel.limits import LimitReasons, LoopLimits
from kernel.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_part_without_metadata,
    content_parts_summary,
)
from kernel.models import (
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
from kernel.resume import ResumeInput
from kernel.scheduler import (
    ToolBatch,
    ToolCatalog,
    ToolCompleted,
    ToolSchedulerProtocol,
    ToolStarted,
)
from kernel.snapshot import RunSnapshot
from kernel.state import AgentState, PauseState
from kernel.status import AgentStatus
from kernel.store import RunStore, StoredCheckpoint
from kernel.tools import (
    ToolOutput,
    ToolRegistryProtocol,
    normalized_tool_risk,
)

_MAX_DISPATCHED_EVENTS_PER_RUNTIME_EVENT = 1000

__all__ = ["AgentLoop", "AgentResult", "ToolSchedulerFactory"]


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
    trace: TracePayload | None = None


class AgentLoop(TimeoutMixin, HookMixin):
    """Lightweight state-machine-driven agent loop."""

    __slots__ = (
        "_hooks",
        "_limits",
        "_model",
        "_model_options",
        "_approval_metadata",
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
    _approval_metadata: Mapping[str, Any]
    _approval_policy: ApprovalPolicy | None
    _response_format: ResponseFormat | None
    _run_journal: RunJournal | None
    _run_store: RunStore | None
    _tool_scheduler_factory: ToolSchedulerFactory
    _tool_choice: ToolChoice
    _trace_enabled: bool
    _tools: ToolRegistryProtocol

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistryProtocol | None = None,
        limits: LoopLimits | None = None,
        model_options: ModelOptions | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        hooks: Sequence[RuntimeHook] | None = None,
        approval_metadata: Mapping[str, Any] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        run_store: RunStore | None = None,
        run_journal: RunJournal | None = None,
        trace: bool = True,
        tool_scheduler_factory: ToolSchedulerFactory | None = None,
    ) -> None:
        if not isinstance(cast(object, trace), bool):
            raise TypeError("trace must be a boolean")
        self._model = model
        if tools is None:
            self._tools = EmptyToolRegistry()
        elif self._is_tool_registry(tools):
            self._tools = tools
        else:
            raise TypeError("tools must implement ToolRegistryProtocol")
        self._limits = limits or LoopLimits()
        self._limits.validate()
        self._model_options = ModelOptions.from_dict((model_options or ModelOptions()).to_dict())
        self._tool_choice = ToolChoice.from_dict((tool_choice or ToolChoice()).to_dict())
        self._response_format = (
            None if response_format is None else ResponseFormat.from_dict(response_format.to_dict())
        )
        self._hooks = tuple(hooks or ())
        if approval_metadata is not None and not isinstance(
            cast(object, approval_metadata), Mapping
        ):
            raise TypeError("approval_metadata must be a mapping or None")
        self._approval_metadata = deepcopy(
            dict(cast(Mapping[str, Any], {} if approval_metadata is None else approval_metadata))
        )
        self._approval_policy = approval_policy
        self._run_store = run_store
        self._run_journal = run_journal
        self._trace_enabled = trace
        self._tool_scheduler_factory = tool_scheduler_factory or default_tool_scheduler_factory

    @staticmethod
    def _is_tool_registry(value: object) -> bool:
        return isinstance(value, ToolRegistryProtocol) and all(
            callable(getattr(value, name, None))
            for name in ("specs", "spec_for", "validate_call", "invoke")
        )

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
        run_started_yielded = False
        child_run_started = False
        child_run_completed = False
        control.initial_snapshot = build_snapshot(state, context, control)
        try:
            for event in await self._events(
                context,
                control,
                EventTypes.RUN_STARTED,
                {"state": state.summary()},
                trace_before_hooks=True,
            ):
                yield event
            run_started_yielded = True
            if context.parent_run_id is not None:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.CHILD_RUN_STARTED,
                    child_run_event_data(context),
                ):
                    yield event
                child_run_started = True

            drive_iterator = self._drive(state, context, control, stream=stream).__aiter__()
            async with aclosing(self._pump_events(drive_iterator)) as events:
                async for event in events:
                    yield event

            terminal_checkpoint_committed = state.is_terminal
            if state.status is not AgentStatus.PAUSED:
                clear_pause_request(control)

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

            if context.parent_run_id is not None:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.CHILD_RUN_COMPLETED,
                    child_run_event_data(context) | {"status": state.status.value},
                ):
                    yield event
                child_run_completed = True

            for event in await self._events(
                context, control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
            ):
                yield event
            clear_pause_request(control)
        except RuntimeTimeoutError:
            async for event in self._terminal_failure_events(
                state,
                context,
                control,
                terminal_status=AgentStatus.LIMIT_EXCEEDED,
                message=LimitReasons.TIMEOUT_SECONDS,
                run_started_yielded=run_started_yielded,
                child_run_started=child_run_started,
                child_run_completed=child_run_completed,
                terminal_checkpoint_committed=terminal_checkpoint_committed,
            ):
                yield event
        except Exception as exc:  # pragma: no cover - defensive boundary
            async for event in self._terminal_failure_events(
                state,
                context,
                control,
                terminal_status=AgentStatus.FAILED,
                message=str(exc) or exc.__class__.__name__,
                run_started_yielded=run_started_yielded,
                child_run_started=child_run_started,
                child_run_completed=child_run_completed,
                terminal_checkpoint_committed=terminal_checkpoint_committed,
            ):
                yield event

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
        record_model_usage(state, response.usage)
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
        if transition is None:
            raise RuntimeError("model response transition metadata is missing")
        for event in await self._transition_boundary_events(
            state, context, control, transition=transition, transition_data=transition_data
        ):
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

        try:
            while True:
                try:
                    stream_event = await self._anext_model_with_interrupt(iterator, control)
                except StopAsyncIteration:
                    break
                accumulator.apply(stream_event)
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

        response_holder.append(accumulator.response())

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
            pending_snapshots = tool_call_snapshots(pending_calls)
            batch = self._tool_scheduler(control).next_batch(
                tool_calls_from_snapshots(pending_snapshots)
            )
            if batch is None:
                for event in await self._limit(
                    state, context, control, LimitReasons.MAX_TOTAL_TOOL_CALLS
                ):
                    yield event
                return
            if not is_prefix_tool_batch(batch, pending_snapshots):
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
                calls=tool_calls_from_snapshots(pending_snapshots[: len(batch.calls)]),
                parallel=batch.parallel,
            )
            scheduler_policy_error = self._scheduler_batch_policy_error(batch)
            if scheduler_policy_error is not None:
                for event in await self._transition(
                    state,
                    AgentStatus.FAILED,
                    context,
                    control,
                    error=scheduler_policy_error,
                ):
                    yield event
                return

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
            replace_tool_call_in_history(state.messages, cast(str, raw_snapshot["id"]), call)
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
                precomputed_results[call.id] = approval_denial_output(
                    call, decision.reason, decision.metadata
                )
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
        batch_snapshots = tool_call_snapshots(batch.calls)
        scheduler_batch = ToolBatch(
            id=batch.id,
            calls=tool_calls_from_snapshots(batch_snapshots),
            parallel=batch.parallel,
        )
        progress_queue: asyncio.Queue[ToolProgressRecord] = asyncio.Queue()

        def reject_scheduler(message: str) -> None:
            nonlocal scheduler_error
            error = AgentError(message)
            scheduler_error = error
            raise error

        async def execute(call: ToolCall) -> ToolOutput:
            call_index = batch_call_index(batch_snapshots, call)
            if call_index not in started_indices:
                reject_scheduler("tool scheduler must start a batch call before executing it")
            if call_index in executing_indices or call_index in execute_results:
                reject_scheduler("tool scheduler must execute each batch call at most once")
            executing_indices.add(call_index)
            try:
                canonical_call = ToolCall.from_dict(batch_snapshots[call_index])
                precomputed = precomputed_results.get(canonical_call.id)
                if precomputed is not None:
                    result = normalize_tool_output(precomputed)
                else:

                    def emit_progress(data: Mapping[str, Any]) -> None:
                        progress_queue.put_nowait(
                            ToolProgressRecord(
                                call=ToolCall.from_dict(canonical_call.to_dict()),
                                batch_id=batch.id,
                                index=call_index,
                                data=dict(data),
                            )
                        )

                    controller = control.run_controller
                    try:
                        result = await self._await_with_timeout(
                            self._execute_tool(
                                canonical_call,
                                context,
                                progress_emitter=emit_progress,
                                cancel_checker=None
                                if controller is None
                                else lambda call_id=canonical_call.id: (
                                    call_id in control.active_tool_call_ids
                                    and controller.is_tool_cancelled(call_id)
                                ),
                            ),
                            control,
                        )
                    except BaseException:
                        finish_tool_execution(control, canonical_call.id)
                        raise
                    result = normalize_tool_output(result)
                result_snapshot = tool_output_snapshot(result)
                execute_results[call_index] = result_snapshot
                return tool_output_from_snapshot(result_snapshot)
            finally:
                executing_indices.discard(call_index)

        async for progress in self._run_tool_scheduler_events(
            scheduler_batch,
            execute,
            control,
            progress_queue,
        ):
            if isinstance(progress, ToolProgressRecord):
                for event in await self._events(
                    context,
                    control,
                    EventTypes.TOOL_PROGRESS,
                    {
                        "id": progress.call.id,
                        "name": progress.call.name,
                        "mode": progress.call.mode,
                        "batch_id": progress.batch_id,
                        "parallel": batch.parallel,
                        "index": progress.index,
                        "progress": dict(progress.data),
                    },
                ):
                    yield event
                continue
            if isinstance(progress, ToolCancelRequest):
                if is_active_tool_cancel(control, progress):
                    for event in await self._events(
                        context,
                        control,
                        EventTypes.TOOL_CANCEL_REQUESTED,
                        {
                            "id": progress.tool_call_id,
                            "reason": progress.reason,
                            "source": progress.source,
                            "metadata": progress.metadata,
                        },
                    ):
                        yield event
                else:
                    clear_tool_cancel(control, progress.tool_call_id)
                continue
            if scheduler_error is not None:
                raise scheduler_error
            validate_scheduler_progress(batch, batch_snapshots, progress)
            canonical_call = ToolCall.from_dict(batch_snapshots[progress.index])
            if isinstance(progress, ToolStarted):
                if progress.index in started_indices:
                    raise AgentError("tool scheduler emitted duplicate tool_started")
                if progress.index in completed_indices:
                    raise AgentError("tool scheduler started a completed tool call")
                max_active = self._active_tool_call_limit(batch)
                active_indices = started_indices - completed_indices
                if len(active_indices) >= max_active:
                    raise AgentError("tool scheduler exceeded max active tool calls")
                started_indices.add(progress.index)
                implementation_invoked = canonical_call.id not in precomputed_results
                if implementation_invoked:
                    begin_tool_execution(control, canonical_call.id)
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
            progress_result_snapshot = tool_output_snapshot(progress.result)
            if progress_result_snapshot != expected_result:
                raise AgentError("tool scheduler must not replace execute results")
            completed_indices.add(progress.index)
            implementation_invoked = canonical_call.id not in precomputed_results
            try:
                result = await self._after_tool(
                    tool_output_from_snapshot(expected_result), context, control
                )
                validate_tool_output_mode(canonical_call, result)
                if not implementation_invoked:
                    validate_non_invoked_tool_output(canonical_call, result)
                tool_message = result.to_message(canonical_call)
                completed[progress.index] = (canonical_call, result, tool_message)
                tool_completed_events = await self._events(
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
                )
                background_events = await self._background_task_events(
                    canonical_call,
                    result,
                    batch,
                    progress.index,
                    implementation_invoked,
                    context,
                    control,
                )
            finally:
                finish_tool_execution(control, canonical_call.id)
            for event in tool_completed_events:
                yield event
            for event in background_events:
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
                for event in await self._transition_boundary_events(
                    state,
                    context,
                    control,
                    transition=transition,
                    transition_data=transition.data,
                ):
                    yield event
                return

            for event in await self._checkpoint_events(state, context, control):
                yield event

        if scheduler_error is not None:
            raise scheduler_error
        if completed_indices != set(range(len(batch.calls))):
            raise AgentError("tool scheduler ended before completing the selected batch")

    def _scheduler_batch_policy_error(self, batch: ToolBatch) -> str | None:
        if not self._is_non_empty_string(batch.id):
            return "tool scheduler batch id must be a non-empty string"
        if not self._is_bool(batch.parallel):
            return "tool scheduler batch parallel flag must be a boolean"
        if not batch.calls:
            return "tool scheduler must return a non-empty prefix batch"
        if self._limits.stop_on_tool_error and len(batch.calls) > 1:
            return "tool scheduler must use single-call batches when stop_on_tool_error is enabled"
        if batch.parallel and len(batch.calls) > 1:
            if self._limits.max_parallel_tool_calls <= 1:
                return "tool scheduler returned a parallel batch while parallelism is disabled"
            unsafe = [call.name for call in batch.calls if not self._parallel_eligible(call)]
            if unsafe:
                names = ", ".join(unsafe)
                return (
                    "tool scheduler returned non-parallel-safe tool(s) in a parallel batch: "
                    f"{names}"
                )
        return None

    @staticmethod
    def _is_non_empty_string(value: object) -> bool:
        return isinstance(value, str) and bool(value)

    @staticmethod
    def _is_bool(value: object) -> bool:
        return isinstance(value, bool)

    def _parallel_eligible(self, call: ToolCall) -> bool:
        spec = self._tools.spec_for(call.name)
        if spec is None:
            return False
        annotations = spec.annotations
        return (
            annotations.get("parallel_safe") is True
            and annotations.get("read_only") is True
            and annotations.get("idempotent") is True
        )

    def _active_tool_call_limit(self, batch: ToolBatch) -> int:
        if self._limits.stop_on_tool_error or not batch.parallel:
            return 1
        return self._limits.max_parallel_tool_calls

    async def _run_tool_scheduler_events(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        control: RunControlState,
        progress_queue: asyncio.Queue[ToolProgressRecord],
    ) -> AsyncIterator[ToolStarted | ToolCompleted | ToolProgressRecord | ToolCancelRequest]:
        iterator = (
            self._tool_scheduler(control)
            .run_batch(
                batch,
                execute,
                stop_on_error=self._limits.stop_on_tool_error,
            )
            .__aiter__()
        )
        scheduler_task: asyncio.Task[ToolStarted | ToolCompleted] | None = asyncio.ensure_future(
            anext(iterator)
        )
        progress_task: asyncio.Task[ToolProgressRecord] | None = asyncio.ensure_future(
            progress_queue.get()
        )
        cancel_task = self._tool_cancel_task(control)

        def drain_progress_queue() -> tuple[ToolProgressRecord, ...]:
            records: list[ToolProgressRecord] = []
            while True:
                try:
                    records.append(progress_queue.get_nowait())
                except asyncio.QueueEmpty:
                    return tuple(records)

        try:
            while scheduler_task is not None:
                tasks: set[asyncio.Task[Any]] = {scheduler_task}
                if progress_task is not None:
                    tasks.add(progress_task)
                if cancel_task is not None:
                    tasks.add(cancel_task)
                done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if progress_task is not None and progress_task in done:
                    yield progress_task.result()
                    for progress in drain_progress_queue():
                        yield progress
                    progress_task = asyncio.ensure_future(progress_queue.get())
                if cancel_task is not None and cancel_task in done:
                    yield cancel_task.result()
                    cancel_task = self._tool_cancel_task(control)
                if scheduler_task in done:
                    if progress_task is not None and progress_task.done():
                        yield progress_task.result()
                        for progress in drain_progress_queue():
                            yield progress
                        progress_task = asyncio.ensure_future(progress_queue.get())
                    else:
                        for progress in drain_progress_queue():
                            yield progress
                    try:
                        yield scheduler_task.result()
                    except StopAsyncIteration:
                        scheduler_task = None
                    else:
                        scheduler_task = asyncio.ensure_future(anext(iterator))
        finally:
            for call in batch.calls:
                finish_tool_execution(control, call.id)
            for task in (scheduler_task, progress_task, cancel_task):
                if task is not None and not task.done():
                    task.cancel()
            close = getattr(iterator, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()

    @staticmethod
    def _tool_cancel_task(control: RunControlState) -> asyncio.Task[ToolCancelRequest] | None:
        controller = control.run_controller
        if controller is None:
            return None
        return asyncio.ensure_future(controller.wait_for_tool_cancel())

    async def _background_task_events(
        self,
        call: ToolCall,
        result: ToolOutput,
        batch: ToolBatch,
        index: int,
        implementation_invoked: bool,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        task = result.background_task
        if task is None:
            return ()
        event_type = background_task_event_type(task)
        return await self._events(
            context,
            control,
            event_type,
            {
                "task": task.to_dict(),
                "tool_call": {
                    "id": call.id,
                    "name": call.name,
                    "mode": call.mode,
                    "batch_id": batch.id,
                    "parallel": batch.parallel,
                    "index": index,
                    "implementation_invoked": implementation_invoked,
                },
            },
        )

    async def _raw_child_run_started_events_if_needed(
        self,
        context: RuntimeContext,
        control: RunControlState,
        *,
        child_run_started: bool,
    ) -> tuple[AgentEvent, ...]:
        if child_run_started or context.parent_run_id is None:
            return ()
        return (
            await self._raw_event(
                control,
                EventTypes.CHILD_RUN_STARTED,
                child_run_event_data(context),
                record_trace=not self._trace_has_kind(control, TraceStepKinds.CHILD_RUN_STARTED),
            ),
        )

    async def _raw_child_run_completed_events(
        self,
        context: RuntimeContext,
        control: RunControlState,
        state: AgentState,
        *,
        child_run_started: bool,
        child_run_completed: bool,
    ) -> tuple[AgentEvent, ...]:
        if not child_run_started or child_run_completed:
            return ()
        record_trace = self._trace_has_child_run_started(control)
        return (
            await self._raw_event(
                control,
                EventTypes.CHILD_RUN_COMPLETED,
                child_run_event_data(context) | {"status": state.status.value},
                record_trace=record_trace,
            ),
        )

    @staticmethod
    def _trace_has_child_run_started(control: RunControlState) -> bool:
        if control.trace is None:
            return True
        kinds = control.trace.kinds()
        return (
            TraceStepKinds.CHILD_RUN_STARTED in kinds
            and TraceStepKinds.CHILD_RUN_COMPLETED not in kinds
        )

    async def _raw_run_started_events_if_needed(
        self,
        state: AgentState,
        control: RunControlState,
        *,
        run_started_yielded: bool,
    ) -> tuple[AgentEvent, ...]:
        if run_started_yielded:
            return ()
        return (
            await self._raw_event(
                control,
                EventTypes.RUN_STARTED,
                {"state": state.summary()},
                record_trace=not self._trace_has_kind(control, TraceStepKinds.RUN_STARTED),
            ),
        )

    @staticmethod
    def _trace_has_kind(control: RunControlState, kind: str) -> bool:
        if control.trace is None:
            return False
        return control.trace.has_kind(kind)

    def _record_child_run_started_trace_if_needed(
        self,
        control: RunControlState,
        events: tuple[AgentEvent, ...],
    ) -> None:
        if not events or control.trace is None:
            return
        if self._trace_has_kind(control, TraceStepKinds.CHILD_RUN_STARTED):
            return
        control.trace.record_event(events[0])

    async def _terminal_failure_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        terminal_status: AgentStatus,
        message: str,
        run_started_yielded: bool,
        child_run_started: bool,
        child_run_completed: bool,
        terminal_checkpoint_committed: bool,
    ) -> AsyncIterator[AgentEvent]:
        for event in await self._raw_run_started_events_if_needed(
            state, control, run_started_yielded=run_started_yielded
        ):
            yield event
        raw_child_started_events = await self._raw_child_run_started_events_if_needed(
            context, control, child_run_started=child_run_started
        )
        for event in raw_child_started_events:
            yield event
        child_run_started = child_run_started or bool(raw_child_started_events)

        if not terminal_checkpoint_committed:
            durable = control.last_checkpoint or control.initial_snapshot
            if durable is None:
                raise RuntimeError("terminal failure recovery requires an initial snapshot")
            restore_state_from_snapshot(state, durable)
            rollback_trace_to_durable(control)
            self._record_child_run_started_trace_if_needed(control, raw_child_started_events)

        if terminal_checkpoint_committed or state.is_terminal:
            clear_pause_request(control)
            yield await self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": message},
            )
            for event in await self._raw_child_run_completed_events(
                context,
                control,
                state,
                child_run_started=child_run_started,
                child_run_completed=child_run_completed,
            ):
                yield event
            yield await self._raw_event(
                control, EventTypes.RUN_COMPLETED, {"state": state.summary()}
            )
            return

        previous = state.status
        state.status = terminal_status
        state.error = message
        state.pause = None
        clear_pause_request(control)
        yield await self._raw_event(
            control,
            EventTypes.STATE_CHANGED,
            {
                "from": previous.value,
                "to": state.status.value,
                "iterations": state.iterations,
                "total_tool_calls": state.total_tool_calls,
                "total_usage": None if state.total_usage is None else state.total_usage.to_dict(),
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
        for event in await self._raw_child_run_completed_events(
            context,
            control,
            state,
            child_run_started=child_run_started,
            child_run_completed=child_run_completed,
        ):
            yield event
        yield await self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.summary()})

    async def _approval_request(
        self, call: ToolCall, context: RuntimeContext, control: RunControlState
    ) -> tuple[ApprovalRequest, tuple[AgentEvent, ...]]:
        policy = self._approval_policy
        if policy is None:
            raise RuntimeError("approval request requires an approval policy")
        spec = self._tools.spec_for(call.name)
        risk = {} if spec is None else dict(normalized_tool_risk(spec.annotations))
        request = ApprovalRequest(
            tool_call=call,
            tool_spec=spec,
            context=context,
            risk=risk,
            metadata=self._approval_metadata,
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

    def _tool_validation_error_output(self, call: ToolCall) -> ToolOutput | None:
        try:
            self._tools.validate_call(call)
        except InvalidToolCall as exc:
            return tool_error_output(call, exc)
        return None

    async def _execute_tool(
        self,
        call: ToolCall,
        context: RuntimeContext,
        *,
        progress_emitter: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> ToolOutput:
        try:
            tool_context = RuntimeContext.from_dict(context.to_dict())
            return await self._tools.invoke(
                call,
                tool_context,
                progress_emitter=progress_emitter,
                cancel_checker=cancel_checker,
            )
        except (InvalidToolCall, ToolError) as exc:
            return tool_error_output(call, exc)

    def _remaining_tool_call_slots(self, state: AgentState) -> int:
        return max(0, self._limits.max_total_tool_calls - state.total_tool_calls)

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

    def _seed_resume_checkpoint(self, control: RunControlState, snapshot: RunSnapshot) -> None:
        control.last_checkpoint_id = self._checkpoint_id(snapshot.context.sequence)

    def _model_supports_streaming(self) -> bool:
        return model_capabilities(self._model).streaming

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

    async def _transition_boundary_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        transition: AppliedTransition,
        transition_data: Mapping[str, Any],
    ) -> tuple[AgentEvent, ...]:
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
            return self._merge_deferred_events(transition_events, limit_events)

        pause_events = await self._pause_if_requested(state, context, control)
        if pause_events:
            if transition_event is not None:
                transition_events = await self._dispatch_deferred_transition_event(
                    transition_to_notify, transition_event, context, control
                )
            return self._merge_deferred_events(transition_events, pause_events)

        if transition_event is not None:
            return await self._checkpoint_events_after_deferred_events(
                transition_event,
                state,
                context,
                control,
                transition=transition_to_notify,
            )
        checkpoint_events = await self._checkpoint_events(state, context, control)
        return (*transition_events, *checkpoint_events)

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
        snapshot = build_snapshot(state, context, control, sequence=sequence)
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
                method = self._hook_method(hook, "on_event")
                if method is None:
                    continue
                replacement = await self._call_hook(
                    method, event, context, emitter, control=control
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
        self,
        control: RunControlState,
        event_type: str,
        data: Mapping[str, Any],
        *,
        record_trace: bool = True,
    ) -> AgentEvent:
        event = AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )
        if record_trace and control.trace is not None:
            control.trace.record_event(event)
        return event

    async def _raw_checkpoint_event(
        self, state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentEvent:
        sequence = control.next_sequence()
        snapshot = build_snapshot(state, context, control, sequence=sequence)
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
    def _result(
        state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentResult:
        _ = context
        snapshot = control.last_checkpoint
        trace = (
            None
            if control.trace is None
            else cast(
                TracePayload,
                freeze_value(
                    control.trace.to_payload(),
                    error_message="trace payload is immutable",
                ),
            )
        )
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
