"""Agent loop implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable, iscoroutinefunction
from time import monotonic, time
from typing import Any, TypeAlias, TypeVar, cast

from agent_runtime.control import ConversationInsert, PauseRequest, RunController
from agent_runtime.errors import AgentError, InvalidToolCall, ModelProviderError, ToolError
from agent_runtime.events import CORE_EVENT_TYPES, AgentEvent, EventEmitter, EventTypes
from agent_runtime.hooks import ModelErrorDecision, RuntimeHook
from agent_runtime.limits import LoopLimits
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
from agent_runtime.scheduler import ToolBatch, ToolScheduler, ToolStarted
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import AgentState, AgentStatus, PauseState
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
ToolSchedulerFactory: TypeAlias = Callable[[ToolRegistry, LoopLimits], ToolScheduler]
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


def _default_tool_scheduler_factory(tools: ToolRegistry, limits: LoopLimits) -> ToolScheduler:
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


@dataclass(slots=True)
class RunControlState:
    """Runtime-owned control state that hooks cannot mutate."""

    run_id: str
    started_at: float
    deadline: float | None = None
    monotonic_deadline: float | None = None
    run_controller: RunController | None = None
    trace: TraceRecorder | None = None
    tool_scheduler: ToolScheduler | None = None
    initial_snapshot: RunSnapshot | None = None
    last_checkpoint: RunSnapshot | None = None
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
        "_response_format",
        "_tool_scheduler_factory",
        "_tool_choice",
        "_trace_enabled",
        "_tools",
    )

    _hooks: tuple[RuntimeHook, ...]
    _limits: LoopLimits
    _model: ModelClient
    _model_options: ModelOptions
    _response_format: ResponseFormat | None
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
            while True:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            runtime_context.sequence = control.sequence
            await self._close_async_iterator(iterator)

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
        if control.trace is not None:
            control.trace.record_resume(resume_input, working_state)
        iterator = self._run_prepared_state_events(
            working_state, runtime_context, control, stream=stream
        ).__aiter__()
        try:
            while True:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            runtime_context.sequence = control.sequence
            await self._close_async_iterator(iterator)

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
            while True:
                try:
                    await anext(iterator)
                except StopAsyncIteration:
                    break
        finally:
            runtime_context.sequence = control.sequence
            await self._close_async_iterator(iterator)
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
            while True:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            runtime_context.sequence = control.sequence
            await self._close_async_iterator(iterator)

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
            try:
                while True:
                    try:
                        event = await anext(drive_iterator)
                    except StopAsyncIteration:
                        break
                    yield event
            finally:
                await self._close_async_iterator(drive_iterator)

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
                yield self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": "timeout_seconds"},
                )
                yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.summary()})
                return
            durable = control.last_checkpoint or control.initial_snapshot
            self._restore_state_from_snapshot(state, durable)
            self._rollback_trace_to_durable(control)
            previous = state.status
            state.status = AgentStatus.LIMIT_EXCEEDED
            state.error = "timeout_seconds"
            state.pause = None
            self._clear_pause_request(control)
            yield self._raw_event(
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
            yield self._raw_checkpoint_event(state, context, control)
            yield self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.summary()})
        except Exception as exc:  # pragma: no cover - defensive boundary
            if terminal_checkpoint_committed:
                self._clear_pause_request(control)
                yield self._raw_event(
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": str(exc) or exc.__class__.__name__},
                )
                yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.summary()})
                return
            durable = control.last_checkpoint or control.initial_snapshot
            self._restore_state_from_snapshot(state, durable)
            self._rollback_trace_to_durable(control)
            previous = state.status
            state.status = AgentStatus.FAILED
            state.error = str(exc) or exc.__class__.__name__
            state.pause = None
            self._clear_pause_request(control)
            yield self._raw_event(
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
            yield self._raw_checkpoint_event(state, context, control)
            yield self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.summary()})

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
                try:
                    while True:
                        try:
                            event = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        yield event
                finally:
                    await self._close_async_iterator(iterator)
                continue

            if state.status is AgentStatus.EXECUTING_TOOLS:
                iterator = self._tool_step(state, context, control).__aiter__()
                try:
                    while True:
                        try:
                            event = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        yield event
                finally:
                    await self._close_async_iterator(iterator)
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
                    try:
                        while True:
                            try:
                                event = await anext(iterator)
                            except StopAsyncIteration:
                                break
                            yield event
                    finally:
                        await self._close_async_iterator(iterator)
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
                for event in await self._limit(state, context, control, "timeout_seconds"):
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
        if assistant_message.tool_calls:
            state.pending_tool_calls = list(assistant_message.tool_calls)
            limit_reason = self._limits.tool_call_reason(state) or self._limits.usage_reason(
                state.total_usage
            )
            if limit_reason is None:
                transition_data = await self._apply_transition(
                    state,
                    AgentStatus.EXECUTING_TOOLS,
                    context,
                    control,
                )
        else:
            limit_reason = self._limits.usage_reason(state.total_usage)
            if limit_reason is None:
                state.final_parts = [content_part_without_metadata(part) for part in response.parts]
                transition_data = await self._apply_transition(
                    state,
                    AgentStatus.COMPLETED,
                    context,
                    control,
                )

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
        transition_events = await self._events(
            context, control, EventTypes.STATE_CHANGED, transition_data
        )
        limit_reason = self._active_limit_reason(state, control)
        if limit_reason is not None:
            limit_events = await self._limit(state, context, control, limit_reason)
            for event in (*transition_events, *limit_events):
                yield event
            return
        pause_events = await self._pause_if_requested(state, context, control)
        if pause_events:
            for event in (*transition_events, *pause_events):
                yield event
            return
        checkpoint_events = await self._checkpoint_events(state, context, control)
        for event in (*transition_events, *checkpoint_events):
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
            batch = self._tool_scheduler(control).next_batch(
                tuple(state.pending_tool_calls[:remaining_slots])
            )
            if batch is None:
                for event in await self._limit(state, context, control, "max_total_tool_calls"):
                    yield event
                return

            prepared = await self._prepare_tool_batch(batch, state, context, control)
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

            try:
                iterator = self._run_tool_batch(prepared, state, context, control).__aiter__()
                try:
                    while True:
                        try:
                            event = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        yield event
                finally:
                    await self._close_async_iterator(iterator)
            except RuntimeTimeoutError:
                for event in await self._limit(state, context, control, "timeout_seconds"):
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

    async def _prepare_tool_batch(
        self,
        batch: ToolBatch,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> ToolBatch | None:
        prepared: list[ToolCall] = []
        for index, raw_call in enumerate(batch.calls):
            call = await self._before_tool(raw_call, context, control)
            if call.id != raw_call.id or call.name != raw_call.name or call.mode != raw_call.mode:
                return None
            self._replace_tool_call_in_history(state, raw_call.id, call)
            state.pending_tool_calls[index] = call
            prepared.append(call)
        return ToolBatch(batch.id, tuple(prepared), batch.parallel)

    async def _run_tool_batch(
        self,
        batch: ToolBatch,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> AsyncIterator[AgentEvent]:
        completed: dict[int, tuple[ToolCall, ToolOutput, Message]] = {}

        async def execute(call: ToolCall) -> ToolOutput:
            return await self._await_with_timeout(self._execute_tool(call, context), control)

        async for progress in self._tool_scheduler(control).run_batch(
            batch, execute, stop_on_error=self._limits.stop_on_tool_error
        ):
            if isinstance(progress, ToolStarted):
                for event in await self._events(
                    context,
                    control,
                    EventTypes.TOOL_STARTED,
                    {
                        "id": progress.call.id,
                        "name": progress.call.name,
                        "mode": progress.call.mode,
                        "arguments": dict(progress.call.arguments),
                        "batch_id": progress.batch.id,
                        "parallel": progress.batch.parallel,
                        "index": progress.index,
                    },
                ):
                    yield event
                continue

            result = await self._after_tool(progress.result, context, control)
            self._validate_tool_output_mode(progress.call, result)
            invocation = ToolInvocation.from_tool_call(progress.call)
            tool_message = result.to_message(invocation)
            completed[progress.index] = (progress.call, result, tool_message)
            for event in await self._events(
                context,
                control,
                EventTypes.TOOL_COMPLETED,
                {
                    "id": progress.call.id,
                    "name": progress.call.name,
                    "mode": progress.call.mode,
                    "batch_id": progress.batch.id,
                    "parallel": progress.batch.parallel,
                    "index": progress.index,
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
                transition_data = await self._apply_transition(
                    state,
                    AgentStatus.PLANNING,
                    context,
                    control,
                )
                transition_events = await self._events(
                    context, control, EventTypes.STATE_CHANGED, transition_data
                )
                limit_reason = self._active_limit_reason(state, control)
                if limit_reason is not None:
                    limit_events = await self._limit(state, context, control, limit_reason)
                    for event in (*transition_events, *limit_events):
                        yield event
                    return
                pause_events = await self._pause_if_requested(state, context, control)
                if pause_events:
                    for event in (*transition_events, *pause_events):
                        yield event
                    return
                checkpoint_events = await self._checkpoint_events(state, context, control)
                for event in (*transition_events, *checkpoint_events):
                    yield event
                return

            for event in await self._checkpoint_events(state, context, control):
                yield event

    async def _execute_tool(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        try:
            tool_context = RuntimeContext.from_dict(context.to_dict())
            return await self._tools.invoke(call, tool_context)
        except (InvalidToolCall, ToolError) as exc:
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

    async def _close_async_iterator(self, iterator: AsyncIterator[object]) -> None:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with suppress(BaseException):
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
        tool_scheduler = self._tool_scheduler_factory(self._tools, self._limits)
        if not isinstance(cast(object, tool_scheduler), ToolScheduler):
            raise TypeError("tool_scheduler_factory must return ToolScheduler")
        control = RunControlState(
            run_id=runtime_context.run_id,
            started_at=runtime_context.started_at,
            deadline=runtime_context.deadline,
            monotonic_deadline=None if remaining is None else now_monotonic + remaining,
            run_controller=controller,
            trace=TraceRecorder(runtime_context.run_id) if self._trace_enabled else None,
            tool_scheduler=tool_scheduler,
            sequence=runtime_context.sequence,
        )
        return runtime_context, control

    @staticmethod
    def _tool_scheduler(control: RunControlState) -> ToolScheduler:
        if control.tool_scheduler is None:
            raise RuntimeError("run control is missing a tool scheduler")
        return control.tool_scheduler

    def _model_supports_streaming(self) -> bool:
        return model_capabilities(self._model).streaming

    @staticmethod
    def _record_model_usage(state: AgentState, usage: ModelUsage | None) -> None:
        existing = state.total_usage
        if usage is None:
            return
        values: dict[str, int] = {}
        for field in _USAGE_FIELDS:
            current = None if existing is None else getattr(existing, field)
            increment = getattr(usage, field)
            if existing is None:
                if increment is not None:
                    values[field] = increment
            elif current is not None and increment is not None:
                values[field] = current + increment
            elif current is not None:
                values[field] = current
            elif increment is not None:
                values[field] = increment
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
            return "timeout_seconds"
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
        events = list(
            await self._events(
                context,
                control,
                EventTypes.CONVERSATION_INSERTED,
                {
                    "insert": insert.to_dict(),
                    "message": message.to_dict(),
                },
            )
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
        events = list(
            await self._events(
                context,
                control,
                EventTypes.PAUSE_REQUESTED,
                {
                    "request": request.to_dict(),
                    "resume_status": resume_status.value,
                    "origin": origin,
                },
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
        transition_data = await self._apply_transition(
            state, status, context, control, error=error, pause=pause
        )
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
        previous = state.status
        state.status = status
        state.error = None if status is AgentStatus.PAUSED else error
        state.pause = pause if status is AgentStatus.PAUSED else None
        transition_state = AgentState.from_dict(state.to_dict())
        await self._notify_hooks(
            "on_transition", previous, status, transition_state, context, control=control
        )
        return {
            "from": previous.value,
            "to": status.value,
            "iterations": state.iterations,
            "total_tool_calls": state.total_tool_calls,
            "total_usage": None if state.total_usage is None else state.total_usage.to_dict(),
            "error": state.error,
            "pause": None if state.pause is None else state.pause.to_dict(),
        }

    async def _checkpoint_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        sequence = control.next_sequence()
        event = AgentEvent(
            EventTypes.CHECKPOINT,
            self._snapshot(state, context, control, sequence=sequence).to_dict(),
            run_id=control.run_id,
            sequence=sequence,
        )
        events = await self._dispatch_events(event, context, control)
        control.last_checkpoint = RunSnapshot.from_dict(event.data)
        if control.trace is not None:
            control.trace.mark_durable()
        return events

    async def _events(
        self,
        context: RuntimeContext,
        control: RunControlState,
        event_type: str,
        data: Mapping[str, Any],
        *,
        trace_before_hooks: bool = False,
    ) -> tuple[AgentEvent, ...]:
        event = AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )
        return await self._dispatch_events(
            event, context, control, trace_before_hooks=trace_before_hooks
        )

    async def _dispatch_events(
        self,
        first_event: AgentEvent,
        context: RuntimeContext,
        control: RunControlState,
        *,
        trace_before_hooks: bool = False,
    ) -> tuple[AgentEvent, ...]:
        specs: list[tuple[AgentEvent, bool]] = [(first_event, True)]
        events: list[AgentEvent] = []
        pending_trace_events: list[AgentEvent] = []
        while specs:
            event, runtime_owned = specs.pop(0)
            emitter = EventEmitter()
            runtime_event = event
            recorded_before_hooks = False
            if runtime_owned and trace_before_hooks and control.trace is not None:
                control.trace.record_event(runtime_event)
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

    @staticmethod
    def _raw_event(
        control: RunControlState, event_type: str, data: Mapping[str, Any]
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

    @staticmethod
    def _raw_checkpoint_event(
        state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentEvent:
        sequence = control.next_sequence()
        event = AgentEvent(
            EventTypes.CHECKPOINT,
            AgentLoop._snapshot(state, context, control, sequence=sequence).to_dict(),
            run_id=control.run_id,
            sequence=sequence,
        )
        if control.trace is not None:
            control.trace.record_event(event)
        control.last_checkpoint = RunSnapshot.from_dict(event.data)
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
