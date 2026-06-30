"""Agent loop implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from inspect import isawaitable, iscoroutinefunction
from time import monotonic, time
from typing import Any, TypeVar, cast

from agent_runtime.errors import AgentError, ModelProviderError
from agent_runtime.events import AgentEvent, EventEmitter, EventTypes
from agent_runtime.hooks import RuntimeHook
from agent_runtime.limits import LoopLimits
from agent_runtime.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_parts_summary,
)
from agent_runtime.models import (
    ModelClient,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelStreamAccumulator,
    ResponseFormat,
    ToolChoice,
    stream_event_to_delta_payload,
)
from agent_runtime.runtime import RuntimeContext
from agent_runtime.scheduler import ToolBatch, ToolScheduler, ToolStarted
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import AgentState, AgentStatus
from agent_runtime.tools import Tool, ToolRegistry, ToolResult

T = TypeVar("T")


class RuntimeTimeoutError(Exception):
    """Raised only when the runtime-owned deadline expires."""


@dataclass(slots=True, frozen=True)
class AgentResult:
    """Final result returned by AgentLoop.run."""

    status: AgentStatus
    final_parts: tuple[ContentPart, ...]
    messages: tuple[Message, ...]
    iterations: int
    total_tool_calls: int
    error: str | None = None
    run_id: str = ""
    state: AgentState | None = None
    context: RuntimeContext | None = None
    snapshot: RunSnapshot | None = None


@dataclass(slots=True)
class RunControlState:
    """Runtime-owned control state that hooks cannot mutate."""

    run_id: str
    started_at: float
    deadline: float | None = None
    monotonic_deadline: float | None = None
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
        "_tool_choice",
        "_tool_scheduler",
        "_tools",
    )

    _hooks: tuple[RuntimeHook, ...]
    _limits: LoopLimits
    _model: ModelClient
    _model_options: ModelOptions
    _response_format: ResponseFormat | None
    _tool_choice: ToolChoice
    _tool_scheduler: ToolScheduler
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
    ) -> None:
        self._model = model
        self._tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self._limits = limits or LoopLimits()
        self._limits.validate()
        self._model_options = ModelOptions.from_dict((model_options or ModelOptions()).to_dict())
        self._tool_choice = ToolChoice.from_dict((tool_choice or ToolChoice()).to_dict())
        self._response_format = (
            None if response_format is None else ResponseFormat.from_dict(response_format.to_dict())
        )
        scheduler_parallelism = (
            1 if self._limits.stop_on_tool_error else (self._limits.max_parallel_tool_calls)
        )
        self._tool_scheduler = ToolScheduler(
            self._tools, max_parallel_tool_calls=scheduler_parallelism
        )
        self._hooks = tuple(hooks or ())

    async def run(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
    ) -> AgentResult:
        state = AgentState(status=AgentStatus.PLANNING, messages=list(messages))
        return await self.run_state(state, context=context, stream=stream)

    async def run_events(
        self,
        messages: Sequence[Message],
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        state = AgentState(status=AgentStatus.PLANNING, messages=list(messages))
        async for event in self.run_state_events(state, context=context, stream=stream):
            yield event

    async def run_state(
        self,
        state: AgentState,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
    ) -> AgentResult:
        runtime_context, control = self._prepare_run(context)
        working_state = AgentState.from_dict(state.to_dict())
        async for _event in self._run_state_events(
            working_state, runtime_context, control, stream=stream
        ):
            pass
        runtime_context.sequence = control.sequence
        return self._result(working_state, runtime_context, control)

    async def run_state_events(
        self,
        state: AgentState,
        *,
        context: RuntimeContext | None = None,
        stream: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        runtime_context, control = self._prepare_run(context)
        working_state = AgentState.from_dict(state.to_dict())
        async for event in self._run_state_events(
            working_state, runtime_context, control, stream=stream
        ):
            yield event
        runtime_context.sequence = control.sequence

    async def run_snapshot(self, snapshot: RunSnapshot, *, stream: bool = False) -> AgentResult:
        return await self.run_state(snapshot.state, context=snapshot.context, stream=stream)

    async def run_snapshot_events(
        self, snapshot: RunSnapshot, *, stream: bool = False
    ) -> AsyncIterator[AgentEvent]:
        async for event in self.run_state_events(
            snapshot.state, context=snapshot.context, stream=stream
        ):
            yield event

    async def _run_state_events(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        try:
            for event in await self._events(
                context,
                control,
                EventTypes.RUN_STARTED,
                {"state": state.snapshot()},
            ):
                yield event

            async for event in self._drive(state, context, control, stream=stream):
                yield event

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
            else:
                for event in await self._events(
                    context,
                    control,
                    EventTypes.ERROR,
                    {"status": state.status.value, "message": state.error or state.status.value},
                ):
                    yield event

            for event in await self._events(
                context, control, EventTypes.RUN_COMPLETED, {"state": state.snapshot()}
            ):
                yield event
        except RuntimeTimeoutError:
            previous = state.status
            state.status = AgentStatus.LIMIT_EXCEEDED
            state.error = "timeout_seconds"
            yield self._raw_event(
                control,
                EventTypes.STATE_CHANGED,
                {
                    "from": previous.value,
                    "to": state.status.value,
                    "iterations": state.iterations,
                    "total_tool_calls": state.total_tool_calls,
                    "error": state.error,
                },
            )
            yield self._raw_checkpoint_event(state, context, control)
            yield self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.snapshot()})
        except Exception as exc:  # pragma: no cover - defensive boundary
            previous = state.status
            state.status = AgentStatus.FAILED
            state.error = str(exc) or exc.__class__.__name__
            yield self._raw_event(
                control,
                EventTypes.STATE_CHANGED,
                {
                    "from": previous.value,
                    "to": state.status.value,
                    "iterations": state.iterations,
                    "total_tool_calls": state.total_tool_calls,
                    "error": state.error,
                },
            )
            yield self._raw_checkpoint_event(state, context, control)
            yield self._raw_event(
                control,
                EventTypes.ERROR,
                {"status": state.status.value, "message": state.error},
            )
            yield self._raw_event(control, EventTypes.RUN_COMPLETED, {"state": state.snapshot()})

    async def _drive(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        *,
        stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        while not state.is_terminal:
            reason = self._timeout_reason(control)
            if reason is not None:
                for event in await self._limit(state, context, control, reason):
                    yield event
                return

            if state.status is AgentStatus.PLANNING:
                async for event in self._planning_step(state, context, control, stream=stream):
                    yield event
                continue

            if state.status is AgentStatus.EXECUTING_TOOLS:
                async for event in self._tool_step(state, context, control):
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
        reason = self._limits.iteration_reason(state)
        if reason is not None:
            for event in await self._limit(state, context, control, reason):
                yield event
            return

        state.iterations += 1
        for event in await self._events(
            context,
            control,
            EventTypes.MODEL_STARTED,
            {"iteration": state.iterations},
        ):
            yield event

        request = await self._before_model(
            ModelRequest(
                messages=tuple(state.messages),
                tools=self._tools.specs(),
                options=self._model_options,
                tool_choice=self._tool_choice,
                response_format=self._response_format,
            ),
            context,
            control,
        )
        try:
            if stream and hasattr(self._model, "stream"):
                response_holder: list[ModelResponse] = []
                async for event in self._stream_model(request, context, control, response_holder):
                    yield event
                response = response_holder[0]
            else:
                response = await self._await_with_timeout(
                    self._model.complete(request, context),
                    control,
                )
            response = await self._after_model(
                ModelResponse.from_dict(response.to_dict()), context, control
            )
            assistant_message = response.to_assistant_message()
        except RuntimeTimeoutError:
            for event in await self._limit(state, context, control, "timeout_seconds"):
                yield event
            return
        except ModelProviderError as exc:
            state.extra = dict(state.extra) | {"model_error": exc.info.to_dict()}
            for event in await self._transition(
                state,
                AgentStatus.FAILED,
                context,
                control,
                error=exc.info.message,
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
        if response.tool_calls:
            state.pending_tool_calls = list(response.tool_calls)
            transition_data = await self._apply_transition(
                state,
                AgentStatus.EXECUTING_TOOLS,
                context,
                control,
            )
        else:
            state.final_parts = list(response.parts)
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
        for event in await self._events(
            context, control, EventTypes.STATE_CHANGED, transition_data
        ):
            yield event
        for event in await self._checkpoint_events(state, context, control):
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

        while True:
            try:
                stream_event = await self._anext_with_timeout(iterator, control)
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

        response_holder.append(completed_response or accumulator.response())

    async def _tool_step(
        self, state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AsyncIterator[AgentEvent]:
        while state.pending_tool_calls:
            reason = self._timeout_reason(control) or self._limits.tool_call_reason(state)
            if reason is not None:
                for event in await self._limit(state, context, control, reason):
                    yield event
                return

            remaining_slots = self._remaining_tool_call_slots(state)
            batch = self._tool_scheduler.next_batch(
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
                    error="before_tool cannot change tool call id or name",
                ):
                    yield event
                return

            try:
                async for event in self._run_tool_batch(prepared, state, context, control):
                    yield event
            except RuntimeTimeoutError:
                for event in await self._limit(state, context, control, "timeout_seconds"):
                    yield event
                return

            if state.is_terminal:
                return

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
            if call.id != raw_call.id or call.name != raw_call.name:
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
        completed: dict[int, tuple[ToolCall, ToolResult, Message]] = {}

        async def execute(call: ToolCall) -> ToolResult:
            return await self._await_with_timeout(self._execute_tool(call, context), control)

        async for progress in self._tool_scheduler.run_batch(
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
                        "arguments": dict(progress.call.arguments),
                        "batch_id": progress.batch.id,
                        "parallel": progress.batch.parallel,
                        "index": progress.index,
                    },
                ):
                    yield event
                continue

            result = await self._after_tool(progress.result, context, control)
            tool_message = result.to_message(progress.call)
            completed[progress.index] = (progress.call, result, tool_message)
            for event in await self._events(
                context,
                control,
                EventTypes.TOOL_COMPLETED,
                {
                    "id": progress.call.id,
                    "name": progress.call.name,
                    "batch_id": progress.batch.id,
                    "parallel": progress.batch.parallel,
                    "index": progress.index,
                    "result": result.summary(),
                },
            ):
                yield event

            if len(completed) < len(batch.calls):
                continue

            committed_results: list[ToolResult] = []
            for index in range(len(batch.calls)):
                _call, committed_result, message = completed[index]
                committed_results.append(committed_result)
                state.total_tool_calls += 1
                state.messages.append(message)
                state.pending_tool_calls.pop(0)

            completed.clear()
            for event in await self._checkpoint_events(state, context, control):
                yield event

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

    async def _execute_tool(self, call: ToolCall, context: RuntimeContext) -> ToolResult:
        try:
            return await self._tools.execute(call, context)
        except Exception as exc:
            return ToolResult(
                parts=[ContentPart.text_part(str(exc) or exc.__class__.__name__)],
                metadata={"error_type": exc.__class__.__name__},
                is_error=True,
            )

    def _remaining_tool_call_slots(self, state: AgentState) -> int:
        return max(0, self._limits.max_total_tool_calls - state.total_tool_calls)

    async def _await_with_timeout(self, awaitable: Awaitable[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(awaitable)
        if remaining is None:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise

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

    async def _anext_with_timeout(self, iterator: AsyncIterator[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(anext(iterator))
        if remaining is None:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise

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
            aclose = getattr(iterator, "aclose", None)
            if callable(aclose):
                with suppress(BaseException):
                    close_result = aclose()
                    if isawaitable(close_result):
                        await cast(Awaitable[object], close_result)
            raise RuntimeTimeoutError
        return await task

    def _prepare_run(
        self, context: RuntimeContext | None = None
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
        control = RunControlState(
            run_id=runtime_context.run_id,
            started_at=runtime_context.started_at,
            deadline=runtime_context.deadline,
            monotonic_deadline=None if remaining is None else now_monotonic + remaining,
            sequence=runtime_context.sequence,
        )
        return runtime_context, control

    def _timeout_reason(self, control: RunControlState) -> str | None:
        remaining = control.remaining_seconds()
        if remaining is not None and remaining <= 0:
            return "timeout_seconds"
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

    async def _transition(
        self,
        state: AgentState,
        status: AgentStatus,
        context: RuntimeContext,
        control: RunControlState,
        *,
        error: str | None = None,
        emit_checkpoint: bool = True,
    ) -> tuple[AgentEvent, ...]:
        transition_data = await self._apply_transition(state, status, context, control, error=error)
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
    ) -> dict[str, Any]:
        previous = state.status
        state.status = status
        state.error = error
        transition_state = AgentState.from_dict(state.to_dict())
        await self._notify_hooks(
            "on_transition", previous, status, transition_state, context, control=control
        )
        return {
            "from": previous.value,
            "to": status.value,
            "iterations": state.iterations,
            "total_tool_calls": state.total_tool_calls,
            "error": state.error,
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
        return await self._dispatch_events(event, context, control)

    async def _events(
        self,
        context: RuntimeContext,
        control: RunControlState,
        event_type: str,
        data: Mapping[str, Any],
    ) -> tuple[AgentEvent, ...]:
        event = AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )
        return await self._dispatch_events(event, context, control)

    async def _dispatch_events(
        self,
        first_event: AgentEvent,
        context: RuntimeContext,
        control: RunControlState,
    ) -> tuple[AgentEvent, ...]:
        specs: list[AgentEvent] = [first_event]
        events: list[AgentEvent] = []
        while specs:
            event = specs.pop(0)
            emitter = EventEmitter()
            for hook in self._hooks:
                replacement = await self._call_hook(
                    hook.on_event, event, context, emitter, control=control
                )
                if replacement is not None:
                    replacement_event = cast(AgentEvent, replacement)
                    event = AgentEvent(
                        replacement_event.type,
                        replacement_event.data,
                        run_id=event.run_id,
                        sequence=event.sequence,
                        created_at=event.created_at,
                        schema_version=event.schema_version,
                    )
            events.append(event)
            specs.extend(
                AgentEvent(
                    queued.type,
                    queued.data,
                    run_id=control.run_id,
                    sequence=control.next_sequence(),
                )
                for queued in emitter.drain()
            )
        return tuple(events)

    @staticmethod
    def _raw_event(
        control: RunControlState, event_type: str, data: Mapping[str, Any]
    ) -> AgentEvent:
        return AgentEvent(
            event_type,
            data,
            run_id=control.run_id,
            sequence=control.next_sequence(),
        )

    @staticmethod
    def _raw_checkpoint_event(
        state: AgentState, context: RuntimeContext, control: RunControlState
    ) -> AgentEvent:
        sequence = control.next_sequence()
        return AgentEvent(
            EventTypes.CHECKPOINT,
            AgentLoop._snapshot(state, context, control, sequence=sequence).to_dict(),
            run_id=control.run_id,
            sequence=sequence,
        )

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
        self, result: ToolResult, context: RuntimeContext, control: RunControlState
    ) -> ToolResult:
        current = result
        for hook in self._hooks:
            replacement = await self._call_hook(hook.after_tool, current, context, control=control)
            if replacement is not None:
                current = cast(ToolResult, replacement)
            current = ToolResult.from_dict(current.to_dict())
        return ToolResult.from_dict(current.to_dict())

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
        snapshot = AgentLoop._snapshot(state, context, control)
        return AgentResult(
            status=state.status,
            final_parts=tuple(state.final_parts),
            messages=tuple(state.messages),
            iterations=state.iterations,
            total_tool_calls=state.total_tool_calls,
            error=state.error,
            run_id=control.run_id,
            state=snapshot.state,
            context=snapshot.context,
            snapshot=snapshot,
        )
