from __future__ import annotations

import asyncio
import gc
import time as time_module
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from time import monotonic
from time import time as wall_time
from typing import Any, cast

import pytest
from diagnostics import RunTrace, TraceStepKinds, replay_trace
from harness import (
    AcceptingWebSearchTool,
    AdapterTimeoutModel,
    ApprovalPolicyByCall,
    CancellationConvertingModel,
    CancellationSwallowingModel,
    CancellationSwallowingThenFailingModel,
    CloseTrackingStreamingModel,
    ContextInspectingModel,
    CustomHandoffTool,
    ExternallyCancelledModel,
    FailingAcceptTool,
    FailingApprovalPolicy,
    FailingCheckpointJournal,
    FailingCustomHandoffTool,
    FailingFixtureTool,
    FailingRunStore,
    FailingSecondCheckpointStore,
    FastStreamingModel,
    FlakyProviderErrorModel,
    HarnessToolRegistry,
    MemoryRunJournal,
    MemoryRunStore,
    ProviderErrorModel,
    RecordingToolRegistry,
    RejectingWebSearchTool,
    RequestCapturingModel,
    ScriptedModel,
    SequencedApprovalPolicy,
    SlowModel,
    SlowRunJournal,
    SlowRunStore,
    SlowStreamingModel,
    StaticApprovalPolicy,
    StreamingProviderErrorModel,
    StreamingTextModel,
    StreamingToolModel,
    StreamingToolThenSlowModel,
    StrictCountFixtureTool,
    StrictCustomHandoffTool,
    TimelineRunJournal,
    collect_events,
    timeline_event_label,
)
from kernel import (
    AgentEvent,
    AgentLoop,
    AgentResult,
    AgentState,
    AgentStatus,
    ApprovalDecision,
    ApprovalRequest,
    BackgroundTask,
    CheckpointSummary,
    ContentPart,
    ConversationInsert,
    EventEmitter,
    EventTypes,
    JournalRecord,
    LimitExceeded,
    LimitReasons,
    LoopLimits,
    Message,
    ModelCapabilities,
    ModelErrorDecision,
    ModelErrorInfo,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PauseRequest,
    PauseSelector,
    PauseState,
    ResumeInput,
    RunController,
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    StoredCheckpoint,
    ToolBatch,
    ToolCall,
    ToolCatalog,
    ToolChoice,
    ToolCompleted,
    ToolObservation,
    ToolOutput,
    ToolScheduler,
    ToolSpec,
    ToolStarted,
)
from prompting import tool_text, user_text
from toolkit import ToolExecutionContext, ToolInvocation, ToolRegistry


def rt(result: AgentResult) -> RunTrace:
    assert result.trace is not None
    return RunTrace.from_dict(result.trace)


class ClearingReadRunController(RunController):
    def __init__(self, request: PauseRequest) -> None:
        super().__init__()
        self._request_copy = request
        self.reads = 0

    @property
    def pause_request(self) -> PauseRequest | None:
        self.reads += 1
        if self.reads == 1:
            return self._request_copy
        return None


class TimedTool:
    def __init__(self, name: str, *, parallel_safe: bool) -> None:
        annotations: dict[str, bool] = {}
        if parallel_safe:
            annotations = {
                "read_only": True,
                "idempotent": True,
                "parallel_safe": True,
            }
        self.spec = ToolSpec(
            name=name,
            description="Record timing and return the call id.",
            input_schema={"type": "object", "properties": {}},
            annotations=annotations,
        )
        self.active = 0
        self.max_active = 0
        self.timeline: list[tuple[str, str, float]] = []

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        call_id = str(invocation.arguments["id"])
        delay = float(invocation.arguments.get("delay", 0.01))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.timeline.append(("start", call_id, monotonic()))
        try:
            await asyncio.sleep(delay)
            return ToolObservation.text(call_id)
        finally:
            self.timeline.append(("end", call_id, monotonic()))
            self.active -= 1


class ContextMutatingParallelTool:
    spec = ToolSpec(
        name="context_mutator",
        description="Mutate context metadata while running in parallel.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    def __init__(self) -> None:
        self.writer_started = asyncio.Event()

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        call_id = str(invocation.arguments["id"])
        if call_id == "writer":
            context.metadata["leaked"] = "yes"
            self.writer_started.set()
            await asyncio.sleep(0.01)
            return ToolObservation.text("writer")
        await self.writer_started.wait()
        return ToolObservation.text(str(context.metadata.get("leaked", "missing")))


class MaybeErrorTimedTool(TimedTool):
    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        text = await super().execute(invocation, context)
        if invocation.arguments.get("error") is True:
            return ToolObservation.text(text.text_content or "tool_error", is_error=True)
        return text


class GappedTimeoutTool:
    spec = ToolSpec(
        name="gapped_timeout",
        description="Complete calls 0 and 2 together while call 1 times out.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    def __init__(self) -> None:
        self.started: list[str] = []
        self._all_started = asyncio.Event()

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        call_id = str(invocation.arguments["id"])
        self.started.append(call_id)
        if len(self.started) == 3:
            self._all_started.set()
        if call_id == "1":
            await asyncio.sleep(1)
        await self._all_started.wait()
        await asyncio.sleep(0)
        return ToolObservation.text(call_id)


class StaggeredGappedTimeoutTool:
    spec = ToolSpec(
        name="staggered_gapped_timeout",
        description="Complete call 0 first, call 2 second, while call 1 times out.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        call_id = str(invocation.arguments["id"])
        if call_id == "1":
            await asyncio.sleep(1)
        if call_id == "2":
            await asyncio.sleep(0.01)
        return ToolObservation.text(call_id)


class StructuralToolRegistry:
    def __init__(self, *tool_names: str) -> None:
        self._registry = HarnessToolRegistry(*tool_names)
        self.invocations = 0

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._registry.specs()

    def spec_for(self, name: str) -> ToolSpec | None:
        return self._registry.spec_for(name)

    def validate_call(self, call: ToolCall) -> None:
        self._registry.validate_call(call)

    async def invoke(
        self,
        call: ToolCall,
        context: object,
        *,
        progress_emitter: Any | None = None,
        cancel_checker: Any | None = None,
    ) -> ToolOutput:
        self.invocations += 1
        return await self._registry.invoke(
            call,
            cast(RuntimeContext, context),
            progress_emitter=progress_emitter,
            cancel_checker=cancel_checker,
        )


class NonCallableToolRegistry:
    specs = ()
    spec_for = None
    validate_call = None
    invoke = None


class AcceptWebSearchModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        mode="accept",
                        arguments={"query": "latest OpenAI Anthropic model comparison"},
                    )
                ]
            )
        assert request.messages[-1].role == "tool"
        assert request.messages[-1].metadata["result_kind"] == "acceptance"
        return ModelResponse.text("search accepted")


class RejectedAcceptModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        mode="accept",
                        arguments={"query": "offline index"},
                    )
                ]
            )
        assert request.messages[-1].role == "tool"
        assert request.messages[-1].metadata["result_kind"] == "rejection"
        assert request.messages[-1].metadata["is_error"] is True
        return ModelResponse.text(f"handled {request.messages[-1].text}")


class InsertAwareModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.first_started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_started.set()
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return ModelResponse.text("stale")
        assert request.messages[-1].role == "external"
        return ModelResponse.text(f"saw {request.messages[-1].text}")


class SameTickInsertModel:
    def __init__(self, controller: RunController) -> None:
        self.controller = controller
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.requests.append(request)
        if len(self.requests) == 1:
            self.controller.insert(
                ConversationInsert.text(
                    "same tick insert",
                    id="insert-race",
                    source="test",
                )
            )
            return ModelResponse.text("stale")
        assert request.messages[-1].role == "external"
        return ModelResponse.text(f"saw {request.messages[-1].text}")


class GateFinalModel:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.started.set()
        await self.release.wait()
        return ModelResponse.text("done")


class SelfPausingToolCallModel:
    def __init__(self, controller: RunController) -> None:
        self.controller = controller

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.controller.request_pause(reason="manual_pause")
        return ModelResponse(
            tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
        )


class StructuralAfterModelHook:
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = response, context
        return ModelResponse.text("structural hook")


class RetryModelErrorHook(RuntimeHook):
    def __init__(self, *, retry: bool = True, message: str | None = None) -> None:
        self.retry = retry
        self.message = message
        self.seen: list[ModelErrorInfo] = []
        self.before_model_calls = 0

    def before_model(self, request: ModelRequest, context: RuntimeContext) -> ModelRequest | None:
        _ = request, context
        self.before_model_calls += 1
        return None

    def on_model_error(
        self,
        error: ModelErrorInfo,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> ModelErrorDecision:
        _ = request, context
        self.seen.append(error)
        return ModelErrorDecision(retry=self.retry and error.retryable, message=self.message)


class RewritingHook(RuntimeHook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context
        if event.type == EventTypes.MODEL_STARTED:
            emitter.emit("custom_progress", {"phase": "model"})
        self.events.append(event.type)

    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = response, context
        return ModelResponse.text("hooked")


class RecordingEventHook(RuntimeHook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        self.events.append(event.type)


class RecordingFullEventHook(RuntimeHook):
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        self.events.append(AgentEvent(**event.to_dict()))


class RecordingVisibilityHook(RuntimeHook):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        self.seen.append(event.type)

    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> None:
        _ = state, context
        self.seen.append(f"transition:{previous.value}->{current.value}")


class TimelineVisibilityHook(RuntimeHook):
    def __init__(self, timeline: list[str]) -> None:
        self.timeline = timeline

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        self.timeline.append(timeline_event_label("hook", event))

    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> None:
        _ = state, context
        self.timeline.append(f"hook:transition:{previous.value}->{current.value}")


class ToolArgumentHook(RuntimeHook):
    def before_tool(self, call: ToolCall, context: RuntimeContext) -> ToolCall:
        _ = context
        return ToolCall(id=call.id, name=call.name, arguments={"text": "rewritten"})


class MutatingToolIdentityHook(RuntimeHook):
    def before_tool(self, call: ToolCall, context: RuntimeContext) -> ToolCall:
        _ = context
        call.id = "mutated-id"
        call.name = "missing"
        call.mode = "accept"
        return call


class ReplacingEventHook(RuntimeHook):
    def on_event(
        self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
    ) -> AgentEvent | None:
        _ = context, emitter
        if event.type == EventTypes.MODEL_STARTED:
            return AgentEvent("renamed_model_started", run_id="bad", sequence=999)
        return None


class CoreEventEmittingHook(RuntimeHook):
    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context
        if event.type == EventTypes.MODEL_STARTED:
            emitter.emit(EventTypes.MODEL_COMPLETED, {"summary": "host-emitted"})


class FailingQueuedEventAfterHook(RuntimeHook):
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context
        if event.type == self.event_type:
            emitter.emit("custom_after_core_event", {})
        if event.type == "custom_after_core_event":
            raise RuntimeError("custom event failed")


class MutatingEventContextHook(RuntimeHook):
    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = event, emitter
        context.run_id = "mutated-run"
        context.sequence = 1000
        context.deadline = None


class MutatingTransitionHook(RuntimeHook):
    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> None:
        _ = previous, current, context
        state.status = AgentStatus.FAILED
        state.error = "mutated"


class BadAfterModelHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        response.finish_reason = cast(Any, 123)
        return response


class BadAfterToolHook(RuntimeHook):
    def after_tool(self, result: ToolOutput, context: RuntimeContext) -> ToolObservation:
        _ = context
        observation = cast(ToolObservation, result)
        observation.is_error = cast(Any, "yes")
        return observation


class SuccessfulAfterToolHook(RuntimeHook):
    def after_tool(self, result: ToolOutput, context: RuntimeContext) -> ToolObservation:
        _ = result, context
        return ToolObservation.text("rewritten")


class PausingAfterToolHook(RuntimeHook):
    def after_tool(self, result: ToolOutput, context: RuntimeContext) -> ToolObservation:
        _ = result, context
        return ToolObservation(
            parts=[ContentPart.text_part("waiting")],
            is_error=True,
            pause=PauseRequest(reason="external_wait", source="tool", wait_id="job-1"),
        )


class SlowAfterModelHook(RuntimeHook):
    async def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        await asyncio.sleep(0.05)
        return response


class SlowTransitionHook(RuntimeHook):
    async def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> None:
        _ = previous, current, state, context
        await asyncio.sleep(0.05)


class SlowEventHook(RuntimeHook):
    async def on_event(
        self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
    ) -> None:
        _ = event, context, emitter
        await asyncio.sleep(0.05)


class SlowOnEventHook(RuntimeHook):
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type

    async def on_event(
        self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
    ) -> None:
        _ = context, emitter
        if event.type == self.event_type:
            await asyncio.sleep(0.05)


class RaisingEventHook(RuntimeHook):
    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = event, context, emitter
        raise RuntimeError("event hook failed")


class RaisingOnEventHook(RuntimeHook):
    def __init__(self, event_type: str) -> None:
        self.event_type = event_type

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        if event.type == self.event_type:
            raise RuntimeError(f"{self.event_type} hook failed")


class RequestPauseOnEventHook(RuntimeHook):
    def __init__(self, event_type: str, controller: RunController) -> None:
        self.event_type = event_type
        self.controller = controller

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        if event.type == self.event_type:
            self.controller.request_pause(reason="leftover")


class RequestPauseOnStateChangeHook(RuntimeHook):
    def __init__(
        self,
        from_status: AgentStatus,
        to_status: AgentStatus,
        controller: RunController,
    ) -> None:
        self.from_status = from_status
        self.to_status = to_status
        self.controller = controller

    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context, emitter
        if event.type != EventTypes.STATE_CHANGED:
            return
        if (
            event.data.get("from") == self.from_status.value
            and event.data.get("to") == self.to_status.value
        ):
            self.controller.request_pause(reason="leftover")


class BlockingAfterModelHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        time_module.sleep(0.05)
        return response


class StandaloneSerialScheduler:
    def __init__(self) -> None:
        self.batch_count = 0

    def next_batch(self, calls: tuple[ToolCall, ...]) -> ToolBatch | None:
        if not calls:
            return None
        self.batch_count += 1
        return ToolBatch(f"standalone-{self.batch_count}", (calls[0],), parallel=False)

    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        for index, call in enumerate(batch.calls):
            yield ToolStarted(batch=batch, index=index, call=call)
            result = await execute(call)
            yield ToolCompleted(batch=batch, index=index, call=call, result=result)
            if stop_on_error and result.is_error:
                return


class NonPrefixScheduler(StandaloneSerialScheduler):
    def next_batch(self, calls: tuple[ToolCall, ...]) -> ToolBatch | None:
        if len(calls) < 2:
            return super().next_batch(calls)
        self.batch_count += 1
        return ToolBatch(f"non-prefix-{self.batch_count}", (calls[1],), parallel=False)


class MutatingNextBatchScheduler(StandaloneSerialScheduler):
    def next_batch(self, calls: tuple[ToolCall, ...]) -> ToolBatch | None:
        if not calls:
            return None
        self.batch_count += 1
        cast(dict[str, Any], calls[0].arguments)["id"] = "mutated"
        return ToolBatch(f"mutating-next-{self.batch_count}", (calls[0],), parallel=False)


class WrongProgressCallScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = stop_on_error
        wrong_call = ToolCall(id="call-2", name="echo", arguments={"text": "second"})
        yield ToolStarted(batch=batch, index=0, call=wrong_call)
        result = await execute(wrong_call)
        yield ToolCompleted(batch=batch, index=0, call=wrong_call, result=result)


class FakeCompletionScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = execute, stop_on_error
        call = batch.calls[0]
        yield ToolStarted(batch=batch, index=0, call=call)
        yield ToolCompleted(
            batch=batch,
            index=0,
            call=call,
            result=ToolObservation.text("fake"),
        )


class ExecuteBeforeStartScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = stop_on_error
        call = batch.calls[0]
        result = await execute(call)
        yield ToolStarted(batch=batch, index=0, call=call)
        yield ToolCompleted(batch=batch, index=0, call=call, result=result)


class ConcurrentDuplicateExecuteScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = stop_on_error
        call = batch.calls[0]
        yield ToolStarted(batch=batch, index=0, call=call)
        first, second = await asyncio.gather(
            execute(call),
            execute(call),
            return_exceptions=True,
        )
        _ = second
        if isinstance(first, BaseException):
            raise first
        yield ToolCompleted(batch=batch, index=0, call=call, result=first)


class MutatingRunBatchScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = stop_on_error
        call = batch.calls[0]
        yield ToolStarted(batch=batch, index=0, call=call)
        cast(dict[str, Any], call.arguments)["id"] = "mutated"
        result = await execute(call)
        yield ToolCompleted(batch=batch, index=0, call=call, result=result)


class MutatingResultScheduler(StandaloneSerialScheduler):
    async def run_batch(
        self,
        batch: ToolBatch,
        execute: Callable[[ToolCall], Awaitable[ToolOutput]],
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolStarted | ToolCompleted]:
        _ = stop_on_error
        call = batch.calls[0]
        yield ToolStarted(batch=batch, index=0, call=call)
        result = await execute(call)
        if result.parts:
            result.parts[0].text = "mutated"
        cast(dict[str, Any], result.metadata)["scheduler"] = "mutated"
        yield ToolCompleted(batch=batch, index=0, call=call, result=result)


class BadModelResponseShapeHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        response.response_id = cast(Any, 123)
        return response


class BadToolObservationShapeHook(RuntimeHook):
    def after_tool(self, result: ToolOutput, context: RuntimeContext) -> ToolObservation:
        _ = context
        observation = cast(ToolObservation, result)
        observation.is_error = cast(Any, "yes")
        return observation


def parts_text(parts: Sequence[Any]) -> str:
    return "".join(part.text or "" for part in parts)


@pytest.mark.asyncio
async def test_model_request_includes_standard_options_and_tool_choice() -> None:
    model = RequestCapturingModel()

    result = await AgentLoop(
        model=model,
        model_options=ModelOptions(model="test-model", temperature=0.1),
        tool_choice=ToolChoice(mode="none", allow_parallel_tool_calls=False),
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert model.request is not None
    assert model.request.options.model == "test-model"
    assert model.request.options.temperature == 0.1
    assert model.request.tool_choice.mode == "none"
    assert model.request.tool_choice.allow_parallel_tool_calls is False


@pytest.mark.asyncio
async def test_streaming_text_deltas_are_observable_but_commit_atomically() -> None:
    events = await collect_events(
        AgentLoop(model=StreamingTextModel()),
        [user_text("stream")],
        stream=True,
    )
    event_types = [event.type for event in events]
    deltas = [event.data for event in events if event.type == EventTypes.MODEL_DELTA]
    final_snapshot = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )

    assert [delta["text_delta"] for delta in deltas] == ["hel", "lo"]
    assert event_types.index(EventTypes.MODEL_DELTA) < event_types.index(EventTypes.MODEL_COMPLETED)
    assert final_snapshot.state.status is AgentStatus.COMPLETED
    assert final_snapshot.state.messages[-1].text == "hello"


@pytest.mark.asyncio
async def test_streaming_tool_call_executes_only_after_model_completed() -> None:
    events = await collect_events(
        AgentLoop(model=StreamingToolModel(), tools=HarnessToolRegistry("echo")),
        [user_text("stream tool")],
        stream=True,
    )
    event_types = [event.type for event in events]
    tool_started_index = event_types.index(EventTypes.TOOL_STARTED)
    first_model_completed_index = event_types.index(EventTypes.MODEL_COMPLETED)
    result = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )

    assert first_model_completed_index < tool_started_index
    assert result.state.status is AgentStatus.COMPLETED
    assert [message.text for message in result.state.messages if message.role == "tool"] == [
        "hello"
    ]
    assert result.state.final_parts[0].text == "done"


@pytest.mark.asyncio
async def test_agent_loop_accepts_structural_tool_registry() -> None:
    registry = StructuralToolRegistry("echo")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="echo",
                        arguments={"text": "structural registry"},
                    )
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    result = await AgentLoop(model=model, tools=registry).run([user_text("echo")])

    assert result.status is AgentStatus.COMPLETED
    assert registry.invocations == 1
    assert [message.text for message in result.messages if message.role == "tool"] == [
        "structural registry"
    ]


def test_agent_loop_rejects_non_callable_tool_registry_members() -> None:
    with pytest.raises(TypeError, match="ToolRegistryProtocol"):
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            tools=cast(Any, NonCallableToolRegistry()),
        )


@pytest.mark.asyncio
async def test_stream_timeout_discards_partial_assistant_message() -> None:
    events = await collect_events(
        AgentLoop(
            model=SlowStreamingModel(),
            limits=LoopLimits(timeout_seconds=0.02),
        ),
        [user_text("stream slow")],
        stream=True,
    )
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]

    assert any(event.type == EventTypes.MODEL_DELTA for event in events)
    assert checkpoints[-1].state.status is AgentStatus.LIMIT_EXCEEDED
    assert checkpoints[-1].state.error == "timeout_seconds"
    assert [message.role for message in checkpoints[-1].state.messages] == ["user"]


@pytest.mark.asyncio
async def test_stream_interrupt_pauses_without_partial_assistant_message() -> None:
    controller = RunController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(model=SlowStreamingModel()).run_events(
        [user_text("stream slow")],
        stream=True,
        controller=controller,
    ):
        events.append(event)
        if event.type == EventTypes.MODEL_DELTA:
            controller.interrupt(reason="user_interrupted")

    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    paused = checkpoints[-1]

    assert EventTypes.RUN_PAUSED in [event.type for event in events]
    assert EventTypes.ERROR not in [event.type for event in events]
    assert paused.state.status is AgentStatus.PAUSED
    assert paused.state.pause is not None
    assert paused.state.pause.reason == "user_interrupted"
    assert paused.state.pause.resume_status is AgentStatus.PLANNING
    assert [message.role for message in paused.state.messages] == ["user"]

    resumed_model = ScriptedModel([ModelResponse.text("fresh answer")])
    result = await AgentLoop(model=resumed_model).run_snapshot(ResumeInput(snapshot=paused))

    assert result.status is AgentStatus.COMPLETED
    assert resumed_model.calls == 1
    assert parts_text(result.final_parts) == "fresh answer"


@pytest.mark.asyncio
async def test_stream_interrupt_does_not_consume_durable_iteration() -> None:
    controller = RunController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(
        model=SlowStreamingModel(),
        limits=LoopLimits(max_iterations=1),
    ).run_events(
        [user_text("stream slow")],
        stream=True,
        controller=controller,
    ):
        events.append(event)
        if event.type == EventTypes.MODEL_DELTA:
            controller.interrupt(reason="user_interrupted")

    paused = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )
    resumed = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("fresh answer")]),
        limits=LoopLimits(max_iterations=1),
    ).run_snapshot(ResumeInput(snapshot=paused))

    assert paused.state.status is AgentStatus.PAUSED
    assert paused.state.iterations == 0
    assert resumed.status is AgentStatus.COMPLETED
    assert parts_text(resumed.final_parts) == "fresh answer"


@pytest.mark.asyncio
async def test_stream_interrupt_wins_over_immediately_ready_next_delta() -> None:
    controller = RunController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(model=FastStreamingModel()).run_events(
        [user_text("stream fast")],
        stream=True,
        controller=controller,
    ):
        events.append(event)
        if event.type == EventTypes.MODEL_DELTA:
            controller.interrupt(reason="user_interrupted")

    event_types = [event.type for event in events]
    paused = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )

    assert event_types.count(EventTypes.MODEL_DELTA) == 1
    assert EventTypes.MODEL_COMPLETED not in event_types
    assert paused.state.status is AgentStatus.PAUSED
    assert paused.state.messages == [user_text("stream fast")]


@pytest.mark.asyncio
async def test_later_stream_timeout_trace_uses_last_checkpoint_as_partial_baseline() -> None:
    result = await AgentLoop(
        model=StreamingToolThenSlowModel(),
        tools=HarnessToolRegistry("echo"),
        limits=LoopLimits(timeout_seconds=0.02),
    ).run(
        [user_text("stream after tool")],
        stream=True,
    )

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert [message.role for message in result.messages] == ["user", "assistant", "tool"]
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_external_cancellation_closes_stream_iterator() -> None:
    model = CloseTrackingStreamingModel()
    events: list[AgentEvent] = []

    async def collect() -> None:
        async for event in AgentLoop(model=model).run_events(
            [user_text("stream")],
            stream=True,
        ):
            events.append(event)

    task = asyncio.create_task(collect())
    await model.next_chunk_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_DELTA,
    ]
    assert model.closed.is_set()


@pytest.mark.asyncio
async def test_early_event_consumer_close_closes_stream_iterator() -> None:
    model = CloseTrackingStreamingModel()
    iterator = AgentLoop(model=model).run_events(
        [user_text("stream")],
        stream=True,
    )

    async for event in iterator:
        if event.type == EventTypes.MODEL_DELTA:
            break
    await cast(Any, iterator).aclose()

    assert model.closed.is_set()


@pytest.mark.asyncio
async def test_stream_flag_falls_back_to_complete_for_non_streaming_model() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")])),
        [user_text("stream flag")],
        stream=True,
    )

    assert EventTypes.MODEL_DELTA not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_stream_flag_uses_complete_when_capability_does_not_advertise_streaming() -> None:
    class CompletePreferredModel(StreamingTextModel):
        capabilities = ModelCapabilities(streaming=False)

        async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
            _ = request, context
            return ModelResponse.text("complete path")

    result = await AgentLoop(model=CompletePreferredModel()).run(
        [user_text("stream flag")],
        stream=True,
    )

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "complete path"


@pytest.mark.asyncio
async def test_model_provider_error_sets_failed_checkpoint_error() -> None:
    result = await AgentLoop(model=ProviderErrorModel()).run([user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "provider unavailable"
    assert result.snapshot is not None
    assert result.snapshot.state.error == "provider unavailable"
    assert "error_details" not in result.snapshot.state.to_dict()


@pytest.mark.asyncio
async def test_model_provider_error_hook_can_retry_with_limit() -> None:
    model = FlakyProviderErrorModel()
    hook = RetryModelErrorHook()

    events = await collect_events(
        AgentLoop(
            model=model,
            hooks=[hook],
            limits=LoopLimits(max_model_retries=1),
        ),
        [user_text("retry")],
    )
    event_types = [event.type for event in events]
    event_trace = RunTrace.from_events(events[0].run_id, events)

    assert events[-1].data["state"]["status"] == AgentStatus.COMPLETED.value
    assert event_types.count(EventTypes.MODEL_STARTED) == 2
    assert event_types.count(EventTypes.MODEL_ERROR) == 1
    assert event_types.index(EventTypes.MODEL_ERROR) < event_types.index(EventTypes.MODEL_COMPLETED)
    model_error_event = next(event for event in events if event.type == EventTypes.MODEL_ERROR)
    assert model_error_event.data["retry"] is True
    assert model_error_event.data["error"]["message"] == "provider unavailable"
    assert [step.kind for step in event_trace.steps].count(TraceStepKinds.MODEL_CALL) == 2
    assert TraceStepKinds.MODEL_ERROR in [step.kind for step in event_trace.steps]
    assert replay_trace(event_trace).final_status is AgentStatus.COMPLETED
    assert model.calls == 2
    assert hook.before_model_calls == 2
    assert [error.message for error in hook.seen] == ["provider unavailable"]


@pytest.mark.asyncio
async def test_model_provider_error_hook_can_rewrite_final_message() -> None:
    result = await AgentLoop(
        model=ProviderErrorModel(),
        hooks=[RetryModelErrorHook(retry=False, message="temporary model outage")],
        limits=LoopLimits(max_model_retries=1),
    ).run([user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "temporary model outage"


@pytest.mark.asyncio
async def test_stream_provider_error_discards_partial_assistant_message() -> None:
    events = await collect_events(
        AgentLoop(model=StreamingProviderErrorModel()),
        [user_text("stream fail")],
        stream=True,
    )
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    event_types = [event.type for event in events]

    assert EventTypes.MODEL_DELTA in event_types
    assert EventTypes.MODEL_COMPLETED not in event_types
    assert checkpoints[-1].state.status is AgentStatus.FAILED
    assert checkpoints[-1].state.error == "provider unavailable"
    assert [message.role for message in checkpoints[-1].state.messages] == ["user"]
    assert "error_details" not in checkpoints[-1].state.to_dict()


@pytest.mark.asyncio
async def test_stream_provider_error_is_not_retried_after_delta() -> None:
    events = await collect_events(
        AgentLoop(
            model=StreamingProviderErrorModel(),
            hooks=[RetryModelErrorHook()],
            limits=LoopLimits(max_model_retries=1),
        ),
        [user_text("stream fail")],
        stream=True,
    )
    event_types = [event.type for event in events]
    model_error_event = next(event for event in events if event.type == EventTypes.MODEL_ERROR)

    assert event_types.count(EventTypes.MODEL_STARTED) == 1
    assert event_types.count(EventTypes.MODEL_ERROR) == 1
    assert model_error_event.data["retry"] is False
    assert EventTypes.MODEL_COMPLETED not in event_types
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_model_direct_final() -> None:
    model = ScriptedModel([ModelResponse.text("done")])
    result = await AgentLoop(model=model).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"
    assert result.iterations == 1
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_trace_can_be_disabled() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        trace=False,
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert result.trace is None


@pytest.mark.asyncio
async def test_custom_tool_scheduler_factory_is_used_per_run() -> None:
    calls: list[tuple[ToolCatalog, LoopLimits]] = []
    limits = LoopLimits(max_parallel_tool_calls=3)

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> ToolScheduler:
        calls.append((tools, runtime_limits))
        assert not hasattr(tools, "invoke")
        assert not hasattr(tools, "validate_call")
        return ToolScheduler(tools, max_parallel_tool_calls=1)

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={})]),
                ModelResponse.text("done"),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        limits=limits,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    assert len(calls) == 1
    assert calls[0][1] is limits
    assert calls[0][0].spec_for("echo") is not None


@pytest.mark.asyncio
async def test_custom_tool_scheduler_receives_defensive_tool_catalog() -> None:
    catalogs: list[ToolCatalog] = []

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> ToolScheduler:
        _ = runtime_limits
        catalogs.append(tools)
        return ToolScheduler(tools)

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={})]),
                ModelResponse.text("done"),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    catalog = catalogs[0]
    specs = catalog.specs()
    cast(dict[str, Any], specs[0].input_schema)["type"] = "string"
    spec = catalog.spec_for("echo")
    assert spec is not None
    cast(dict[str, Any], spec.annotations)["parallel_safe"] = True

    fresh_spec = catalog.spec_for("echo")
    assert fresh_spec is not None
    assert fresh_spec.input_schema["type"] == "object"
    assert fresh_spec.annotations == {}
    assert catalog.specs()[0].input_schema["type"] == "object"


@pytest.mark.asyncio
async def test_custom_tool_scheduler_factory_accepts_protocol_implementations() -> None:
    schedulers: list[StandaloneSerialScheduler] = []

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> StandaloneSerialScheduler:
        _ = tools, runtime_limits
        scheduler = StandaloneSerialScheduler()
        schedulers.append(scheduler)
        return scheduler

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={})]),
                ModelResponse.text("done"),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    assert len(schedulers) == 1
    assert schedulers[0].batch_count == 1


@pytest.mark.asyncio
async def test_custom_tool_scheduler_must_return_non_empty_prefix_batch() -> None:
    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> NonPrefixScheduler:
        _ = tools, runtime_limits
        return NonPrefixScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="call-1", name="echo", arguments={"text": "first"}),
                        ToolCall(id="call-2", name="echo", arguments={"text": "second"}),
                    ]
                )
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "tool scheduler must return a non-empty prefix batch"
    assert result.total_tool_calls == 0
    assert result.snapshot is not None
    assert [call.id for call in result.snapshot.state.pending_tool_calls] == [
        "call-1",
        "call-2",
    ]


@pytest.mark.asyncio
async def test_custom_tool_scheduler_cannot_mutate_pending_calls_in_next_batch() -> None:
    tool = RecordingToolRegistry()

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> MutatingNextBatchScheduler:
        _ = tools, runtime_limits
        return MutatingNextBatchScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "original"})]
                )
            ]
        ),
        tools=tool,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "tool scheduler must return a non-empty prefix batch"
    assert tool.calls == []
    assert result.snapshot is not None
    assert result.snapshot.state.pending_tool_calls[0].arguments["id"] == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        (WrongProgressCallScheduler(), "progress call does not match batch index"),
        (FakeCompletionScheduler(), "complete results produced by execute"),
    ],
)
async def test_custom_tool_scheduler_progress_is_validated(
    scheduler: StandaloneSerialScheduler,
    message: str,
) -> None:
    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> StandaloneSerialScheduler:
        _ = tools, runtime_limits
        return scheduler

    result = await AgentLoop(
        model=ScriptedModel(
            [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={})])]
        ),
        tools=HarnessToolRegistry("echo"),
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert message in result.error
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scheduler", "message", "expected_calls"),
    [
        (
            ExecuteBeforeStartScheduler(),
            "start a batch call before executing it",
            [],
        ),
        (
            ConcurrentDuplicateExecuteScheduler(),
            "execute each batch call at most once",
            ["job"],
        ),
    ],
)
async def test_custom_tool_scheduler_execute_gate_prevents_lifecycle_violations(
    scheduler: StandaloneSerialScheduler,
    message: str,
    expected_calls: list[str],
) -> None:
    tool = RecordingToolRegistry()

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> StandaloneSerialScheduler:
        _ = tools, runtime_limits
        return scheduler

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "job"})]
                )
            ]
        ),
        tools=tool,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert message in result.error
    assert tool.calls == expected_calls
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_custom_tool_scheduler_cannot_mutate_call_after_start() -> None:
    tool = RecordingToolRegistry()

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> MutatingRunBatchScheduler:
        _ = tools, runtime_limits
        return MutatingRunBatchScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "original"})]
                )
            ]
        ),
        tools=tool,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "outside the selected batch" in result.error
    assert tool.calls == []
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_custom_tool_scheduler_cannot_mutate_execute_result() -> None:
    tool = RecordingToolRegistry()

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> MutatingResultScheduler:
        _ = tools, runtime_limits
        return MutatingResultScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "original"})]
                )
            ]
        ),
        tools=tool,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "must not replace execute results" in result.error
    assert tool.calls == ["original"]
    assert result.total_tool_calls == 0
    assert all(message.role != "tool" for message in result.messages)


@pytest.mark.asyncio
async def test_custom_tool_scheduler_cannot_mutate_approval_allowed_result() -> None:
    tool = RecordingToolRegistry()
    policy = StaticApprovalPolicy(ApprovalDecision.allow("safe"))

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> MutatingResultScheduler:
        _ = tools, runtime_limits
        return MutatingResultScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "allowed"})]
                )
            ]
        ),
        tools=tool,
        approval_policy=policy,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "must not replace execute results" in result.error
    assert tool.calls == ["allowed"]
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_custom_tool_scheduler_cannot_mutate_approval_denied_result() -> None:
    tool = RecordingToolRegistry()
    policy = StaticApprovalPolicy(ApprovalDecision.deny("blocked"))

    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> MutatingResultScheduler:
        _ = tools, runtime_limits
        return MutatingResultScheduler()

    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "denied"})]
                )
            ]
        ),
        tools=tool,
        approval_policy=policy,
        tool_scheduler_factory=factory,
    ).run([user_text("tool")])

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "must not replace execute results" in result.error
    assert tool.calls == []
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_custom_tool_scheduler_factory_must_return_scheduler() -> None:
    def factory(tools: ToolCatalog, runtime_limits: LoopLimits) -> Any:
        _ = tools, runtime_limits
        return object()

    with pytest.raises(TypeError, match="ToolSchedulerProtocol"):
        await AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            tool_scheduler_factory=factory,
        ).run([user_text("finish")])


@pytest.mark.asyncio
async def test_model_usage_is_aggregated_into_result_and_snapshot() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={})],
                    usage=ModelUsage(
                        input_tokens=2,
                        output_tokens=3,
                        total_tokens=5,
                        metadata={"provider_cost": "raw"},
                    ),
                ),
                ModelResponse(
                    parts=[ContentPart.text_part("done")],
                    usage=ModelUsage(
                        input_tokens=4,
                        output_tokens=5,
                        total_tokens=9,
                        metadata={"provider_cost": "raw"},
                    ),
                ),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    assert result.total_usage == ModelUsage(input_tokens=6, output_tokens=8, total_tokens=14)
    assert result.snapshot is not None
    assert result.snapshot.state.total_usage == result.total_usage
    assert result.snapshot.state.summary()["total_usage"] == {
        "input_tokens": 6,
        "output_tokens": 8,
        "total_tokens": 14,
    }


@pytest.mark.asyncio
async def test_model_usage_missing_fields_preserve_known_cumulative_values() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={})],
                    usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
                ),
                ModelResponse(
                    parts=[ContentPart.text_part("done")],
                    usage=ModelUsage(output_tokens=4),
                ),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        limits=LoopLimits(max_total_tokens=10),
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    assert result.total_usage == ModelUsage(input_tokens=2, output_tokens=7, total_tokens=5)
    assert result.snapshot is not None
    assert result.snapshot.state.summary()["total_usage"] == {
        "input_tokens": 2,
        "output_tokens": 7,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
async def test_model_usage_none_preserves_known_cumulative_values() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={})],
                    usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
                ),
                ModelResponse(parts=[ContentPart.text_part("done")]),
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        limits=LoopLimits(max_total_tokens=10),
    ).run([user_text("tool")])

    assert result.status is AgentStatus.COMPLETED
    assert result.total_usage == ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5)


@pytest.mark.asyncio
async def test_model_usage_limit_stops_after_final_model_response() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    parts=[ContentPart.text_part("done")],
                    usage=ModelUsage(total_tokens=11),
                )
            ]
        ),
        limits=LoopLimits(max_total_tokens=10),
    ).run([user_text("finish")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_total_tokens"
    assert result.final_parts == ()
    assert result.total_usage == ModelUsage(total_tokens=11)


@pytest.mark.asyncio
async def test_model_response_metadata_is_not_persisted_in_checkpoint_state() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    parts=[ContentPart.text_part("done", metadata={"secret": "part"})],
                    metadata={"secret": "response"},
                )
            ]
        )
    ).run([user_text("finish")])

    assert result.snapshot is not None
    assert result.final_parts[0].metadata == {}
    assert result.snapshot.state.final_parts[0].metadata == {}
    assert result.snapshot.state.messages[-1].parts[0].metadata == {}
    assert "secret" not in result.snapshot.state.messages[-1].metadata


@pytest.mark.asyncio
async def test_tool_result_metadata_is_not_persisted_in_checkpoint_message() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="metadata_tool", arguments={})]),
            ModelResponse.text("done"),
        ]
    )
    result = await AgentLoop(model=model, tools=HarnessToolRegistry("metadata_tool")).run(
        [user_text("tool")]
    )

    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.parts[0].metadata == {}
    assert tool_message.metadata == {"result_kind": "observation"}


@pytest.mark.asyncio
async def test_accept_tool_call_commits_acceptance_without_execute_wrapper() -> None:
    model = AcceptWebSearchModel()
    tool = AcceptingWebSearchTool()

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run([user_text("search")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "search accepted"
    assert result.total_tool_calls == 1
    assert len(tool.accepted) == 1
    assert tool.accepted[0].mode == "accept"
    assert tool.accepted[0].name == "web_search"
    assert tool.accepted[0].arguments == {"query": "latest OpenAI Anthropic model comparison"}

    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "accepted: latest OpenAI Anthropic model comparison"
    assert tool_message.metadata == {
        "result_kind": "acceptance",
        "correlation_id": "web-search:call-1",
    }

    assert result.trace is not None
    tool_call_step = next(
        step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_CALL
    )
    tool_result_step = next(
        step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_RESULT
    )
    assert tool_call_step.payload["mode"] == "accept"
    assert tool_result_step.payload["mode"] == "accept"
    assert tool_result_step.payload["result"]["result_kind"] == "acceptance"
    assert tool_result_step.payload["result"]["correlation_id"] == "web-search:call-1"


@pytest.mark.asyncio
async def test_accept_tool_call_commits_rejection_and_recovers() -> None:
    result = await AgentLoop(
        model=RejectedAcceptModel(),
        tools=ToolRegistry([RejectingWebSearchTool()]),
    ).run([user_text("search")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled rejected: offline index"
    assert result.total_tool_calls == 1

    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "rejected: offline index"
    assert tool_message.metadata == {
        "result_kind": "rejection",
        "is_error": True,
    }

    assert result.trace is not None
    tool_result_step = next(
        step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_RESULT
    )
    assert tool_result_step.payload["mode"] == "accept"
    assert tool_result_step.payload["result"]["result_kind"] == "rejection"
    assert tool_result_step.payload["result"]["is_error"] is True
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_accept_tool_exception_is_committed_as_rejection() -> None:
    result = await AgentLoop(
        model=RejectedAcceptModel(),
        tools=ToolRegistry([FailingAcceptTool()]),
    ).run([user_text("search")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled accept unavailable"
    assert result.total_tool_calls == 1
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "accept unavailable"
    assert tool_message.metadata == {
        "result_kind": "rejection",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_unknown_accept_tool_call_is_committed_as_rejection() -> None:
    result = await AgentLoop(model=RejectedAcceptModel()).run([user_text("search")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled unknown tool: web_search"
    assert result.total_tool_calls == 1
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "unknown tool: web_search"
    assert tool_message.metadata == {
        "result_kind": "rejection",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_accept_tool_input_schema_error_is_committed_as_rejection() -> None:
    tool = AcceptingWebSearchTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        mode="accept",
                        arguments={"query": 123},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run([user_text("search")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled"
    assert tool.accepted == []
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "input_schema" in tool_message.text
    assert tool_message.metadata == {
        "result_kind": "rejection",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_accept_tool_rejection_honors_stop_on_tool_error() -> None:
    result = await AgentLoop(
        model=RejectedAcceptModel(),
        tools=ToolRegistry([FailingAcceptTool()]),
        limits=LoopLimits(stop_on_tool_error=True),
    ).run([user_text("search")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "accept unavailable"
    assert result.total_tool_calls == 1
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.metadata["result_kind"] == "rejection"


@pytest.mark.asyncio
async def test_custom_tool_output_pause_is_applied_and_replayable() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="handoff",
                        mode="handoff",
                        arguments={"text": "handoff started", "wait_id": "job-1"},
                    )
                ]
            )
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([CustomHandoffTool()])).run(
        [user_text("start handoff")]
    )

    assert result.status is AgentStatus.PAUSED
    assert result.total_tool_calls == 1
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.wait_id == "job-1"
    assert result.snapshot.state.pause.resume_status is AgentStatus.PLANNING
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.PAUSED
    tool_result_step = next(
        step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_RESULT
    )
    assert tool_result_step.payload["mode"] == "handoff"
    assert tool_result_step.payload["result"]["result_kind"] == "handoff"
    assert tool_result_step.payload["result"]["pause"]["wait_id"] == "job-1"


@pytest.mark.asyncio
async def test_custom_tool_output_error_honors_stop_on_tool_error() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="handoff",
                        mode="handoff",
                        arguments={"text": "handoff failed", "is_error": True},
                    )
                ]
            )
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([CustomHandoffTool()]),
        limits=LoopLimits(stop_on_tool_error=True),
    ).run([user_text("start handoff")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "handoff failed"
    assert result.total_tool_calls == 1


@pytest.mark.asyncio
async def test_custom_tool_exception_commits_extension_tool_error() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="handoff",
                        mode="handoff",
                        arguments={},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([FailingCustomHandoffTool()])).run(
        [user_text("start handoff")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.total_tool_calls == 1
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "handoff unavailable"
    assert tool_message.metadata == {
        "result_kind": "tool_error",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_unknown_custom_tool_call_commits_extension_tool_error() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="missing",
                        mode="handoff",
                        arguments={},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model).run([user_text("start handoff")])

    assert result.status is AgentStatus.COMPLETED
    assert result.total_tool_calls == 1
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.text == "unknown tool: missing"
    assert tool_message.metadata == {
        "result_kind": "tool_error",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_custom_tool_input_schema_error_commits_extension_tool_error() -> None:
    tool = StrictCustomHandoffTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="strict_handoff",
                        mode="handoff",
                        arguments={"target": 123},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run(
        [user_text("start handoff")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.total_tool_calls == 1
    assert tool.calls == 0
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "input_schema" in tool_message.text
    assert tool_message.metadata == {
        "result_kind": "tool_error",
        "is_error": True,
    }


@pytest.mark.asyncio
async def test_custom_tool_output_rejects_reserved_result_kind() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="handoff",
                        mode="handoff",
                        arguments={"kind": "observation", "text": "invalid"},
                    )
                ]
            )
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([CustomHandoffTool()])).run(
        [user_text("start handoff")]
    )

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "extension ToolOutput kind" in result.error
    assert result.total_tool_calls == 0
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert all(message.role != "tool" for message in result.messages)


@pytest.mark.asyncio
async def test_conversation_insert_interrupts_model_and_replans_with_external_message() -> None:
    model = InsertAwareModel()
    controller = RunController()
    task = asyncio.create_task(
        AgentLoop(model=model).run([user_text("wait for search")], controller=controller)
    )

    await asyncio.wait_for(model.first_started.wait(), timeout=1)
    controller.insert(
        ConversationInsert.text(
            "web search finished",
            id="insert-1",
            source="web_search",
            correlation_id="web-search:call-1",
        )
    )
    result = await asyncio.wait_for(task, timeout=1)

    assert model.cancelled.is_set()
    assert len(model.requests) == 2
    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "saw web search finished"
    external_message = next(message for message in result.messages if message.role == "external")
    assert external_message.text == "web search finished"
    assert external_message.metadata == {
        "insert_id": "insert-1",
        "source": "web_search",
        "correlation_id": "web-search:call-1",
    }
    assert model.requests[-1].messages[-1].role == "external"

    assert result.trace is not None
    insert_step = next(
        step for step in rt(result).steps if step.kind == TraceStepKinds.CONVERSATION_INSERT
    )
    assert insert_step.payload["id"] == "insert-1"
    assert insert_step.payload["source"] == "web_search"
    assert insert_step.payload["correlation_id"] == "web-search:call-1"


@pytest.mark.asyncio
async def test_conversation_insert_wins_when_model_completes_in_same_wait_cycle() -> None:
    controller = RunController()
    model = SameTickInsertModel(controller)

    result = await AgentLoop(model=model).run(
        [user_text("race")],
        controller=controller,
    )

    assert len(model.requests) == 2
    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "saw same tick insert"
    assert [message.role for message in result.messages] == [
        "user",
        "external",
        "assistant",
    ]
    assert "stale" not in [message.text for message in result.messages]


@pytest.mark.asyncio
async def test_result_snapshot_is_last_durable_checkpoint() -> None:
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run(
        [user_text("finish")]
    )

    assert result.snapshot is not None
    assert result.trace is not None
    checkpoint_step = next(
        step for step in reversed(rt(result).steps) if step.kind == TraceStepKinds.CHECKPOINT
    )
    completed_step = rt(result).steps[-1]
    assert result.snapshot.context.sequence == checkpoint_step.references["event_sequence"]
    assert result.snapshot.context.sequence < completed_step.references["event_sequence"]


@pytest.mark.asyncio
async def test_one_tool_then_final() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
            ),
            ModelResponse.text("hello"),
        ]
    )
    result = await AgentLoop(model=model, tools=HarnessToolRegistry("echo")).run(
        [user_text("echo")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "hello"
    assert result.iterations == 2
    assert result.total_tool_calls == 1
    assert result.messages[-2].role == "tool"
    assert result.messages[-2].text == "hello"


@pytest.mark.asyncio
async def test_controller_pauses_before_model_call_and_snapshot_resumes() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    model = ScriptedModel([ModelResponse.text("done")])

    result = await AgentLoop(model=model).run(
        [user_text("finish")],
        controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert model.calls == 0
    assert controller.pause_request is None
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.reason == "manual_pause"
    assert result.snapshot.state.pause.resume_status is AgentStatus.PLANNING

    resumed = await AgentLoop(model=model).run_snapshot(
        ResumeInput(snapshot=result.snapshot or raise_assertion())
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert model.calls == 1
    assert parts_text(resumed.final_parts) == "done"


@pytest.mark.asyncio
async def test_pause_request_is_captured_once_before_applying_pause() -> None:
    controller = ClearingReadRunController(PauseRequest(reason="manual_pause"))

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [user_text("finish")],
        controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.reason == "manual_pause"
    assert controller.reads >= 2


@pytest.mark.asyncio
async def test_pause_during_tool_call_model_response_has_no_executing_checkpoint() -> None:
    controller = RunController()
    events = await collect_events(
        AgentLoop(model=SelfPausingToolCallModel(controller), tools=HarnessToolRegistry("echo")),
        [user_text("call tool")],
        controller=controller,
    )
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    paused = checkpoints[-1]

    assert paused.state.status is AgentStatus.PAUSED
    assert paused.state.pause is not None
    assert paused.state.pause.resume_status is AgentStatus.EXECUTING_TOOLS
    assert paused.state.pending_tool_calls == [
        ToolCall(id="call-1", name="echo", arguments={"text": "hello"})
    ]
    assert AgentStatus.EXECUTING_TOOLS not in {snapshot.state.status for snapshot in checkpoints}

    resumed = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        tools=HarnessToolRegistry("echo"),
    ).run_snapshot(ResumeInput(snapshot=paused))

    assert resumed.status is AgentStatus.COMPLETED
    assert [message.text for message in resumed.messages if message.role == "tool"] == ["hello"]


def test_pause_payloads_reject_schema_invalid_types() -> None:
    controller = RunController()

    with pytest.raises(TypeError, match="pause request metadata"):
        controller.request_pause(metadata=cast(Any, []))
    with pytest.raises(TypeError, match="pause request metadata"):
        controller.interrupt(metadata=cast(Any, []))

    with pytest.raises(TypeError, match="pause interrupt"):
        PauseRequest.from_dict(
            {
                "reason": "manual_pause",
                "source": "host",
                "wait_id": None,
                "metadata": {},
                "interrupt": "false",
            }
        )

    with pytest.raises(TypeError, match="pause reason"):
        PauseState.from_dict(
            {
                "reason": 123,
                "resume_status": "planning",
                "source": "host",
                "wait_id": None,
                "metadata": {},
            }
        )

    with pytest.raises(ValueError, match="correlation_id"):
        ConversationInsert.text("external", id="insert-1", source="host", correlation_id="")


def test_controller_exposes_defensive_request_copies() -> None:
    controller = RunController()
    returned = controller.request_pause(metadata={"nested": {"value": 1}})
    cast(dict[str, Any], returned.metadata)["nested"]["value"] = 2

    stored = controller.pause_request
    assert stored is not None
    assert stored.metadata == {"nested": {"value": 1}}

    cast(dict[str, Any], stored.metadata)["nested"]["value"] = 3

    stored_again = controller.pause_request
    assert stored_again is not None
    assert stored_again.metadata == {"nested": {"value": 1}}


def test_agent_state_from_dict_rejects_schema_invalid_required_fields() -> None:
    payload = AgentState(status=AgentStatus.PLANNING, messages=[]).to_dict()
    del payload["pause"]
    with pytest.raises(KeyError):
        AgentState.from_dict(payload)

    payload = AgentState(status=AgentStatus.PLANNING, messages=[]).to_dict()
    payload["messages"] = None
    with pytest.raises(TypeError, match="messages"):
        AgentState.from_dict(payload)

    payload = AgentState(status=AgentStatus.PLANNING, messages=[]).to_dict()
    payload["iterations"] = True
    with pytest.raises(TypeError, match="iterations"):
        AgentState.from_dict(payload)


@pytest.mark.asyncio
async def test_expired_deadline_wins_over_pause_request() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    model = ScriptedModel([ModelResponse.text("done")])
    now = wall_time()

    result = await AgentLoop(model=model).run(
        [user_text("finish")],
        context=RuntimeContext(started_at=now - 2, deadline=now - 1),
        controller=controller,
    )

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert model.calls == 0


@pytest.mark.asyncio
async def test_final_completion_wins_over_pause_requested_during_model_call() -> None:
    controller = RunController()
    model = GateFinalModel()
    run_task = asyncio.create_task(
        AgentLoop(model=model).run(
            [user_text("finish")],
            controller=controller,
        )
    )
    await model.started.wait()

    controller.request_pause(reason="manual_pause")
    model.release.set()
    result = await run_task

    assert result.status is AgentStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert controller.pause_request is None
    assert parts_text(result.final_parts) == "done"


def test_controller_can_be_reused_across_event_loops() -> None:
    class SlowFinalModel:
        async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
            _ = request, context
            await asyncio.sleep(0.01)
            return ModelResponse.text("done")

    async def run_once(controller: RunController) -> AgentStatus:
        result = await AgentLoop(model=SlowFinalModel()).run(
            [user_text("finish")],
            controller=controller,
        )
        return result.status

    controller = RunController()

    assert asyncio.run(run_once(controller)) is AgentStatus.COMPLETED
    assert asyncio.run(run_once(controller)) is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_interrupt_from_worker_thread_pauses_inflight_model_call() -> None:
    controller = RunController()
    model = GateFinalModel()
    run_task = asyncio.create_task(
        AgentLoop(model=model).run(
            [user_text("finish")],
            controller=controller,
        )
    )
    await model.started.wait()

    await asyncio.to_thread(controller.interrupt, reason="thread_interrupt")
    result = await run_task

    assert result.status is AgentStatus.PAUSED
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.reason == "thread_interrupt"


def raise_assertion() -> RunSnapshot:
    raise AssertionError("expected result snapshot")


@pytest.mark.asyncio
async def test_multi_step_tools() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "a"})]),
            ModelResponse(tool_calls=[ToolCall(id="call-2", name="echo", arguments={"text": "b"})]),
            ModelResponse.text("a b"),
        ]
    )
    result = await AgentLoop(model=model, tools=HarnessToolRegistry("echo")).run(
        [user_text("echo twice")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "a b"
    assert result.iterations == 3
    assert result.total_tool_calls == 2


@pytest.mark.asyncio
async def test_max_iterations() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "a"})]),
            ModelResponse(tool_calls=[ToolCall(id="call-2", name="echo", arguments={"text": "b"})]),
            ModelResponse.text("should not be reached"),
        ]
    )
    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("echo"),
        limits=LoopLimits(max_iterations=2),
    ).run([user_text("loop")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_iterations"
    assert result.iterations == 2
    assert result.total_tool_calls == 2


@pytest.mark.asyncio
async def test_max_total_tool_calls_does_not_block_direct_final() -> None:
    model = ScriptedModel([ModelResponse.text("done")])
    result = await AgentLoop(
        model=model,
        limits=LoopLimits(max_total_tool_calls=0),
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"


@pytest.mark.asyncio
async def test_tool_call_limit_takes_precedence_over_pause_after_model_response() -> None:
    controller = RunController()
    events = await collect_events(
        AgentLoop(
            model=SelfPausingToolCallModel(controller),
            tools=HarnessToolRegistry("echo"),
            limits=LoopLimits(max_total_tool_calls=0),
        ),
        [user_text("echo")],
        controller=controller,
    )

    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value
    assert EventTypes.RUN_PAUSED not in [event.type for event in events]
    assert AgentStatus.PAUSED.value not in [
        RunSnapshot.from_dict(event.data).state.status.value
        for event in events
        if event.type == EventTypes.CHECKPOINT
    ]


@pytest.mark.asyncio
async def test_tool_error_recovery_by_default() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="fail", arguments={})]),
            ModelResponse.text("handled"),
        ]
    )
    result = await AgentLoop(model=model, tools=ToolRegistry([FailingFixtureTool()])).run(
        [user_text("fail")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled"
    assert result.total_tool_calls == 1
    assert result.messages[-2].role == "tool"
    assert "tool failed" in result.messages[-2].text


@pytest.mark.asyncio
async def test_waiting_tool_result_pauses_after_tool_commit_and_resumes() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"})]
            ),
            ModelResponse.text("should not be called before resume"),
        ]
    )

    result = await AgentLoop(model=model, tools=HarnessToolRegistry("wait")).run(
        [user_text("start external work")]
    )

    assert result.status is AgentStatus.PAUSED
    assert result.total_tool_calls == 1
    assert model.calls == 1
    assert [message.role for message in result.messages] == ["user", "assistant", "tool"]
    assert result.messages[-1].text == "external job started"
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.reason == "external_callback"
    assert result.snapshot.state.pause.source == "tool"
    assert result.snapshot.state.pause.wait_id == "job-1"
    assert result.snapshot.state.pause.resume_status is AgentStatus.PLANNING

    resumed_model = ScriptedModel([ModelResponse.text("external job complete")])
    resumed = await AgentLoop(
        model=resumed_model,
        tools=HarnessToolRegistry("wait"),
    ).run_snapshot(ResumeInput(snapshot=result.snapshot or raise_assertion()))

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed_model.calls == 1
    assert parts_text(resumed.final_parts) == "external job complete"


@pytest.mark.asyncio
async def test_external_wait_has_no_resumable_checkpoint_before_paused_decision() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"})]
            ),
        ]
    )

    events = await collect_events(
        AgentLoop(model=model, tools=HarnessToolRegistry("wait")),
        [user_text("start external work")],
    )
    snapshots = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    committed_tool_snapshots = [
        snapshot for snapshot in snapshots if snapshot.state.total_tool_calls == 1
    ]

    assert committed_tool_snapshots
    assert {snapshot.state.status for snapshot in committed_tool_snapshots} == {AgentStatus.PAUSED}
    assert committed_tool_snapshots[-1].state.pause is not None
    assert committed_tool_snapshots[-1].state.pause.wait_id == "job-1"

    resumed = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
        ResumeInput(snapshot=committed_tool_snapshots[-1])
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert parts_text(resumed.final_parts) == "done"


@pytest.mark.asyncio
async def test_host_pause_during_tool_completion_replaces_unpaused_commit_checkpoint() -> None:
    controller = RunController()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
            )
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            hooks=[RequestPauseOnEventHook(EventTypes.TOOL_COMPLETED, controller)],
        ),
        [user_text("run tool")],
        controller=controller,
    )
    committed_tool_snapshots = [
        RunSnapshot.from_dict(event.data)
        for event in events
        if event.type == EventTypes.CHECKPOINT
        and RunSnapshot.from_dict(event.data).state.total_tool_calls == 1
    ]

    assert committed_tool_snapshots
    assert {snapshot.state.status for snapshot in committed_tool_snapshots} == {AgentStatus.PAUSED}
    paused = committed_tool_snapshots[-1].state.pause
    assert paused is not None
    assert paused.resume_status is AgentStatus.PLANNING


@pytest.mark.asyncio
async def test_parallel_waiting_tool_results_commit_batch_and_first_pause_wins() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="parallel_wait",
                        arguments={"wait_id": "job-1", "delay": 0.02},
                    ),
                    ToolCall(
                        id="call-2",
                        name="parallel_wait",
                        arguments={"wait_id": "job-2", "delay": 0},
                    ),
                ]
            ),
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("parallel_wait"),
        limits=LoopLimits(max_parallel_tool_calls=2),
    ).run([user_text("start external work")])

    assert result.status is AgentStatus.PAUSED
    assert result.total_tool_calls == 2
    assert [message.tool_call_id for message in result.messages if message.role == "tool"] == [
        "call-1",
        "call-2",
    ]
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.wait_id == "job-1"
    assert result.snapshot.state.pause.resume_status is AgentStatus.PLANNING


@pytest.mark.asyncio
async def test_stop_on_tool_error() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="fail", arguments={})])]
    )
    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([FailingFixtureTool()]),
        limits=LoopLimits(stop_on_tool_error=True),
    ).run([user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "tool failed"
    assert result.total_tool_calls == 1
    assert result.messages[-1].role == "tool"
    assert result.messages[-1].metadata["is_error"] is True


@pytest.mark.asyncio
async def test_stop_on_tool_error_has_no_resumable_checkpoint_before_failed_decision() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="fail", arguments={})])]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([FailingFixtureTool()]),
            limits=LoopLimits(stop_on_tool_error=True),
        ),
        [user_text("fail")],
    )
    snapshots = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    committed_tool_snapshots = [
        snapshot for snapshot in snapshots if snapshot.state.total_tool_calls == 1
    ]

    assert committed_tool_snapshots
    assert {snapshot.state.status for snapshot in committed_tool_snapshots} == {AgentStatus.FAILED}

    with pytest.raises(ValueError, match="terminal"):
        ResumeInput(snapshot=committed_tool_snapshots[-1])


@pytest.mark.asyncio
async def test_unknown_tool_is_observation_error() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="missing", arguments={})]),
            ModelResponse.text("handled"),
        ]
    )
    result = await AgentLoop(model=model).run([user_text("call missing")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled"
    assert "unknown tool" in result.messages[-2].text
    assert result.messages[-2].metadata["is_error"] is True


@pytest.mark.asyncio
async def test_unknown_tool_is_not_implementation_invoked() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="missing", arguments={})]),
            ModelResponse.text("handled"),
        ]
    )
    result = await AgentLoop(model=model).run([user_text("call missing")])

    assert result.trace is not None
    tool_call = next(step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_CALL)
    tool_result = next(step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_RESULT)
    assert tool_call.payload["implementation_invoked"] is False
    assert tool_result.payload["implementation_invoked"] is False
    assert tool_result.payload["result"]["is_error"] is True
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_tool_input_schema_error_is_committed_as_observation_error() -> None:
    tool = StrictCountFixtureTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="strict_count",
                        arguments={"count": "bad"},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run(
        [user_text("call strict")]
    )

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled"
    assert tool.calls == 0
    assert "input_schema" in result.messages[-2].text
    assert result.messages[-2].metadata["is_error"] is True


@pytest.mark.asyncio
async def test_tool_input_schema_error_is_not_implementation_invoked() -> None:
    tool = StrictCountFixtureTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="strict_count",
                        arguments={"count": "bad"},
                    )
                ]
            ),
            ModelResponse.text("handled"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run(
        [user_text("call strict")]
    )

    assert tool.calls == 0
    assert result.trace is not None
    tool_call = next(step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_CALL)
    tool_result = next(step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_RESULT)
    assert tool_call.payload["implementation_invoked"] is False
    assert tool_result.payload["implementation_invoked"] is False
    assert tool_result.payload["result"]["is_error"] is True
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_after_tool_cannot_turn_runtime_validation_error_into_success() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel(
                [ModelResponse(tool_calls=[ToolCall(id="call-1", name="missing", arguments={})])]
            ),
            hooks=[SuccessfulAfterToolHook()],
        ),
        [user_text("call missing")],
    )

    assert EventTypes.TOOL_COMPLETED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value
    assert "non-invoked tool result" in events[-2].data["message"]


@pytest.mark.asyncio
async def test_after_tool_cannot_turn_approval_denial_into_success() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=[
                            ToolCall(id="call-1", name="record", arguments={"id": "denied"})
                        ]
                    )
                ]
            ),
            tools=RecordingToolRegistry(),
            approval_policy=StaticApprovalPolicy(ApprovalDecision.deny("blocked")),
            hooks=[SuccessfulAfterToolHook()],
        ),
        [user_text("use tool")],
    )

    assert EventTypes.TOOL_COMPLETED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value
    assert "non-invoked tool result" in events[-2].data["message"]


@pytest.mark.asyncio
async def test_after_tool_cannot_turn_runtime_validation_error_into_pause() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel(
                [ModelResponse(tool_calls=[ToolCall(id="call-1", name="missing", arguments={})])]
            ),
            hooks=[PausingAfterToolHook()],
        ),
        [user_text("call missing")],
    )

    assert EventTypes.TOOL_COMPLETED not in [event.type for event in events]
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value
    assert "must not request pause" in events[-2].data["message"]


@pytest.mark.asyncio
async def test_after_tool_cannot_turn_approval_denial_into_pause() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=[
                            ToolCall(id="call-1", name="record", arguments={"id": "denied"})
                        ]
                    )
                ]
            ),
            tools=RecordingToolRegistry(),
            approval_policy=StaticApprovalPolicy(ApprovalDecision.deny("blocked")),
            hooks=[PausingAfterToolHook()],
        ),
        [user_text("use tool")],
    )

    assert EventTypes.TOOL_COMPLETED not in [event.type for event in events]
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value
    assert "must not request pause" in events[-2].data["message"]


@pytest.mark.asyncio
async def test_model_timeout_is_hard_limit() -> None:
    result = await AgentLoop(
        model=SlowModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_adapter_timeout_error_is_failure_not_runtime_limit() -> None:
    result = await AgentLoop(model=AdapterTimeoutModel()).run([user_text("slow")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "provider timeout"


@pytest.mark.asyncio
async def test_runtime_timeout_wins_over_converted_cancellation_error() -> None:
    result = await AgentLoop(
        model=CancellationConvertingModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"


@pytest.mark.asyncio
async def test_runtime_timeout_does_not_wait_for_swallowed_cancellation() -> None:
    started_at = monotonic()
    result = await AgentLoop(
        model=CancellationSwallowingModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([user_text("slow")])
    elapsed = monotonic() - started_at

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert elapsed < 0.04


@pytest.mark.asyncio
async def test_runtime_timeout_consumes_late_background_task_exception() -> None:
    loop = asyncio.get_running_loop()
    captured: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def handler(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        captured.append(context)

    loop.set_exception_handler(handler)
    try:
        result = await AgentLoop(
            model=CancellationSwallowingThenFailingModel(),
            limits=LoopLimits(timeout_seconds=0.01),
        ).run([user_text("slow")])
        await asyncio.sleep(0.03)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert captured == []


@pytest.mark.asyncio
async def test_external_cancellation_cancels_child_task_and_consumes_late_exception() -> None:
    loop = asyncio.get_running_loop()
    captured: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    model = ExternallyCancelledModel()

    def handler(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        captured.append(context)

    loop.set_exception_handler(handler)
    try:
        task = asyncio.create_task(AgentLoop(model=model).run([user_text("cancel")]))
        await model.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.03)
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert captured == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", [SlowAfterModelHook(), SlowTransitionHook(), SlowEventHook()])
async def test_hook_timeout_is_runtime_limit(hook: RuntimeHook) -> None:
    started_at = monotonic()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=0.01),
        hooks=[hook],
    ).run([user_text("slow hook")])
    elapsed = monotonic() - started_at

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert elapsed < 0.04


@pytest.mark.asyncio
async def test_raw_timeout_terminal_events_are_ordered() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            limits=LoopLimits(timeout_seconds=0.01),
            hooks=[SlowEventHook()],
        ),
        [user_text("slow hook")],
    )

    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.ERROR,
        EventTypes.RUN_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_generic_exception_terminal_events_are_ordered() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingEventHook()],
        ),
        [user_text("raise")],
    )

    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.ERROR,
        EventTypes.RUN_COMPLETED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_type",
    [EventTypes.FINAL, EventTypes.RUN_COMPLETED],
)
async def test_post_completed_event_hook_failure_does_not_rewrite_terminal_checkpoint(
    event_type: str,
) -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(event_type)],
        ),
        [user_text("finish")],
    )
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]

    assert checkpoints
    assert {snapshot.state.status for snapshot in checkpoints} == {AgentStatus.COMPLETED}
    assert events[-2].type == EventTypes.ERROR
    assert events[-2].data["status"] == AgentStatus.COMPLETED.value
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.COMPLETED.value

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(event_type)],
    ).run([user_text("finish")])

    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_model_completed_event_hook_failure_rolls_back_uncheckpointed_transition() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.MODEL_COMPLETED)],
    ).run([user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert [message.role for message in result.messages] == ["user"]
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED
    transitions = [step for step in rt(result).steps if step.kind == TraceStepKinds.STATE_CHANGED]
    assert transitions[-1].before_status is AgentStatus.PLANNING
    assert transitions[-1].after_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_started_event_hook_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.RUN_STARTED)],
    ).run([user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert rt(result).steps[0].kind == TraceStepKinds.RUN_STARTED
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_started_event_hook_failure_event_stream_still_replays() -> None:
    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.RUN_STARTED)],
        ).run_events([user_text("finish")])
    ]

    assert events[0].type == EventTypes.RUN_STARTED
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert replay_trace(RunTrace.from_events(events[0].run_id, events)).final_status is (
        AgentStatus.FAILED
    )


@pytest.mark.asyncio
async def test_child_run_events_emit_when_run_started_hook_fails() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.RUN_STARTED)],
        ).run_events([user_text("finish")], context=context)
    ]

    event_types = [event.type for event in events]
    assert event_types[:2] == [EventTypes.RUN_STARTED, EventTypes.CHILD_RUN_STARTED]
    assert event_types.count(EventTypes.CHILD_RUN_STARTED) == 1
    assert event_types.count(EventTypes.CHILD_RUN_COMPLETED) == 1
    assert event_types.index(EventTypes.CHILD_RUN_COMPLETED) < event_types.index(
        EventTypes.RUN_COMPLETED
    )
    assert replay_trace(RunTrace.from_events(events[0].run_id, events)).final_status is (
        AgentStatus.FAILED
    )

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.RUN_STARTED)],
    ).run([user_text("finish")], context=context)
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED
    result_kinds = [step.kind for step in rt(result).steps]
    assert TraceStepKinds.CHILD_RUN_STARTED in result_kinds
    assert TraceStepKinds.CHILD_RUN_COMPLETED in result_kinds


@pytest.mark.asyncio
async def test_checkpoint_event_hook_failure_preserves_persisted_terminal_checkpoint() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.CHECKPOINT)],
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED
    transitions = [step for step in rt(result).steps if step.kind == TraceStepKinds.STATE_CHANGED]
    assert transitions[-1].after_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_run_completed_hook_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ProviderErrorModel(),
        hooks=[RaisingOnEventHook(EventTypes.RUN_COMPLETED)],
    ).run([user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_checkpoint_custom_event_failure_preserves_trace_durability() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[FailingQueuedEventAfterHook(EventTypes.CHECKPOINT)],
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.state.status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_run_completed_custom_event_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[FailingQueuedEventAfterHook(EventTypes.RUN_COMPLETED)],
    ).run([user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_post_paused_event_hook_failure_does_not_rewrite_terminal_checkpoint() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.RUN_PAUSED)],
        ),
        [user_text("finish")],
        controller=controller,
    )
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]

    assert checkpoints
    assert {snapshot.state.status for snapshot in checkpoints} == {AgentStatus.PAUSED}
    assert events[-2].type == EventTypes.ERROR
    assert events[-2].data["status"] == AgentStatus.PAUSED.value
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.PAUSED.value

    controller = RunController()
    controller.request_pause(reason="manual_pause")
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.RUN_PAUSED)],
    ).run(
        [user_text("finish")],
        controller=controller,
    )

    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.PAUSED


@pytest.mark.asyncio
async def test_pause_request_made_during_run_paused_is_cleared_at_invocation_end() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RequestPauseOnEventHook(EventTypes.RUN_PAUSED, controller)],
    ).run(
        [user_text("finish")],
        controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert controller.pause_request is None


@pytest.mark.asyncio
async def test_pause_requested_event_hook_failure_clears_controller() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")

    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.PAUSE_REQUESTED)],
        ),
        [user_text("finish")],
        controller=controller,
    )

    assert controller.pause_request is None
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_sync_blocking_hook_timeout_is_runtime_limit() -> None:
    started_at = monotonic()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=0.01),
        hooks=[BlockingAfterModelHook()],
    ).run([user_text("blocking hook")])
    elapsed = monotonic() - started_at

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert elapsed < 0.04


@pytest.mark.asyncio
async def test_tool_timeout_is_hard_limit() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="slow", arguments={})])]
    )
    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("slow"),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.total_tool_calls == 0


@pytest.mark.asyncio
async def test_approval_allow_tool_timeout_trace_replays() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="slow", arguments={})])]
    )
    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("slow"),
        limits=LoopLimits(timeout_seconds=0.01),
        approval_policy=StaticApprovalPolicy(ApprovalDecision.allow("safe")),
    ).run([user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.trace is not None
    tool_call = next(step for step in rt(result).steps if step.kind == TraceStepKinds.TOOL_CALL)
    assert tool_call.payload["implementation_invoked"] is True
    assert not any(step.kind == TraceStepKinds.TOOL_RESULT for step in rt(result).steps)
    assert replay_trace(result.trace).final_status is AgentStatus.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_tool_call_limit_leaves_only_unexecuted_pending_calls() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="call-2", name="echo", arguments={"text": "b"}),
                ]
            )
        ]
    )
    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [user_text("echo twice")],
    )

    assert events[-1].data["state"]["pending_tool_call_count"] == 1
    assert events[-1].data["state"]["total_tool_calls"] == 1


@pytest.mark.asyncio
async def test_tool_call_limit_wins_over_tool_result_pause() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"}),
                    ToolCall(id="call-2", name="echo", arguments={"text": "after"}),
                ]
            )
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("wait", "echo"),
        limits=LoopLimits(max_total_tool_calls=1),
    ).run([user_text("wait then echo")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_total_tool_calls"
    assert result.total_tool_calls == 1
    assert result.snapshot is not None
    assert [call.id for call in result.snapshot.state.pending_tool_calls] == ["call-2"]
    assert result.snapshot.state.pause is None
    assert result.trace is not None
    assert TraceStepKinds.PAUSE_REQUESTED not in [step.kind for step in rt(result).steps]


@pytest.mark.asyncio
async def test_tool_call_limit_wins_over_pause_requested_after_tool_completed() -> None:
    controller = RunController()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="call-2", name="echo", arguments={"text": "b"}),
                ]
            )
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            hooks=[RequestPauseOnEventHook(EventTypes.TOOL_COMPLETED, controller)],
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [user_text("echo twice")],
        controller=controller,
    )

    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value
    assert events[-1].data["state"]["error"] == "max_total_tool_calls"
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]


@pytest.mark.asyncio
async def test_tool_result_pause_loses_to_iteration_limit_after_final_tool() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"})]
            )
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("wait"),
        limits=LoopLimits(max_iterations=1),
    ).run([user_text("wait")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_iterations"
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert result.trace is not None
    assert TraceStepKinds.PAUSE_REQUESTED not in [step.kind for step in rt(result).steps]


@pytest.mark.asyncio
async def test_tool_final_planning_boundary_applies_pause_before_checkpoint() -> None:
    controller = RunController()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "done"})]
            ),
            ModelResponse.text("unused"),
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            hooks=[
                RequestPauseOnStateChangeHook(
                    AgentStatus.EXECUTING_TOOLS,
                    AgentStatus.PLANNING,
                    controller,
                )
            ],
        ),
        [user_text("echo")],
        controller=controller,
    )
    event_types = [event.type for event in events]
    planning_transition_index = next(
        index
        for index, event in enumerate(events)
        if event.type == EventTypes.STATE_CHANGED
        and event.data["from"] == AgentStatus.EXECUTING_TOOLS.value
        and event.data["to"] == AgentStatus.PLANNING.value
    )
    pause_index = event_types.index(EventTypes.PAUSE_REQUESTED)

    assert events[-1].data["state"]["status"] == AgentStatus.PAUSED.value
    assert events[-1].data["state"]["pause"]["resume_status"] == AgentStatus.PLANNING.value
    assert EventTypes.CHECKPOINT not in event_types[planning_transition_index + 1 : pause_index]
    assert model.calls == 1


@pytest.mark.asyncio
async def test_iteration_limit_wins_over_pause_requested_after_final_tool() -> None:
    controller = RunController()
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "done"})])]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            hooks=[
                RequestPauseOnStateChangeHook(
                    AgentStatus.EXECUTING_TOOLS,
                    AgentStatus.PLANNING,
                    controller,
                )
            ],
            limits=LoopLimits(max_iterations=1),
        ),
        [user_text("echo")],
        controller=controller,
    )

    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value
    assert events[-1].data["state"]["error"] == "max_iterations"
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]


@pytest.mark.asyncio
async def test_parallel_safe_tools_default_to_serial_execution() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id=f"call-{index}", name="timed", arguments={"id": str(index)})
                    for index in range(3)
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    result = await AgentLoop(model=model, tools=ToolRegistry([tool])).run([user_text("timed")])

    assert result.status is AgentStatus.COMPLETED
    assert result.total_tool_calls == 3
    assert tool.max_active == 1


@pytest.mark.asyncio
async def test_parallel_safe_tools_obey_max_parallel_tool_calls() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id=f"call-{index}",
                        name="timed",
                        arguments={"id": str(index), "delay": 0.02},
                    )
                    for index in range(5)
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([tool]),
        limits=LoopLimits(max_parallel_tool_calls=4),
    ).run([user_text("timed")])
    tool_messages = [message.text for message in result.messages if message.role == "tool"]

    assert result.status is AgentStatus.COMPLETED
    assert tool.max_active == 4
    assert tool_messages == ["0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_parallel_tool_context_mutation_is_isolated_per_call() -> None:
    tool = ContextMutatingParallelTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-writer",
                        name="context_mutator",
                        arguments={"id": "writer"},
                    ),
                    ToolCall(
                        id="call-reader",
                        name="context_mutator",
                        arguments={"id": "reader"},
                    ),
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([tool]),
        limits=LoopLimits(max_parallel_tool_calls=2),
    ).run([user_text("run")])
    tool_messages = [message.text for message in result.messages if message.role == "tool"]

    assert result.status is AgentStatus.COMPLETED
    assert tool_messages == ["writer", "missing"]


@pytest.mark.asyncio
async def test_tool_batch_ids_reset_for_each_run_on_reused_loop() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={})]),
            ModelResponse.text("first"),
            ModelResponse(tool_calls=[ToolCall(id="call-2", name="echo", arguments={})]),
            ModelResponse.text("second"),
        ]
    )
    agent = AgentLoop(model=model, tools=HarnessToolRegistry("echo"))

    first_events = await collect_events(agent, [user_text("first")])
    second_events = await collect_events(agent, [user_text("second")])

    assert [
        event.data["batch_id"] for event in first_events if event.type == EventTypes.TOOL_STARTED
    ] == ["tool-batch-1"]
    assert [
        event.data["batch_id"] for event in second_events if event.type == EventTypes.TOOL_STARTED
    ] == ["tool-batch-1"]


@pytest.mark.asyncio
async def test_parallel_completion_events_can_be_out_of_order_but_history_is_ordered() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="slow", name="timed", arguments={"id": "slow", "delay": 0.03}),
                    ToolCall(id="fast", name="timed", arguments={"id": "fast", "delay": 0.01}),
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(max_parallel_tool_calls=2),
        ),
        [user_text("timed")],
    )
    completed_ids = [
        str(event.data["id"]) for event in events if event.type == EventTypes.TOOL_COMPLETED
    ]
    final_snapshot = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-2].data
    )
    tool_messages = [
        message.tool_call_id for message in final_snapshot.state.messages if message.role == "tool"
    ]

    assert completed_ids == ["fast", "slow"]
    assert tool_messages == ["slow", "fast"]


@pytest.mark.asyncio
async def test_parallel_same_tick_completion_events_are_index_ordered() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id=f"call-{index}", name="timed", arguments={"id": str(index)})
                    for index in range(3)
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(max_parallel_tool_calls=3),
        ),
        [user_text("timed")],
    )

    assert [event.data["id"] for event in events if event.type == EventTypes.TOOL_COMPLETED] == [
        "call-0",
        "call-1",
        "call-2",
    ]


@pytest.mark.asyncio
async def test_parallel_checkpoint_waits_for_completed_order_gap() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="slow", name="timed", arguments={"id": "slow", "delay": 0.03}),
                    ToolCall(id="fast", name="timed", arguments={"id": "fast", "delay": 0.01}),
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(max_parallel_tool_calls=2),
        ),
        [user_text("timed")],
    )
    snapshots = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    tool_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.state.status is AgentStatus.PLANNING and snapshot.state.total_tool_calls == 2
    ]
    invalid_tool_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.state.status is AgentStatus.EXECUTING_TOOLS
        and not snapshot.state.pending_tool_calls
    ]

    assert tool_snapshots
    assert tool_snapshots[0].state.total_tool_calls == 2
    assert tool_snapshots[0].state.pending_tool_calls == []
    assert invalid_tool_snapshots == []
    assert ResumeInput(snapshot=tool_snapshots[0]).snapshot.to_dict() == tool_snapshots[0].to_dict()
    assert [
        message.tool_call_id
        for message in tool_snapshots[0].state.messages
        if message.role == "tool"
    ] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_parallel_checkpoint_waits_for_same_wait_completed_gap() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-0", name="timed", arguments={"id": "0", "delay": 0}),
                    ToolCall(id="call-1", name="timed", arguments={"id": "1", "delay": 0.03}),
                    ToolCall(id="call-2", name="timed", arguments={"id": "2", "delay": 0}),
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(max_parallel_tool_calls=3),
        ),
        [user_text("timed")],
    )
    tool_commit_counts = [
        RunSnapshot.from_dict(event.data).state.total_tool_calls
        for event in events
        if event.type == EventTypes.CHECKPOINT
        and RunSnapshot.from_dict(event.data).state.status is AgentStatus.EXECUTING_TOOLS
    ]

    assert 1 not in tool_commit_counts


@pytest.mark.asyncio
async def test_parallel_timeout_does_not_checkpoint_partial_same_wait_commit() -> None:
    tool = GappedTimeoutTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-0", name="gapped_timeout", arguments={"id": "0"}),
                    ToolCall(id="call-1", name="gapped_timeout", arguments={"id": "1"}),
                    ToolCall(id="call-2", name="gapped_timeout", arguments={"id": "2"}),
                ]
            )
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(timeout_seconds=0.02, max_parallel_tool_calls=3),
        ),
        [user_text("timed")],
    )
    snapshots = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]

    assert 1 not in [snapshot.state.total_tool_calls for snapshot in snapshots]
    assert snapshots[-1].state.status is AgentStatus.LIMIT_EXCEEDED
    assert snapshots[-1].state.total_tool_calls == 0


@pytest.mark.asyncio
async def test_parallel_timeout_does_not_checkpoint_staggered_partial_commit() -> None:
    tool = StaggeredGappedTimeoutTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-0", name="staggered_gapped_timeout", arguments={"id": "0"}),
                    ToolCall(id="call-1", name="staggered_gapped_timeout", arguments={"id": "1"}),
                    ToolCall(id="call-2", name="staggered_gapped_timeout", arguments={"id": "2"}),
                ]
            )
        ]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(timeout_seconds=0.03, max_parallel_tool_calls=3),
        ),
        [user_text("timed")],
    )
    snapshots = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]

    assert 1 not in [snapshot.state.total_tool_calls for snapshot in snapshots]
    assert snapshots[-1].state.status is AgentStatus.LIMIT_EXCEEDED
    assert snapshots[-1].state.total_tool_calls == 0
    assert [
        message.tool_call_id for message in snapshots[-1].state.messages if message.role == "tool"
    ] == []


@pytest.mark.asyncio
async def test_stop_on_tool_error_disables_parallel_backfill() -> None:
    tool = MaybeErrorTimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-0", name="timed", arguments={"id": "0", "delay": 0.02}),
                    ToolCall(
                        id="call-1",
                        name="timed",
                        arguments={"id": "1", "delay": 0.001, "error": True},
                    ),
                    ToolCall(id="call-2", name="timed", arguments={"id": "2", "delay": 0.001}),
                    ToolCall(id="call-3", name="timed", arguments={"id": "3", "delay": 0.001}),
                ]
            )
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([tool]),
        limits=LoopLimits(max_parallel_tool_calls=4, stop_on_tool_error=True),
    ).run([user_text("timed")])
    started = [call_id for phase, call_id, _at in tool.timeline if phase == "start"]

    assert result.status is AgentStatus.FAILED
    assert started == ["0", "1"]
    assert tool.max_active == 1
    assert result.total_tool_calls == 2


@pytest.mark.asyncio
async def test_unsafe_tool_call_is_parallel_barrier() -> None:
    safe = TimedTool("safe", parallel_safe=True)
    unsafe = TimedTool("unsafe", parallel_safe=False)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="a", name="safe", arguments={"id": "a", "delay": 0.02}),
                    ToolCall(id="b", name="safe", arguments={"id": "b", "delay": 0.02}),
                    ToolCall(id="c", name="unsafe", arguments={"id": "c", "delay": 0.01}),
                    ToolCall(id="d", name="safe", arguments={"id": "d", "delay": 0.02}),
                    ToolCall(id="e", name="safe", arguments={"id": "e", "delay": 0.02}),
                ]
            ),
            ModelResponse.text("done"),
        ]
    )

    result = await AgentLoop(
        model=model,
        tools=ToolRegistry([safe, unsafe]),
        limits=LoopLimits(max_parallel_tool_calls=4),
    ).run([user_text("timed")])

    safe_times = {(phase, call_id): at for phase, call_id, at in safe.timeline}
    unsafe_times = {(phase, call_id): at for phase, call_id, at in unsafe.timeline}

    assert result.status is AgentStatus.COMPLETED
    assert safe.max_active == 2
    assert unsafe.max_active == 1
    assert unsafe_times[("start", "c")] >= max(safe_times[("end", "a")], safe_times[("end", "b")])
    assert min(safe_times[("start", "d")], safe_times[("start", "e")]) >= unsafe_times[("end", "c")]


@pytest.mark.asyncio
async def test_parallel_tool_call_limit_starts_only_available_slots() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id=f"call-{index}", name="timed", arguments={"id": str(index)})
                    for index in range(5)
                ]
            )
        ]
    )
    events = await collect_events(
        AgentLoop(
            model=model,
            tools=ToolRegistry([tool]),
            limits=LoopLimits(max_total_tool_calls=3, max_parallel_tool_calls=4),
        ),
        [user_text("timed")],
    )

    assert tool.max_active == 3
    assert events[-1].data["state"]["pending_tool_call_count"] == 2
    assert events[-1].data["state"]["total_tool_calls"] == 3


@pytest.mark.asyncio
async def test_checkpoint_event_after_each_tool_commit_is_resumable() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="echo", arguments={"text": "a"}),
                    ToolCall(id="call-2", name="echo", arguments={"text": "b"}),
                ]
            )
        ]
    )
    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [user_text("echo twice")],
    )
    checkpoint_events = [event for event in events if event.type == EventTypes.CHECKPOINT]
    snapshots = [RunSnapshot.from_dict(event.data) for event in checkpoint_events]
    tool_snapshot = next(snapshot for snapshot in snapshots if snapshot.state.total_tool_calls == 1)

    assert [RunSnapshot.from_dict(event.data).context.sequence for event in checkpoint_events] == [
        event.sequence for event in checkpoint_events
    ]
    assert tool_snapshot.context.sequence > 0
    assert tool_snapshot.state.pending_tool_calls == [
        ToolCall(id="call-2", name="echo", arguments={"text": "b"})
    ]
    assert tool_snapshot.state.messages[-1].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_model_tool_checkpoint_resumes_into_tool_execution_without_recalling_model() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="record", arguments={"id": "call-1"}),
                    ToolCall(id="call-2", name="record", arguments={"id": "call-2"}),
                ]
            ),
            ModelResponse.text("initial done"),
        ]
    )
    events = await collect_events(
        AgentLoop(model=model, tools=RecordingToolRegistry()),
        [user_text("record twice")],
    )
    first_checkpoint = next(event for event in events if event.type == EventTypes.CHECKPOINT)
    snapshot = RunSnapshot.from_dict(first_checkpoint.data)
    resumed_model = ScriptedModel([ModelResponse.text("resumed done")])
    resumed_tool = RecordingToolRegistry()

    result = await AgentLoop(model=resumed_model, tools=resumed_tool).run_snapshot(
        ResumeInput(snapshot=snapshot)
    )

    assert snapshot.state.status is AgentStatus.EXECUTING_TOOLS
    assert resumed_tool.calls == ["call-1", "call-2"]
    assert resumed_model.calls == 1
    assert result.status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_terminal_snapshot_resume_is_rejected() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("first-final")])),
        [user_text("finish")],
    )
    first_checkpoint = next(event for event in events if event.type == EventTypes.CHECKPOINT)
    snapshot = RunSnapshot.from_dict(first_checkpoint.data)
    resumed_model = ScriptedModel([ModelResponse.text("second-final")])

    assert snapshot.state.status is AgentStatus.COMPLETED
    with pytest.raises(ValueError, match="terminal"):
        ResumeInput(snapshot=snapshot)
    assert resumed_model.calls == 0


@pytest.mark.asyncio
async def test_hooks_can_modify_model_response_and_observe_events() -> None:
    hook = RewritingHook()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("original")]),
        hooks=[hook],
    ).run([user_text("hello")])

    assert parts_text(result.final_parts) == "hooked"
    assert hook.events[0] == EventTypes.RUN_STARTED
    assert EventTypes.FINAL in hook.events


@pytest.mark.asyncio
async def test_hooks_can_emit_custom_events() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")]), hooks=[RewritingHook()]),
        [user_text("hello")],
    )

    assert "custom_progress" in [event.type for event in events]


@pytest.mark.asyncio
async def test_context_metadata_and_run_id_are_injected() -> None:
    context = RuntimeContext(
        run_id="run-test",
        metadata={"tenant": "acme"},
    )
    events = await collect_events(
        AgentLoop(model=ContextInspectingModel()),
        [user_text("hello")],
    )
    result = await AgentLoop(model=ContextInspectingModel()).run(
        [user_text("hello")],
        context=context,
    )

    assert parts_text(result.final_parts) == "acme"
    assert result.run_id == "run-test"
    assert result.snapshot is not None
    assert result.snapshot.context.run_id == "run-test"
    assert result.snapshot.context.metadata == {"tenant": "acme"}
    assert all(event.run_id for event in events)


@pytest.mark.asyncio
async def test_structural_hook_without_runtime_hook_base_is_called() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("original")]),
        hooks=[StructuralAfterModelHook()],
    ).run([user_text("hello")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "structural hook"


@pytest.mark.asyncio
async def test_runtime_context_uses_wall_clock_checkpoint_deadline() -> None:
    started_at = wall_time()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=10),
    ).run([user_text("hello")])

    assert result.snapshot is not None
    assert result.snapshot.context.started_at >= started_at
    assert result.snapshot.context.deadline is not None
    assert result.snapshot.context.deadline > wall_time()
    assert result.snapshot.context.deadline - result.snapshot.context.started_at <= 10.1


@pytest.mark.asyncio
async def test_event_envelope_ignores_context_mutation_by_hook() -> None:
    context = RuntimeContext(run_id="stable-run")
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingEventContextHook()],
    )
    events = [event async for event in agent.run_events([user_text("hello")], context=context)]

    assert [event.run_id for event in events] == ["stable-run"] * len(events)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert context.run_id == "stable-run"


@pytest.mark.asyncio
async def test_result_uses_runtime_control_identity_after_context_mutation() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingEventContextHook()],
    ).run([user_text("hello")], context=RuntimeContext(run_id="stable-run"))

    assert result.run_id == "stable-run"
    assert result.snapshot is not None
    assert result.snapshot.context.run_id == "stable-run"
    assert result.snapshot.context.sequence > 0


def test_raw_state_runner_is_not_public_resume_api() -> None:
    agent = AgentLoop(model=ScriptedModel([ModelResponse.text("done")]))

    assert not hasattr(agent, "run_state")
    assert not hasattr(agent, "run_state_events")


@pytest.mark.asyncio
async def test_run_snapshot_round_trip_and_resume() -> None:
    snapshot = RunSnapshot(
        state=AgentState(
            status=AgentStatus.PLANNING,
            messages=[user_text("finish")],
        ),
        context=RuntimeContext(run_id="snapshot-run", metadata={"tenant": "acme"}),
    )
    restored = RunSnapshot.from_dict(snapshot.to_dict())
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
        ResumeInput(snapshot=restored)
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.run_id == "snapshot-run"
    assert result.snapshot is not None
    assert result.snapshot.context.run_id == "snapshot-run"
    assert result.snapshot.context.metadata == {"tenant": "acme"}


@pytest.mark.asyncio
async def test_run_snapshot_requires_resume_input() -> None:
    snapshot = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("finish")]),
        context=RuntimeContext(run_id="snapshot-run"),
    )

    with pytest.raises(TypeError, match="ResumeInput"):
        await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
            cast(Any, snapshot)
        )

    iterator = AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot_events(
        cast(Any, snapshot)
    )
    with pytest.raises(TypeError, match="ResumeInput"):
        await anext(iterator)


@pytest.mark.asyncio
async def test_resume_input_appends_messages_and_matches_expected_pause() -> None:
    controller = RunController()
    controller.request_pause(
        reason="external_callback",
        source="tool",
        wait_id="job-1",
        metadata={"tenant": "acme"},
    )
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [user_text("start")],
        controller=controller,
    )
    model = RequestCapturingModel()

    result = await AgentLoop(model=model).run_snapshot(
        ResumeInput(
            snapshot=paused.snapshot or raise_assertion(),
            append_messages=[user_text("callback complete")],
            expected_pause=PauseSelector(
                source="tool", wait_id="job-1", metadata={"tenant": "acme"}
            ),
            metadata={"resumed_by": "test"},
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert model.request is not None
    assert [message.text for message in model.request.messages] == [
        "start",
        "callback complete",
    ]


@pytest.mark.asyncio
async def test_controller_pause_source_tool_replays_as_control_origin() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause", source="tool", wait_id="job-1")

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [user_text("start")],
        controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert result.trace is not None
    pause_steps = [step for step in rt(result).steps if step.kind == TraceStepKinds.PAUSE_REQUESTED]
    assert pause_steps[-1].payload["origin"] == "control"
    assert replay_trace(result.trace).final_status is AgentStatus.PAUSED


@pytest.mark.asyncio
async def test_resume_input_rejects_append_messages_when_resuming_pending_tools() -> None:
    paused = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"}),
                        ToolCall(id="call-2", name="echo", arguments={"text": "after"}),
                    ]
                )
            ]
        ),
        tools=HarnessToolRegistry("wait", "echo"),
    ).run([user_text("start")])

    assert paused.snapshot is not None
    assert paused.snapshot.state.status is AgentStatus.PAUSED
    assert paused.snapshot.state.pause is not None
    assert paused.snapshot.state.pause.resume_status is AgentStatus.EXECUTING_TOOLS
    with pytest.raises(ValueError, match="resumes to planning"):
        ResumeInput(
            snapshot=paused.snapshot,
            append_messages=[user_text("callback complete")],
        )


def test_resume_input_rejects_inconsistent_pending_tool_snapshots() -> None:
    pending_call = ToolCall(id="call-1", name="echo", arguments={})
    second_pending_call = ToolCall(id="call-2", name="echo", arguments={})

    with pytest.raises(ValueError, match="planning.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PLANNING,
                    messages=[user_text("start")],
                    pending_tool_calls=[pending_call],
                ),
                context=RuntimeContext(run_id="planning-pending"),
            )
        )

    with pytest.raises(ValueError, match="executing_tools.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.EXECUTING_TOOLS,
                    messages=[user_text("start")],
                ),
                context=RuntimeContext(run_id="executing-empty"),
            )
        )

    with pytest.raises(ValueError, match="matching tool messages"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PLANNING,
                    messages=[
                        user_text("start"),
                        Message.assistant([], tool_calls=[pending_call]),
                    ],
                ),
                context=RuntimeContext(run_id="planning-orphan-tool-call"),
            )
        )

    with pytest.raises(ValueError, match="preceding assistant tool_calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PLANNING,
                    messages=[
                        user_text("start"),
                        tool_text("orphan", tool_call_id="call-1"),
                    ],
                ),
                context=RuntimeContext(run_id="planning-orphan-tool-message"),
            )
        )

    with pytest.raises(ValueError, match="assistant tool_calls history"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.EXECUTING_TOOLS,
                    messages=[user_text("start")],
                    pending_tool_calls=[pending_call],
                ),
                context=RuntimeContext(run_id="executing-orphan-pending"),
            )
        )

    with pytest.raises(ValueError, match="contiguous tool messages"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.EXECUTING_TOOLS,
                    messages=[
                        user_text("start"),
                        Message.assistant([], tool_calls=[pending_call]),
                        user_text("interleaved"),
                    ],
                    pending_tool_calls=[pending_call],
                ),
                context=RuntimeContext(run_id="executing-interleaved-pending"),
            )
        )

    with pytest.raises(ValueError, match="tool call order"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PLANNING,
                    messages=[
                        user_text("start"),
                        Message.assistant(
                            [],
                            tool_calls=[pending_call, second_pending_call],
                        ),
                        tool_text("second", tool_call_id="call-2"),
                        tool_text("first", tool_call_id="call-1"),
                    ],
                ),
                context=RuntimeContext(run_id="planning-tool-message-order-mismatch"),
            )
        )

    with pytest.raises(ValueError, match="unresolved assistant tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.EXECUTING_TOOLS,
                    messages=[
                        user_text("start"),
                        Message.assistant(
                            [],
                            tool_calls=[pending_call, second_pending_call],
                        ),
                        tool_text("first", tool_call_id="call-1"),
                    ],
                    pending_tool_calls=[pending_call],
                ),
                context=RuntimeContext(run_id="executing-pending-suffix-mismatch"),
            )
        )

    with pytest.raises(ValueError, match="preceding assistant tool_calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.EXECUTING_TOOLS,
                    messages=[
                        user_text("start"),
                        tool_text("orphan", tool_call_id="orphan-call"),
                        Message.assistant([], tool_calls=[pending_call]),
                    ],
                    pending_tool_calls=[pending_call],
                ),
                context=RuntimeContext(run_id="executing-mixed-orphan-tool-message"),
            )
        )

    with pytest.raises(ValueError, match="resumes to planning.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[user_text("start")],
                    pending_tool_calls=[pending_call],
                    pause=PauseState(
                        reason="manual_pause",
                        resume_status=AgentStatus.PLANNING,
                    ),
                ),
                context=RuntimeContext(run_id="paused-planning-pending"),
            )
        )

    with pytest.raises(ValueError, match="contiguous tool messages"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[
                        user_text("start"),
                        Message.assistant([], tool_calls=[pending_call]),
                    ],
                    pause=PauseState(
                        reason="manual_pause",
                        resume_status=AgentStatus.PLANNING,
                    ),
                ),
                context=RuntimeContext(run_id="paused-planning-orphan-tool-call"),
            ),
            append_messages=[user_text("callback complete")],
        )

    with pytest.raises(ValueError, match="preceding assistant tool_calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[user_text("start")],
                    pause=PauseState(
                        reason="manual_pause",
                        resume_status=AgentStatus.PLANNING,
                    ),
                ),
                context=RuntimeContext(run_id="paused-planning-append-orphan-tool-message"),
            ),
            append_messages=[tool_text("orphan", tool_call_id="call-1")],
        )

    with pytest.raises(ValueError, match="resumes to executing_tools.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[user_text("start")],
                    pause=PauseState(
                        reason="external_wait",
                        resume_status=AgentStatus.EXECUTING_TOOLS,
                        source="tool",
                        wait_id="job-1",
                    ),
                ),
                context=RuntimeContext(run_id="paused-executing-empty"),
            )
        )

    with pytest.raises(ValueError, match="assistant tool_calls history"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[user_text("start")],
                    pending_tool_calls=[pending_call],
                    pause=PauseState(
                        reason="external_wait",
                        resume_status=AgentStatus.EXECUTING_TOOLS,
                        source="tool",
                        wait_id="job-1",
                    ),
                ),
                context=RuntimeContext(run_id="paused-executing-orphan-pending"),
            )
        )


@pytest.mark.asyncio
async def test_resume_input_rejects_unexpected_pause() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause", wait_id="pause-1")
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [user_text("start")],
        controller=controller,
    )

    with pytest.raises(ValueError, match="does not match"):
        ResumeInput(
            snapshot=paused.snapshot or raise_assertion(),
            expected_pause=PauseSelector(wait_id="other"),
        )


def test_resume_input_rejects_unknown_fields_and_empty_selector_text() -> None:
    snapshot = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("finish")]),
        context=RuntimeContext(run_id="snapshot-run"),
    )
    payload = ResumeInput(snapshot=snapshot).to_dict()
    payload["legacy"] = True

    with pytest.raises(ValueError, match="unknown"):
        ResumeInput.from_dict(payload)
    with pytest.raises(ValueError, match="reason"):
        PauseSelector(reason="")
    with pytest.raises(ValueError, match="source"):
        PauseSelector(source="")


@pytest.mark.asyncio
async def test_result_trace_replays_completed_run() -> None:
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run(
        [user_text("finish")]
    )

    assert result.trace is not None
    replay = replay_trace(result.trace)
    assert replay.valid is True
    assert replay.final_status is AgentStatus.COMPLETED
    assert [step.kind for step in rt(result).steps][-4:] == [
        TraceStepKinds.STATE_CHANGED,
        TraceStepKinds.CHECKPOINT,
        TraceStepKinds.FINAL,
        TraceStepKinds.RUN_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_resume_trace_starts_with_resume_step() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [user_text("start")],
        controller=controller,
    )

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
        ResumeInput(snapshot=paused.snapshot or raise_assertion())
    )

    assert result.trace is not None
    assert [step.kind for step in rt(result).steps[:2]] == [
        TraceStepKinds.RESUME,
        TraceStepKinds.RUN_STARTED,
    ]
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_transition_hook_cannot_mutate_live_state() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingTransitionHook()],
    ).run([user_text("hello")])

    assert result.status is AgentStatus.COMPLETED
    assert result.error is None


@pytest.mark.asyncio
async def test_after_model_hook_invalid_response_is_rejected() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[BadAfterModelHook()],
    ).run([user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "finish_reason" in (result.error or "")


@pytest.mark.asyncio
async def test_model_completed_event_is_not_emitted_if_after_model_result_is_invalid() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[BadModelResponseShapeHook()],
        ),
        [user_text("hello")],
    )

    assert EventTypes.MODEL_COMPLETED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == "failed"


@pytest.mark.asyncio
async def test_after_tool_hook_invalid_result_is_rejected() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})])]
    )
    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("echo"),
        hooks=[BadAfterToolHook()],
    ).run([user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "is_error" in (result.error or "")


@pytest.mark.asyncio
async def test_tool_completed_event_is_not_emitted_if_after_tool_result_is_invalid() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})])]
    )
    events = await collect_events(
        AgentLoop(
            model=model,
            tools=HarnessToolRegistry("echo"),
            hooks=[BadToolObservationShapeHook()],
        ),
        [user_text("hello")],
    )

    assert EventTypes.TOOL_COMPLETED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == "failed"


@pytest.mark.asyncio
async def test_before_tool_argument_rewrite_updates_assistant_history() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "original"})]
            ),
            ModelResponse.text("done"),
        ]
    )
    result = await AgentLoop(
        model=model,
        tools=HarnessToolRegistry("echo"),
        hooks=[ToolArgumentHook()],
    ).run([user_text("echo")])

    assistant_call = result.messages[1].tool_calls[0]
    assert assistant_call.arguments == {"text": "rewritten"}
    assert result.messages[2].text == "rewritten"


@pytest.mark.asyncio
async def test_before_tool_cannot_mutate_tool_call_identity_in_place() -> None:
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})]
                )
            ]
        ),
        tools=HarnessToolRegistry("echo"),
        hooks=[MutatingToolIdentityHook()],
    ).run([user_text("echo")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "before_tool cannot change tool call id, name, or mode"
    assert result.total_tool_calls == 0
    assert result.snapshot is not None
    assistant_call = next(
        message for message in result.messages if message.role == "assistant"
    ).tool_calls[0]
    assert assistant_call.id == "call-1"
    assert assistant_call.name == "echo"
    assert assistant_call.mode == "execute"
    assert [call.id for call in result.snapshot.state.pending_tool_calls] == ["call-1"]
    assert result.snapshot.state.pending_tool_calls[0].name == "echo"
    assert result.snapshot.state.pending_tool_calls[0].mode == "execute"
    assert all(message.role != "tool" for message in result.messages)


@pytest.mark.asyncio
async def test_event_replacement_cannot_change_core_runtime_events() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[ReplacingEventHook()],
        ),
        [user_text("hello")],
    )

    assert "renamed_model_started" not in [event.type for event in events]
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_hook_emitted_events_cannot_use_core_event_types() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[CoreEventEmittingHook()],
        ),
        [user_text("hello")],
    )

    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_custom_hook_events_do_not_pollute_runtime_trace() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RewritingHook()],
    ).run([user_text("hello")])

    assert result.trace is not None
    assert "custom_progress" not in [step.kind for step in rt(result).steps]
    model_call_events = [
        step.references["event_type"]
        for step in rt(result).steps
        if step.kind == TraceStepKinds.MODEL_CALL
    ]
    assert model_call_events == [EventTypes.MODEL_STARTED]
    model_result_events = [
        step.references["event_type"]
        for step in rt(result).steps
        if step.kind == TraceStepKinds.MODEL_RESULT
    ]
    assert model_result_events == [EventTypes.MODEL_COMPLETED]
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_event_order_is_stable() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
            ),
            ModelResponse.text("hello"),
        ]
    )
    events = await collect_events(
        AgentLoop(model=model, tools=HarnessToolRegistry("echo")),
        [user_text("echo")],
    )

    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.TOOL_STARTED,
        EventTypes.TOOL_COMPLETED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.FINAL,
        EventTypes.RUN_COMPLETED,
    ]
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert len({event.run_id for event in events}) == 1


@pytest.mark.asyncio
async def test_run_store_and_journal_record_checkpoint_identity() -> None:
    store = MemoryRunStore()
    journal = MemoryRunJournal()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=store,
        run_journal=journal,
    )

    result = await agent.run([user_text("hello")])

    assert result.status is AgentStatus.COMPLETED
    assert result.snapshot is not None
    assert len(store.checkpoints) == 1
    stored = store.checkpoints[0]
    assert stored.run_id == result.run_id
    assert stored.checkpoint_id == f"checkpoint-{result.snapshot.context.sequence}"
    assert stored.parent_checkpoint_id is None
    assert stored.snapshot.to_dict() == result.snapshot.to_dict()
    loaded_latest = await store.load_checkpoint(result.run_id)
    loaded_by_id = await store.load_checkpoint(result.run_id, stored.checkpoint_id)
    summaries = await store.list_checkpoints(result.run_id)
    assert loaded_latest.to_dict() == result.snapshot.to_dict()
    assert loaded_by_id.to_dict() == result.snapshot.to_dict()
    assert summaries == [stored.summary()]

    journal_types = [record.event_type for record in journal.records]
    assert journal_types == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
        EventTypes.CHECKPOINT,
        EventTypes.FINAL,
        EventTypes.RUN_COMPLETED,
    ]
    checkpoint_records = [
        record for record in journal.records if record.event_type == EventTypes.CHECKPOINT
    ]
    assert len(checkpoint_records) == 1
    assert checkpoint_records[0].checkpoint_id == stored.checkpoint_id
    read_records = [
        record async for record in journal.read(result.run_id, after_sequence=stored.sequence - 1)
    ]
    assert [record.sequence for record in read_records] == [
        stored.sequence,
        stored.sequence + 1,
        stored.sequence + 2,
    ]


@pytest.mark.asyncio
async def test_stored_checkpoint_rejects_snapshot_identity_mismatch() -> None:
    store = MemoryRunStore()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=store,
    ).run([user_text("hello")])
    assert result.snapshot is not None
    stored = store.checkpoints[0]

    with pytest.raises(ValueError, match="run_id must match"):
        StoredCheckpoint(
            run_id="other-run",
            checkpoint_id=stored.checkpoint_id,
            parent_checkpoint_id=stored.parent_checkpoint_id,
            sequence=stored.sequence,
            status=stored.status,
            snapshot=stored.snapshot,
            created_at=stored.created_at,
            metadata=stored.metadata,
        )
    with pytest.raises(ValueError, match="sequence must match"):
        StoredCheckpoint(
            run_id=stored.run_id,
            checkpoint_id=stored.checkpoint_id,
            parent_checkpoint_id=stored.parent_checkpoint_id,
            sequence=stored.sequence + 1,
            status=stored.status,
            snapshot=stored.snapshot,
            created_at=stored.created_at,
            metadata=stored.metadata,
        )
    with pytest.raises(ValueError, match="status must match"):
        StoredCheckpoint(
            run_id=stored.run_id,
            checkpoint_id=stored.checkpoint_id,
            parent_checkpoint_id=stored.parent_checkpoint_id,
            sequence=stored.sequence,
            status=AgentStatus.FAILED,
            snapshot=stored.snapshot,
            created_at=stored.created_at,
            metadata=stored.metadata,
        )


def test_extension_value_objects_round_trip_and_reject_unknown_fields() -> None:
    context = RuntimeContext(
        run_id="run-1",
        started_at=1.0,
        deadline=2.0,
        metadata={"tenant": "acme"},
    )
    call = ToolCall(id="call-1", name="record", arguments={"id": "a"})
    for decision in (
        ApprovalDecision.allow("safe", metadata={"policy": "test"}),
        ApprovalDecision.deny("blocked", metadata={"policy": "test"}),
        ApprovalDecision.pause("approval_required", metadata={"policy": "test"}),
    ):
        decision_payload = decision.to_dict()
        assert ApprovalDecision.from_dict(decision_payload).to_dict() == decision_payload
        with pytest.raises(ValueError, match="unknown"):
            ApprovalDecision.from_dict(decision_payload | {"legacy": True})
        with pytest.raises(KeyError):
            ApprovalDecision.from_dict(
                {key: payload for key, payload in decision_payload.items() if key != "metadata"}
            )
    with pytest.raises(ValueError, match="action"):
        ApprovalDecision.from_dict({"action": "later", "reason": "bad", "metadata": {}})
    with pytest.raises(KeyError):
        ApprovalDecision.from_dict({"action": "allow", "metadata": {}})

    spec = ToolSpec(
        name="record",
        description="Record.",
        input_schema={"type": "object", "properties": {}},
        annotations={"read_only": True},
    )
    request = ApprovalRequest(
        tool_call=call,
        context=context,
        tool_spec=spec,
        risk={"read_only": True},
        metadata={"policy": "test"},
    )
    request_payload = request.to_dict()
    assert ApprovalRequest.from_dict(request_payload).to_dict() == request_payload
    with pytest.raises(ValueError, match="unknown"):
        ApprovalRequest.from_dict(request_payload | {"legacy": True})
    for required_key in ("tool_spec", "risk", "metadata"):
        with pytest.raises(KeyError):
            ApprovalRequest.from_dict(
                {key: payload for key, payload in request_payload.items() if key != required_key}
            )

    snapshot = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hi")]),
        context=context,
    )
    summary = CheckpointSummary(
        run_id="run-1",
        checkpoint_id="checkpoint-1",
        parent_checkpoint_id=None,
        sequence=0,
        status=AgentStatus.PLANNING,
        created_at=3.0,
        metadata={"tier": "gold"},
    )
    summary_payload = summary.to_dict()
    assert CheckpointSummary.from_dict(summary_payload).to_dict() == summary_payload
    with pytest.raises(ValueError, match="unknown"):
        CheckpointSummary.from_dict(summary_payload | {"legacy": True})
    with pytest.raises(KeyError):
        CheckpointSummary.from_dict(
            {key: payload for key, payload in summary_payload.items() if key != "metadata"}
        )

    stored = StoredCheckpoint(
        run_id="run-1",
        checkpoint_id="checkpoint-0",
        parent_checkpoint_id=None,
        sequence=0,
        status=AgentStatus.PLANNING,
        snapshot=snapshot,
        created_at=3.0,
        metadata={"tier": "gold"},
    )
    stored_payload = stored.to_dict()
    assert StoredCheckpoint.from_dict(stored_payload).to_dict() == stored_payload
    with pytest.raises(ValueError, match="unknown"):
        StoredCheckpoint.from_dict(stored_payload | {"legacy": True})
    with pytest.raises(KeyError):
        StoredCheckpoint.from_dict(
            {key: payload for key, payload in stored_payload.items() if key != "metadata"}
        )

    event = AgentEvent(EventTypes.RUN_STARTED, {"state": snapshot.state.summary()}, run_id="run-1")
    record = JournalRecord(
        event=event,
        checkpoint_id=None,
        trace_step_id=1,
        payload_ref="blob://event",
        payload_hash="sha256:test",
        metadata={"sink": "memory"},
    )
    record_payload = record.to_dict()
    assert JournalRecord.from_dict(record_payload).to_dict() == record_payload
    with pytest.raises(ValueError, match="unknown"):
        JournalRecord.from_dict(record_payload | {"legacy": True})
    with pytest.raises(KeyError):
        JournalRecord.from_dict(
            {key: payload for key, payload in record_payload.items() if key != "metadata"}
        )
    with pytest.raises(ValueError, match="run_id must match"):
        JournalRecord.from_dict(record_payload | {"run_id": "other-run"})


def test_approval_request_nested_objects_are_defensive_copies() -> None:
    request = ApprovalRequest(
        tool_call=ToolCall(
            id="call-1",
            name="record",
            arguments={"nested": {"value": 1}},
        ),
        context=RuntimeContext(run_id="run-1", metadata={"tenant": "acme"}),
        risk={"risk": {"level": "low"}},
    )
    payload = request.to_dict()

    returned_call = request.tool_call
    cast(dict[str, Any], returned_call.arguments)["nested"] = {"value": 2}
    returned_context = request.context
    cast(dict[str, Any], returned_context.metadata)["tenant"] = "other"

    assert request.to_dict() == payload
    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, Any], request.risk)["risk"] = {"level": "high"}


@pytest.mark.asyncio
async def test_approval_denial_commits_tool_error_without_invoking_tool() -> None:
    tool = RecordingToolRegistry()
    policy = StaticApprovalPolicy(
        ApprovalDecision.deny("requires human approval", metadata={"policy": "test"})
    )
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "denied"})]
                ),
                ModelResponse.text("recovered"),
            ]
        ),
        tools=tool,
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("use tool")])]

    assert tool.calls == []
    assert len(policy.requests) == 1
    assert policy.requests[0].tool_call.id == "call-1"
    event_types = [event.type for event in events]
    assert event_types.index(EventTypes.APPROVAL_REQUESTED) < event_types.index(
        EventTypes.APPROVAL_COMPLETED
    )
    assert event_types.index(EventTypes.APPROVAL_COMPLETED) < event_types.index(
        EventTypes.TOOL_STARTED
    )
    tool_started = next(event for event in events if event.type == EventTypes.TOOL_STARTED)
    assert tool_started.data["implementation_invoked"] is False
    tool_completed = next(event for event in events if event.type == EventTypes.TOOL_COMPLETED)
    assert tool_completed.data["implementation_invoked"] is False
    assert tool_completed.data["result"]["is_error"] is True
    assert tool_completed.data["result"]["metadata"]["approval"] == "denied"
    checkpoint = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )
    tool_messages = [message for message in checkpoint.state.messages if message.role == "tool"]
    assert tool_messages
    assert tool_messages[0].metadata["is_error"] is True

    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_approval_allow_invokes_tool_implementation() -> None:
    tool = RecordingToolRegistry()
    policy = StaticApprovalPolicy(ApprovalDecision.allow("safe"))
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "allowed"})]
                ),
                ModelResponse.text("done"),
            ]
        ),
        tools=tool,
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("use tool")])]

    assert tool.calls == ["allowed"]
    tool_started = next(event for event in events if event.type == EventTypes.TOOL_STARTED)
    tool_completed = next(event for event in events if event.type == EventTypes.TOOL_COMPLETED)
    assert tool_started.data["implementation_invoked"] is True
    assert tool_completed.data["implementation_invoked"] is True


@pytest.mark.asyncio
async def test_approval_denial_of_accept_mode_commits_tool_rejection() -> None:
    tool = AcceptingWebSearchTool()
    policy = StaticApprovalPolicy(ApprovalDecision.deny("web access blocked"))
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="web_search",
                            mode="accept",
                            arguments={"query": "runtime"},
                        )
                    ]
                ),
                ModelResponse.text("done"),
            ]
        ),
        tools=ToolRegistry([tool]),
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("search")])]

    assert tool.accepted == []
    tool_completed = next(event for event in events if event.type == EventTypes.TOOL_COMPLETED)
    assert tool_completed.data["implementation_invoked"] is False
    assert tool_completed.data["result"]["result_kind"] == "rejection"
    assert tool_completed.data["result"]["is_error"] is True


@pytest.mark.asyncio
async def test_approval_denial_of_extension_mode_commits_tool_error_output() -> None:
    policy = StaticApprovalPolicy(ApprovalDecision.deny("handoff blocked"))
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="handoff",
                            mode="handoff",
                            arguments={"text": "delegate"},
                        )
                    ]
                ),
                ModelResponse.text("done"),
            ]
        ),
        tools=ToolRegistry([CustomHandoffTool()]),
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("handoff")])]

    tool_completed = next(event for event in events if event.type == EventTypes.TOOL_COMPLETED)
    assert tool_completed.data["implementation_invoked"] is False
    assert tool_completed.data["result"]["result_kind"] == "tool_error"
    assert tool_completed.data["result"]["is_error"] is True


@pytest.mark.asyncio
async def test_invalid_tool_call_is_not_sent_to_approval_policy() -> None:
    policy = StaticApprovalPolicy(ApprovalDecision.pause("approval_required"))
    result = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="missing", arguments={})]),
                ModelResponse.text("handled"),
            ]
        ),
        approval_policy=policy,
    ).run([user_text("use missing")])

    assert result.status is AgentStatus.COMPLETED
    assert policy.requests == []
    assert result.messages[-2].metadata["is_error"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "slow_event_type"),
    [
        (ApprovalDecision.allow("safe"), EventTypes.TOOL_STARTED),
        (ApprovalDecision.deny("blocked"), EventTypes.TOOL_STARTED),
        (ApprovalDecision.pause("approval_required"), EventTypes.PAUSE_REQUESTED),
    ],
)
async def test_approval_event_trace_replays_when_resolution_event_hook_times_out(
    decision: ApprovalDecision,
    slow_event_type: str,
) -> None:
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "x"})]
                )
            ]
        ),
        tools=RecordingToolRegistry(),
        approval_policy=StaticApprovalPolicy(decision),
        limits=LoopLimits(timeout_seconds=0.01),
        hooks=[SlowOnEventHook(slow_event_type)],
    )

    events = [event async for event in agent.run_events([user_text("use tool")])]

    event_types = [event.type for event in events]
    assert EventTypes.APPROVAL_REQUESTED in event_types
    assert EventTypes.APPROVAL_COMPLETED in event_types
    assert slow_event_type not in event_types
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).final_status is AgentStatus.LIMIT_EXCEEDED


@pytest.mark.asyncio
async def test_store_failure_does_not_journal_unemitted_state_change() -> None:
    journal = MemoryRunJournal()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=FailingRunStore(),
        run_journal=journal,
    )

    events: list[AgentEvent] = []
    with pytest.raises(RuntimeError, match="store unavailable"):
        async for event in agent.run_events([user_text("hello")]):
            events.append(event)

    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
    ]
    assert events[-1].data["to"] == AgentStatus.FAILED.value
    journal_types = [record.event_type for record in journal.records]
    assert journal_types == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
    ]
    assert journal.records[-1].event.data["to"] == AgentStatus.FAILED.value
    assert EventTypes.CHECKPOINT not in journal_types


@pytest.mark.asyncio
async def test_store_failure_does_not_dispatch_unpersisted_checkpoint_to_hooks() -> None:
    hook = RecordingEventHook()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=FailingRunStore(),
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        async for _event in agent.run_events([user_text("hello")]):
            pass

    assert EventTypes.CHECKPOINT not in hook.events


@pytest.mark.asyncio
async def test_store_failure_does_not_dispatch_uncheckpointed_state_change_to_hooks() -> None:
    hook = RecordingVisibilityHook()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=FailingRunStore(),
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        async for _event in agent.run_events([user_text("hello")]):
            pass

    assert EventTypes.STATE_CHANGED not in hook.seen
    assert "transition:planning->completed" not in hook.seen


@pytest.mark.asyncio
async def test_store_failure_does_not_dispatch_uncheckpointed_pause_request_to_hooks() -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    hook = RecordingVisibilityHook()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=FailingRunStore(),
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        async for _event in agent.run_events(
            [user_text("hello")],
            controller=controller,
        ):
            pass

    assert EventTypes.PAUSE_REQUESTED not in hook.seen
    assert EventTypes.STATE_CHANGED not in hook.seen
    assert EventTypes.CHECKPOINT not in hook.seen
    assert "transition:planning->paused" not in hook.seen


@pytest.mark.asyncio
async def test_store_failure_does_not_dispatch_uncheckpointed_insert_to_hooks() -> None:
    controller = RunController()
    controller.insert(ConversationInsert.text("external result", id="insert-1", source="test"))
    hook = RecordingVisibilityHook()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=FailingRunStore(),
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        async for _event in agent.run_events(
            [user_text("hello")],
            controller=controller,
        ):
            pass

    assert EventTypes.CONVERSATION_INSERTED not in hook.seen
    assert EventTypes.CHECKPOINT not in hook.seen


@pytest.mark.asyncio
@pytest.mark.parametrize("with_journal", [False, True])
async def test_run_store_pause_hook_order_matches_trace_order(with_journal: bool) -> None:
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    hook = RecordingVisibilityHook()
    store = MemoryRunStore()
    journal = MemoryRunJournal() if with_journal else None

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=store,
        run_journal=journal,
    ).run(
        [user_text("hello")],
        controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.PAUSED
    assert hook.seen.index(EventTypes.PAUSE_REQUESTED) < hook.seen.index(
        "transition:planning->paused"
    )
    assert hook.seen.index("transition:planning->paused") < hook.seen.index(
        EventTypes.STATE_CHANGED
    )
    assert hook.seen.index(EventTypes.STATE_CHANGED) < hook.seen.index(EventTypes.CHECKPOINT)
    if journal is not None:
        journal_types = [record.event_type for record in journal.records]
        assert journal_types.index(EventTypes.PAUSE_REQUESTED) < journal_types.index(
            EventTypes.STATE_CHANGED
        )
        assert journal_types.index(EventTypes.STATE_CHANGED) < journal_types.index(
            EventTypes.CHECKPOINT
        )


@pytest.mark.asyncio
async def test_run_store_journal_pause_global_visibility_order_is_consistent() -> None:
    timeline: list[str] = []
    controller = RunController()
    controller.request_pause(reason="manual_pause")
    journal = TimelineRunJournal(timeline)

    events: list[AgentEvent] = []
    async for event in AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[TimelineVisibilityHook(timeline)],
        run_store=MemoryRunStore(),
        run_journal=journal,
    ).run_events(
        [user_text("hello")],
        controller=controller,
    ):
        events.append(event)
        timeline.append(timeline_event_label("caller", event))

    assert events[-1].type == EventTypes.RUN_COMPLETED
    filtered = [
        item
        for item in timeline
        if item
        in {
            f"journal:{EventTypes.PAUSE_REQUESTED}",
            f"hook:{EventTypes.PAUSE_REQUESTED}",
            f"caller:{EventTypes.PAUSE_REQUESTED}",
            f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            "hook:transition:planning->paused",
            f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            f"journal:{EventTypes.CHECKPOINT}",
            f"hook:{EventTypes.CHECKPOINT}",
            f"caller:{EventTypes.CHECKPOINT}",
        }
    ]
    assert filtered == [
        f"journal:{EventTypes.PAUSE_REQUESTED}",
        f"hook:{EventTypes.PAUSE_REQUESTED}",
        f"caller:{EventTypes.PAUSE_REQUESTED}",
        f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        "hook:transition:planning->paused",
        f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        f"journal:{EventTypes.CHECKPOINT}",
        f"hook:{EventTypes.CHECKPOINT}",
        f"caller:{EventTypes.CHECKPOINT}",
    ]


@pytest.mark.asyncio
async def test_run_store_journal_pause_after_deferred_transition_order_is_consistent() -> None:
    timeline: list[str] = []
    controller = RunController()
    journal = TimelineRunJournal(timeline)

    events: list[AgentEvent] = []
    async for event in AgentLoop(
        model=SelfPausingToolCallModel(controller),
        tools=HarnessToolRegistry("echo"),
        hooks=[TimelineVisibilityHook(timeline)],
        run_store=MemoryRunStore(),
        run_journal=journal,
    ).run_events(
        [user_text("hello")],
        controller=controller,
    ):
        events.append(event)
        timeline.append(timeline_event_label("caller", event))

    assert events[-1].type == EventTypes.RUN_COMPLETED
    filtered = [
        item
        for item in timeline
        if item
        in {
            f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
            "hook:transition:planning->executing_tools",
            f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
            f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
            f"journal:{EventTypes.PAUSE_REQUESTED}",
            f"hook:{EventTypes.PAUSE_REQUESTED}",
            f"caller:{EventTypes.PAUSE_REQUESTED}",
            f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            "hook:transition:executing_tools->paused",
            f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
            f"journal:{EventTypes.CHECKPOINT}",
            f"hook:{EventTypes.CHECKPOINT}",
            f"caller:{EventTypes.CHECKPOINT}",
        }
    ]
    assert filtered == [
        f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
        "hook:transition:planning->executing_tools",
        f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
        f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.EXECUTING_TOOLS.value}",
        f"journal:{EventTypes.PAUSE_REQUESTED}",
        f"hook:{EventTypes.PAUSE_REQUESTED}",
        f"caller:{EventTypes.PAUSE_REQUESTED}",
        f"journal:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        "hook:transition:executing_tools->paused",
        f"hook:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        f"caller:{EventTypes.STATE_CHANGED}:{AgentStatus.PAUSED.value}",
        f"journal:{EventTypes.CHECKPOINT}",
        f"hook:{EventTypes.CHECKPOINT}",
        f"caller:{EventTypes.CHECKPOINT}",
    ]


@pytest.mark.asyncio
async def test_store_failure_after_tool_result_rolls_back_to_prior_checkpoint() -> None:
    store = FailingSecondCheckpointStore()
    events: list[AgentEvent] = []
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "job"})]
                )
            ]
        ),
        tools=RecordingToolRegistry(),
        run_store=store,
    )

    with pytest.raises(RuntimeError, match="store unavailable"):
        async for event in agent.run_events([user_text("use tool")]):
            events.append(event)

    assert len(store.checkpoints) == 1
    assert store.checkpoints[0].status is AgentStatus.EXECUTING_TOOLS
    assert [call.id for call in store.checkpoints[0].snapshot.state.pending_tool_calls] == [
        "call-1"
    ]
    assert EventTypes.TOOL_COMPLETED in [event.type for event in events]
    failed_state = next(
        event
        for event in reversed(events)
        if event.type == EventTypes.STATE_CHANGED and event.data["to"] == AgentStatus.FAILED.value
    )
    assert failed_state.data["total_tool_calls"] == 0
    checkpoints = [
        RunSnapshot.from_dict(event.data) for event in events if event.type == EventTypes.CHECKPOINT
    ]
    assert not any(snapshot.state.total_tool_calls == 1 for snapshot in checkpoints)


@pytest.mark.asyncio
async def test_journal_failure_prevents_checkpoint_delivery_after_store_save() -> None:
    store = MemoryRunStore()
    journal = FailingCheckpointJournal()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=store,
        run_journal=journal,
    )

    events: list[AgentEvent] = []
    with pytest.raises(RuntimeError, match="journal unavailable"):
        async for event in agent.run_events([user_text("hello")]):
            events.append(event)

    assert store.checkpoints
    assert [event.type for event in events] == [
        EventTypes.RUN_STARTED,
        EventTypes.MODEL_STARTED,
        EventTypes.MODEL_COMPLETED,
        EventTypes.STATE_CHANGED,
    ]
    assert EventTypes.CHECKPOINT not in [record.event_type for record in journal.records]


@pytest.mark.asyncio
async def test_journal_failure_does_not_dispatch_unjournaled_checkpoint_to_hooks() -> None:
    hook = RecordingEventHook()
    store = MemoryRunStore()
    journal = FailingCheckpointJournal()
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[hook],
        run_store=store,
        run_journal=journal,
    )

    with pytest.raises(RuntimeError, match="journal unavailable"):
        async for _event in agent.run_events([user_text("hello")]):
            pass

    assert store.checkpoints
    assert EventTypes.STATE_CHANGED in hook.events
    assert EventTypes.CHECKPOINT not in hook.events


@pytest.mark.asyncio
async def test_store_save_uses_runtime_deadline() -> None:
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_store=SlowRunStore(),
        limits=LoopLimits(timeout_seconds=0.01),
    )

    with pytest.raises(LimitExceeded, match=LimitReasons.TIMEOUT_SECONDS):
        await asyncio.wait_for(agent.run([user_text("hello")]), timeout=1)


@pytest.mark.asyncio
async def test_journal_append_uses_runtime_deadline() -> None:
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        run_journal=SlowRunJournal(),
        limits=LoopLimits(timeout_seconds=0.01),
    )

    with pytest.raises(LimitExceeded, match=LimitReasons.TIMEOUT_SECONDS):
        await asyncio.wait_for(agent.run([user_text("hello")]), timeout=1)


@pytest.mark.asyncio
async def test_approval_failure_journals_request_without_completion() -> None:
    journal = MemoryRunJournal()
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "boom"})]
                )
            ]
        ),
        tools=RecordingToolRegistry(),
        approval_policy=FailingApprovalPolicy(),
        run_journal=journal,
    )

    result = await agent.run([user_text("use tool")])

    assert result.status is AgentStatus.FAILED
    journal_types = [record.event_type for record in journal.records]
    assert EventTypes.APPROVAL_REQUESTED in journal_types
    assert EventTypes.APPROVAL_COMPLETED not in journal_types
    assert EventTypes.TOOL_STARTED not in journal_types
    request_index = journal_types.index(EventTypes.APPROVAL_REQUESTED)
    failed_index = next(
        index
        for index, record in enumerate(journal.records)
        if record.event_type == EventTypes.STATE_CHANGED
        and record.event.data["to"] == AgentStatus.FAILED.value
    )
    assert request_index < failed_index
    approval_request = next(
        record.event
        for record in journal.records
        if record.event_type == EventTypes.APPROVAL_REQUESTED
    )
    assert approval_request.data["id"] == "call-1"


@pytest.mark.asyncio
async def test_resume_checkpoint_store_records_parent_checkpoint_id() -> None:
    first = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="wait", arguments={"wait_id": "job-1"})]
                )
            ]
        ),
        tools=HarnessToolRegistry("wait"),
    ).run([user_text("start job")])
    assert first.snapshot is not None
    parent_id = f"checkpoint-{first.snapshot.context.sequence}"

    store = MemoryRunStore()
    resumed = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("resumed")]),
        tools=HarnessToolRegistry("wait"),
        run_store=store,
    ).run_snapshot(
        ResumeInput(
            snapshot=first.snapshot,
            append_messages=[user_text("job done")],
        )
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert store.checkpoints
    assert store.checkpoints[0].parent_checkpoint_id == parent_id


@pytest.mark.asyncio
async def test_approval_pause_stops_before_tool_execution_and_preserves_pending_call() -> None:
    tool = RecordingToolRegistry()
    policy = StaticApprovalPolicy(
        ApprovalDecision.pause("approval_required", metadata={"policy": "test"})
    )
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "waiting"})]
                )
            ]
        ),
        tools=tool,
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("use tool")])]

    assert tool.calls == []
    event_types = [event.type for event in events]
    assert EventTypes.TOOL_STARTED not in event_types
    assert event_types.index(EventTypes.APPROVAL_COMPLETED) < event_types.index(
        EventTypes.PAUSE_REQUESTED
    )
    paused = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )
    assert paused.state.status is AgentStatus.PAUSED
    assert paused.state.pause is not None
    assert paused.state.pause.source == "approval"
    assert paused.state.pause.wait_id == "call-1"
    assert [call.id for call in paused.state.pending_tool_calls] == ["call-1"]

    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_approval_pause_can_resume_and_execute_pending_call_once() -> None:
    tool = RecordingToolRegistry()
    policy = SequencedApprovalPolicy(
        [
            ApprovalDecision.pause("approval_required"),
            ApprovalDecision.allow("approved_after_resume"),
        ]
    )
    first = await AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "job"})]
                )
            ]
        ),
        tools=tool,
        approval_policy=policy,
    ).run([user_text("use tool")])

    assert first.status is AgentStatus.PAUSED
    assert first.snapshot is not None
    assert tool.calls == []
    parent_id = f"checkpoint-{first.snapshot.context.sequence}"

    store = MemoryRunStore()
    resumed = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        tools=tool,
        approval_policy=policy,
        run_store=store,
    ).run_snapshot(ResumeInput(snapshot=first.snapshot))

    assert resumed.status is AgentStatus.COMPLETED
    assert tool.calls == ["job"]
    assert len(policy.requests) == 2
    assert [request.tool_call.id for request in policy.requests] == ["call-1", "call-1"]
    assert store.checkpoints
    assert store.checkpoints[0].parent_checkpoint_id == parent_id
    assert resumed.trace is not None
    assert replay_trace(resumed.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_approval_pause_after_prior_parallel_call_has_only_applied_approval_events() -> None:
    tool = TimedTool("timed", parallel_safe=True)
    policy = ApprovalPolicyByCall(
        {
            "call-1": ApprovalDecision.allow("safe"),
            "call-2": ApprovalDecision.pause("approval_required"),
        }
    )
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(id="call-1", name="timed", arguments={"id": "allowed"}),
                        ToolCall(id="call-2", name="timed", arguments={"id": "paused"}),
                    ]
                )
            ]
        ),
        tools=ToolRegistry([tool]),
        limits=LoopLimits(max_parallel_tool_calls=2),
        approval_policy=policy,
    )

    events = [event async for event in agent.run_events([user_text("use tools")])]

    assert [entry[1] for entry in tool.timeline if entry[0] == "start"] == ["allowed"]
    assert [request.tool_call.id for request in policy.requests] == ["call-1", "call-2"]
    approval_events = [
        event
        for event in events
        if event.type in {EventTypes.APPROVAL_REQUESTED, EventTypes.APPROVAL_COMPLETED}
    ]
    assert [event.data["id"] for event in approval_events] == [
        "call-1",
        "call-1",
        "call-2",
        "call-2",
    ]
    assert [event.data["id"] for event in events if event.type == EventTypes.TOOL_STARTED] == [
        "call-1"
    ]
    paused = RunSnapshot.from_dict(
        [event for event in events if event.type == EventTypes.CHECKPOINT][-1].data
    )
    assert paused.state.total_tool_calls == 1
    assert [call.id for call in paused.state.pending_tool_calls] == ["call-2"]


@pytest.mark.asyncio
async def test_approval_policy_receives_normalized_risk_annotations() -> None:
    tool = RecordingToolRegistry(
        ToolSpec(
            name="record",
            description="Record with risk annotations.",
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
            annotations={
                "read_only": False,
                "risk": {
                    "filesystem": "write",
                    "network": "none",
                    "subprocess": True,
                    "destructive": False,
                    "requires_approval": True,
                },
            },
        )
    )

    policy = StaticApprovalPolicy(ApprovalDecision.allow("ok"))
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="record", arguments={"id": "a"})]
                ),
                ModelResponse.text("done"),
            ]
        ),
        tools=tool,
        approval_metadata={"surface": "cli"},
        approval_policy=policy,
    )

    result = await agent.run([user_text("use tool")])

    assert result.status is AgentStatus.COMPLETED
    assert policy.requests[0].risk == {
        "filesystem": "write",
        "network": "none",
        "subprocess": True,
        "destructive": False,
        "requires_approval": True,
    }
    assert policy.requests[0].metadata == {"surface": "cli"}


@pytest.mark.asyncio
async def test_tool_progress_and_cancel_request_are_live_events_and_trace_replays() -> None:
    class ProgressFixtureTool:
        spec = ToolSpec(
            name="progress",
            description="Emit progress and cooperatively observe cancellation.",
            input_schema={"type": "object", "properties": {}},
        )

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation
            context.emit_progress({"phase": "started"})
            for _ in range(100):
                if context.cancel_requested:
                    return ToolObservation.text("cancelled", is_error=True)
                await asyncio.sleep(0.001)
            return ToolObservation.text("finished")

    controller = RunController()
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="progress")]),
                ModelResponse.text("handled"),
            ]
        ),
        tools=ToolRegistry([ProgressFixtureTool()]),
    )

    events: list[AgentEvent] = []
    async for event in agent.run_events(
        [user_text("use progress")],
        controller=controller,
    ):
        events.append(event)
        if event.type == EventTypes.TOOL_PROGRESS:
            controller.cancel_tool("call-1", reason="test_cancel", metadata={"test": True})

    event_types = [event.type for event in events]
    assert EventTypes.TOOL_PROGRESS in event_types
    assert EventTypes.TOOL_CANCEL_REQUESTED in event_types
    cancel_event = next(event for event in events if event.type == EventTypes.TOOL_CANCEL_REQUESTED)
    assert cancel_event.data["reason"] == "test_cancel"
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_tool_progress_drains_queued_events_before_tool_completion() -> None:
    class BurstProgressTool:
        spec = ToolSpec(
            name="burst_progress",
            description="Emit several progress records before returning.",
            input_schema={"type": "object", "properties": {}},
        )

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation
            context.emit_progress({"step": 1})
            context.emit_progress({"step": 2})
            context.emit_progress({"step": 3})
            return ToolObservation.text("finished")

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel(
                [
                    ModelResponse(tool_calls=[ToolCall(id="call-1", name="burst_progress")]),
                    ModelResponse.text("done"),
                ]
            ),
            tools=ToolRegistry([BurstProgressTool()]),
        ).run_events([user_text("use burst progress")])
    ]

    progress_events = [
        event
        for event in events
        if event.type == EventTypes.TOOL_PROGRESS and event.data["id"] == "call-1"
    ]
    assert [event.data["progress"]["step"] for event in progress_events] == [1, 2, 3]
    completed_index = next(
        index
        for index, event in enumerate(events)
        if event.type == EventTypes.TOOL_COMPLETED and event.data["id"] == "call-1"
    )
    assert all(events.index(event) < completed_index for event in progress_events)
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_tool_cancel_request_is_cleared_after_tool_completion() -> None:
    class CancelRecordingTool:
        spec = ToolSpec(
            name="cancel_record",
            description="Record cancel state.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancel_seen: list[bool] = []

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation
            self.started.set()
            for _ in range(100):
                if context.cancel_requested:
                    self.cancel_seen.append(True)
                    return ToolObservation.text("cancelled", is_error=True)
                await asyncio.sleep(0.001)
            self.cancel_seen.append(False)
            return ToolObservation.text("not cancelled")

    tool = CancelRecordingTool()
    controller = RunController()
    agent = AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="cancel_record")]),
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="cancel_record")]),
                ModelResponse.text("done"),
            ]
        ),
        tools=ToolRegistry([tool]),
    )

    async def cancel_first_call() -> None:
        await tool.started.wait()
        controller.cancel_tool("call-1", reason="first_call_only")

    cancel_task = asyncio.create_task(cancel_first_call())
    result = await agent.run([user_text("use tool twice")], controller=controller)
    await cancel_task

    assert result.status is AgentStatus.COMPLETED
    assert tool.cancel_seen == [True, False]


@pytest.mark.asyncio
async def test_late_tool_cancel_does_not_affect_later_reused_call_id() -> None:
    class CancelRecordingTool:
        spec = ToolSpec(
            name="cancel_record",
            description="Record cancel state.",
            input_schema={"type": "object", "properties": {}},
        )

        def __init__(self) -> None:
            self.cancel_seen: list[bool] = []

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation
            self.cancel_seen.append(context.cancel_requested)
            return ToolObservation.text("done")

    tool = CancelRecordingTool()
    controller = RunController()
    events: list[AgentEvent] = []
    async for event in AgentLoop(
        model=ScriptedModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="cancel_record")]),
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="cancel_record")]),
                ModelResponse.text("done"),
            ]
        ),
        tools=ToolRegistry([tool]),
    ).run_events([user_text("use tool twice")], controller=controller):
        events.append(event)
        if (
            event.type == EventTypes.TOOL_COMPLETED
            and event.data["id"] == "call-1"
            and len(tool.cancel_seen) == 1
        ):
            controller.cancel_tool("call-1", reason="too_late")

    assert tool.cancel_seen == [False, False]
    assert EventTypes.TOOL_CANCEL_REQUESTED not in [event.type for event in events]
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_background_task_tool_result_emits_lifecycle_event_and_trace_replays() -> None:
    class BackgroundResearchTool:
        spec = ToolSpec(
            name="research",
            description="Start host-owned background research.",
            input_schema={"type": "object", "properties": {}},
        )

        async def execute(
            self, invocation: ToolInvocation, context: ToolExecutionContext
        ) -> ToolObservation:
            _ = invocation, context
            task = BackgroundTask(
                id="research-1",
                status="accepted",
                kind="research",
                correlation_id="call-1",
                metadata={"topic": "runtime"},
            )
            return ToolObservation.waiting(
                "research started",
                wait_id=task.id,
                reason="research_callback",
                background_task=task,
            )

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel(
                [ModelResponse(tool_calls=[ToolCall(id="call-1", name="research")])]
            ),
            tools=ToolRegistry([BackgroundResearchTool()]),
        ).run_events([user_text("research")])
    ]

    event_types = [event.type for event in events]
    assert event_types.index(EventTypes.TOOL_COMPLETED) < event_types.index(
        EventTypes.BACKGROUND_TASK_STARTED
    )
    task_event = next(event for event in events if event.type == EventTypes.BACKGROUND_TASK_STARTED)
    assert task_event.data["task"]["id"] == "research-1"
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_child_run_context_emits_relation_events_and_trace_replays() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )

    events = [
        event
        async for event in AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_events(
            [user_text("child")],
            context=context,
        )
    ]

    event_types = [event.type for event in events]
    assert EventTypes.CHILD_RUN_STARTED in event_types
    assert EventTypes.CHILD_RUN_COMPLETED in event_types
    started = next(event for event in events if event.type == EventTypes.CHILD_RUN_STARTED)
    completed = next(event for event in events if event.type == EventTypes.CHILD_RUN_COMPLETED)
    assert started.data == {
        "parent_run_id": "parent-run",
        "parent_tool_call_id": "call-1",
        "run_kind": "subagent",
    }
    assert completed.data == dict(started.data) | {"status": AgentStatus.COMPLETED.value}
    trace = RunTrace.from_events("child-run", events)
    assert replay_trace(trace).valid
    assert replay_trace(RunTrace.from_events(events[0].run_id, events)).valid


@pytest.mark.asyncio
async def test_child_run_completed_emits_on_terminal_error_tail() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.FINAL)],
        ).run_events(
            [user_text("child")],
            context=context,
        )
    ]

    event_types = [event.type for event in events]
    assert event_types.count(EventTypes.CHILD_RUN_STARTED) == 1
    assert event_types.count(EventTypes.CHILD_RUN_COMPLETED) == 1
    assert event_types.index(EventTypes.ERROR) < event_types.index(EventTypes.CHILD_RUN_COMPLETED)
    assert event_types.index(EventTypes.CHILD_RUN_COMPLETED) < event_types.index(
        EventTypes.RUN_COMPLETED
    )
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_child_run_completed_not_duplicated_when_run_completed_hook_fails() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.RUN_COMPLETED)],
        ).run_events(
            [user_text("child")],
            context=context,
        )
    ]

    event_types = [event.type for event in events]
    assert event_types.count(EventTypes.CHILD_RUN_STARTED) == 1
    assert event_types.count(EventTypes.CHILD_RUN_COMPLETED) == 1
    assert event_types[-2:] == [EventTypes.ERROR, EventTypes.RUN_COMPLETED]
    trace = RunTrace.from_events(events[0].run_id, events)
    assert replay_trace(trace).valid


@pytest.mark.asyncio
async def test_child_run_completed_does_not_orphan_result_trace_after_rollback() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )
    agent = AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.STATE_CHANGED)],
    )

    result = await agent.run([user_text("child")], context=context)

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert replay_trace(result.trace).valid
    result_kinds = [step.kind for step in rt(result).steps]
    assert TraceStepKinds.CHILD_RUN_STARTED not in result_kinds
    assert TraceStepKinds.CHILD_RUN_COMPLETED not in result_kinds

    events = [
        event
        async for event in AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.STATE_CHANGED)],
        ).run_events(
            [user_text("child")],
            context=context,
        )
    ]
    assert EventTypes.CHILD_RUN_STARTED in [event.type for event in events]
    assert EventTypes.CHILD_RUN_COMPLETED in [event.type for event in events]
    assert replay_trace(RunTrace.from_events(events[0].run_id, events)).valid
