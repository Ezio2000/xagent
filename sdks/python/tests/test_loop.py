from __future__ import annotations

import asyncio
import gc
import time as time_module
from collections.abc import AsyncIterator, Awaitable, Sequence
from time import monotonic
from time import time as wall_time
from typing import Any, cast

import pytest

import agent_runtime.loop as loop_module
from agent_runtime import (
    AgentEvent,
    AgentLoop,
    AgentState,
    AgentStatus,
    ContentPart,
    EventEmitter,
    EventTypes,
    LoopLimits,
    Message,
    ModelContentDelta,
    ModelErrorInfo,
    ModelOptions,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    PauseController,
    PauseRequest,
    PauseSelector,
    PauseState,
    ResumeInput,
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
    TraceStepKinds,
    replay_trace,
)
from agent_runtime.loop import RunControlState


class ScriptedModel:
    def __init__(self, steps: Sequence[ModelResponse]) -> None:
        self._steps = list(steps)
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        response = self._steps[self.calls]
        self.calls += 1
        return response


class RequestCapturingModel:
    def __init__(self) -> None:
        self.request: ModelRequest | None = None

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.request = request
        return ModelResponse.text("done")


class StreamingTextModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="hel")
        yield ModelContentDelta(index=0, text_delta="lo")


class StreamingToolModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = context
        self.calls += 1
        if self.calls == 1:
            yield ModelToolCallDelta(index=0, id="call-1", name="echo")
            yield ModelToolCallDelta(index=0, arguments_delta='{"text":')
            yield ModelToolCallDelta(index=0, arguments_delta='"hello"}')
            return
        assert request.messages[-1].role == "tool"
        yield ModelContentDelta(index=0, text_delta="done")


class StreamingToolThenSlowModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = context
        self.calls += 1
        if self.calls == 1:
            yield ModelToolCallDelta(index=0, id="call-1", name="echo")
            yield ModelToolCallDelta(index=0, arguments_delta='{"text":"hello"}')
            return
        assert request.messages[-1].role == "tool"
        yield ModelContentDelta(index=0, text_delta="partial")
        await asyncio.sleep(1)


class SlowStreamingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="partial")
        await asyncio.sleep(1)


class FastStreamingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="first")
        yield ModelContentDelta(index=0, text_delta="second")


class CloseTrackingStreamingModel:
    def __init__(self) -> None:
        self.next_chunk_started = asyncio.Event()
        self.closed = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        try:
            yield ModelContentDelta(index=0, text_delta="partial")
            self.next_chunk_started.set()
            await asyncio.sleep(1)
        finally:
            self.closed.set()


class ProviderErrorModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise ModelProviderError(
            ModelErrorInfo(
                message="provider unavailable",
                provider="test-provider",
                code="rate_limit",
                status_code=429,
                retryable=True,
                request_id="req-1",
            )
        )


class ClearingReadPauseController(PauseController):
    def __init__(self, request: PauseRequest) -> None:
        super().__init__()
        self._request_copy = request
        self.reads = 0

    @property
    def request(self) -> PauseRequest | None:
        self.reads += 1
        if self.reads == 1:
            return self._request_copy
        return None


class StreamingProviderErrorModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="partial")
        raise ModelProviderError(
            ModelErrorInfo(
                message="provider unavailable",
                provider="test-provider",
                code="rate_limit",
                status_code=429,
                retryable=True,
                request_id="req-1",
            )
        )


class RecordingTool:
    spec = ToolSpec(
        name="record",
        description="Record executed call ids.",
        input_schema={"type": "object", "properties": {}},
    )

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        call_id = str(arguments["id"])
        self.calls.append(call_id)
        return ToolResult.text(call_id)


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

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        call_id = str(arguments["id"])
        delay = float(arguments.get("delay", 0.01))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.timeline.append(("start", call_id, monotonic()))
        try:
            await asyncio.sleep(delay)
            return ToolResult.text(call_id)
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

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        call_id = str(arguments["id"])
        if call_id == "writer":
            cast(dict[str, Any], context.metadata)["leaked"] = "yes"
            self.writer_started.set()
            await asyncio.sleep(0.01)
            return ToolResult.text("writer")
        await self.writer_started.wait()
        return ToolResult.text(str(context.metadata.get("leaked", "missing")))


class MaybeErrorTimedTool(TimedTool):
    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        text = await super().execute(arguments, context)
        if arguments.get("error") is True:
            return ToolResult.text(text.text_content or "tool_error", is_error=True)
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

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        call_id = str(arguments["id"])
        self.started.append(call_id)
        if len(self.started) == 3:
            self._all_started.set()
        if call_id == "1":
            await asyncio.sleep(1)
        await self._all_started.wait()
        await asyncio.sleep(0)
        return ToolResult.text(call_id)


class StaggeredGappedTimeoutTool:
    spec = ToolSpec(
        name="staggered_gapped_timeout",
        description="Complete call 0 first, call 2 second, while call 1 times out.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        call_id = str(arguments["id"])
        if call_id == "1":
            await asyncio.sleep(1)
        if call_id == "2":
            await asyncio.sleep(0.01)
        return ToolResult.text(call_id)


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.text(str(arguments.get("text", "")))


class MetadataTool:
    spec = ToolSpec(
        name="metadata_tool",
        description="Return metadata-bearing content.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        return ToolResult(
            parts=[ContentPart.text_part("tool", metadata={"secret": "part"})],
            metadata={"secret": "result"},
        )


class WaitingTool:
    spec = ToolSpec(
        name="wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.waiting(
            "external job started",
            wait_id=str(arguments["wait_id"]),
            reason="external_callback",
        )


class ParallelWaitingTool:
    spec = ToolSpec(
        name="parallel_wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        await asyncio.sleep(float(arguments.get("delay", 0)))
        return ToolResult.waiting(
            str(arguments["wait_id"]),
            wait_id=str(arguments["wait_id"]),
            reason="external_callback",
        )


class FailTool:
    spec = ToolSpec(
        name="fail",
        description="Raise an error.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        raise RuntimeError("tool failed")


class SlowTool:
    spec = ToolSpec(
        name="slow",
        description="Sleep too long.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        await asyncio.sleep(1)
        return ToolResult.text("late")


class SlowModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        await asyncio.sleep(1)
        return ModelResponse.text("late")


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
    def __init__(self, controller: PauseController) -> None:
        self.controller = controller

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.controller.request_pause(reason="manual_pause")
        return ModelResponse(
            tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
        )


class AdapterTimeoutModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise TimeoutError("provider timeout")


class CancellationConvertingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError as exc:
            raise RuntimeError("provider converted cancellation") from exc
        return ModelResponse.text("late")


class CancellationSwallowingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
        return ModelResponse.text("late")


class CancellationSwallowingThenFailingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise RuntimeError("late provider failure") from None
        return ModelResponse.text("late")


class ExternallyCancelledModel:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        self.started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise RuntimeError("late provider failure") from None
        return ModelResponse.text("late")


class ContextInspectingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request
        return ModelResponse.text(str(context.metadata["tenant"]))


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


class ToolArgumentHook(RuntimeHook):
    def before_tool(self, call: ToolCall, context: RuntimeContext) -> ToolCall:
        _ = context
        return ToolCall(id=call.id, name=call.name, arguments={"text": "rewritten"})


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
    def after_tool(self, result: ToolResult, context: RuntimeContext) -> ToolResult:
        _ = context
        result.is_error = cast(Any, "yes")
        return result


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
    def __init__(self, event_type: str, controller: PauseController) -> None:
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
        controller: PauseController,
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


class BadModelResponseShapeHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        response.response_id = cast(Any, 123)
        return response


class BadToolResultShapeHook(RuntimeHook):
    def after_tool(self, result: ToolResult, context: RuntimeContext) -> ToolResult:
        _ = context
        result.is_error = cast(Any, "yes")
        return result


async def collect_events(
    agent: AgentLoop,
    messages: Sequence[Message],
    *,
    stream: bool = False,
    pause_controller: PauseController | None = None,
) -> list[AgentEvent]:
    return [
        event
        async for event in agent.run_events(
            messages,
            stream=stream,
            pause_controller=pause_controller,
        )
    ]


class PauseExposingLoop(AgentLoop):
    async def await_model_for_test(
        self,
        awaitable: Awaitable[ModelResponse],
        control: RunControlState,
    ) -> ModelResponse:
        return await self._await_model_with_interrupt(awaitable, control)

    async def apply_pause_for_test(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        request: PauseRequest,
    ) -> tuple[AgentEvent, ...]:
        return await self._pause(
            state,
            context,
            control,
            request,
            resume_status=AgentStatus.EXECUTING_TOOLS,
            origin="control",
        )


def parts_text(parts: Sequence[Any]) -> str:
    return "".join(part.text or "" for part in parts)


@pytest.mark.asyncio
async def test_model_request_includes_standard_options_and_tool_choice() -> None:
    model = RequestCapturingModel()

    result = await AgentLoop(
        model=model,
        model_options=ModelOptions(model="test-model", temperature=0.1),
        tool_choice=ToolChoice(mode="none", allow_parallel_tool_calls=False),
    ).run([Message.user_text("finish")])

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
        [Message.user_text("stream")],
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
    tool = EchoTool()
    events = await collect_events(
        AgentLoop(model=StreamingToolModel(), tools=[tool]),
        [Message.user_text("stream tool")],
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
async def test_stream_timeout_discards_partial_assistant_message() -> None:
    events = await collect_events(
        AgentLoop(
            model=SlowStreamingModel(),
            limits=LoopLimits(timeout_seconds=0.02),
        ),
        [Message.user_text("stream slow")],
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
    controller = PauseController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(model=SlowStreamingModel()).run_events(
        [Message.user_text("stream slow")],
        stream=True,
        pause_controller=controller,
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
    controller = PauseController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(
        model=SlowStreamingModel(),
        limits=LoopLimits(max_iterations=1),
    ).run_events(
        [Message.user_text("stream slow")],
        stream=True,
        pause_controller=controller,
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
    controller = PauseController()
    events: list[AgentEvent] = []

    async for event in AgentLoop(model=FastStreamingModel()).run_events(
        [Message.user_text("stream fast")],
        stream=True,
        pause_controller=controller,
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
    assert paused.state.messages == [Message.user_text("stream fast")]


@pytest.mark.asyncio
async def test_later_stream_timeout_trace_uses_last_checkpoint_as_partial_baseline() -> None:
    result = await AgentLoop(
        model=StreamingToolThenSlowModel(),
        tools=[EchoTool()],
        limits=LoopLimits(timeout_seconds=0.02),
    ).run(
        [Message.user_text("stream after tool")],
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
            [Message.user_text("stream")],
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
        [Message.user_text("stream")],
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
        [Message.user_text("stream flag")],
        stream=True,
    )

    assert EventTypes.MODEL_DELTA not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_model_provider_error_sets_failed_checkpoint_error() -> None:
    result = await AgentLoop(model=ProviderErrorModel()).run([Message.user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "provider unavailable"
    assert result.snapshot is not None
    assert result.snapshot.state.error == "provider unavailable"
    assert "error_details" not in result.snapshot.state.to_dict()


@pytest.mark.asyncio
async def test_stream_provider_error_discards_partial_assistant_message() -> None:
    events = await collect_events(
        AgentLoop(model=StreamingProviderErrorModel()),
        [Message.user_text("stream fail")],
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
async def test_model_direct_final() -> None:
    model = ScriptedModel([ModelResponse.text("done")])
    result = await AgentLoop(model=model).run([Message.user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"
    assert result.iterations == 1
    assert result.total_tool_calls == 0


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
    ).run([Message.user_text("finish")])

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
    result = await AgentLoop(model=model, tools=[MetadataTool()]).run([Message.user_text("tool")])

    tool_message = next(message for message in result.messages if message.role == "tool")
    assert tool_message.parts[0].metadata == {}
    assert tool_message.metadata == {}


@pytest.mark.asyncio
async def test_result_snapshot_is_last_durable_checkpoint() -> None:
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run(
        [Message.user_text("finish")]
    )

    assert result.snapshot is not None
    assert result.trace is not None
    checkpoint_step = next(
        step for step in reversed(result.trace.steps) if step.kind == TraceStepKinds.CHECKPOINT
    )
    completed_step = result.trace.steps[-1]
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
    result = await AgentLoop(model=model, tools=[EchoTool()]).run([Message.user_text("echo")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "hello"
    assert result.iterations == 2
    assert result.total_tool_calls == 1
    assert result.messages[-2].role == "tool"
    assert result.messages[-2].text == "hello"


@pytest.mark.asyncio
async def test_pause_controller_pauses_before_model_call_and_snapshot_resumes() -> None:
    controller = PauseController()
    controller.request_pause(reason="manual_pause")
    model = ScriptedModel([ModelResponse.text("done")])

    result = await AgentLoop(model=model).run(
        [Message.user_text("finish")],
        pause_controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert model.calls == 0
    assert controller.request is None
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
    controller = ClearingReadPauseController(PauseRequest(reason="manual_pause"))

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [Message.user_text("finish")],
        pause_controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert result.snapshot is not None
    assert result.snapshot.state.pause is not None
    assert result.snapshot.state.pause.reason == "manual_pause"
    assert controller.reads >= 2


@pytest.mark.asyncio
async def test_pause_during_tool_call_model_response_has_no_executing_checkpoint() -> None:
    controller = PauseController()
    events = await collect_events(
        AgentLoop(model=SelfPausingToolCallModel(controller), tools=[EchoTool()]),
        [Message.user_text("call tool")],
        pause_controller=controller,
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
        tools=[EchoTool()],
    ).run_snapshot(ResumeInput(snapshot=paused))

    assert resumed.status is AgentStatus.COMPLETED
    assert [message.text for message in resumed.messages if message.role == "tool"] == ["hello"]


def test_pause_payloads_reject_schema_invalid_types() -> None:
    controller = PauseController()

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


def test_pause_controller_exposes_defensive_request_copies() -> None:
    controller = PauseController()
    returned = controller.request_pause(metadata={"nested": {"value": 1}})
    cast(dict[str, Any], returned.metadata)["nested"]["value"] = 2

    stored = controller.request
    assert stored is not None
    assert stored.metadata == {"nested": {"value": 1}}

    cast(dict[str, Any], stored.metadata)["nested"]["value"] = 3

    stored_again = controller.request
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
    controller = PauseController()
    controller.request_pause(reason="manual_pause")
    model = ScriptedModel([ModelResponse.text("done")])
    now = wall_time()

    result = await AgentLoop(model=model).run(
        [Message.user_text("finish")],
        context=RuntimeContext(started_at=now - 2, deadline=now - 1),
        pause_controller=controller,
    )

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert model.calls == 0


@pytest.mark.asyncio
async def test_final_completion_wins_over_pause_requested_during_model_call() -> None:
    controller = PauseController()
    model = GateFinalModel()
    run_task = asyncio.create_task(
        AgentLoop(model=model).run(
            [Message.user_text("finish")],
            pause_controller=controller,
        )
    )
    await model.started.wait()

    controller.request_pause(reason="manual_pause")
    model.release.set()
    result = await run_task

    assert result.status is AgentStatus.COMPLETED
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert controller.request is None
    assert parts_text(result.final_parts) == "done"


@pytest.mark.asyncio
async def test_completed_model_task_wins_same_turn_interrupt_race() -> None:
    controller = PauseController()
    controller.interrupt(reason="race_interrupt")
    loop = PauseExposingLoop(model=ScriptedModel([]))
    future: asyncio.Future[ModelResponse] = asyncio.get_running_loop().create_future()
    future.set_result(ModelResponse.text("done"))

    response = await loop.await_model_for_test(
        future,
        RunControlState(
            run_id="race-run",
            started_at=wall_time(),
            pause_controller=controller,
        ),
    )

    assert parts_text(response.parts) == "done"


def test_pause_controller_can_be_reused_across_event_loops() -> None:
    class SlowFinalModel:
        async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
            _ = request, context
            await asyncio.sleep(0.01)
            return ModelResponse.text("done")

    async def run_once(controller: PauseController) -> AgentStatus:
        result = await AgentLoop(model=SlowFinalModel()).run(
            [Message.user_text("finish")],
            pause_controller=controller,
        )
        return result.status

    controller = PauseController()

    assert asyncio.run(run_once(controller)) is AgentStatus.COMPLETED
    assert asyncio.run(run_once(controller)) is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_interrupt_from_worker_thread_pauses_inflight_model_call() -> None:
    controller = PauseController()
    model = GateFinalModel()
    run_task = asyncio.create_task(
        AgentLoop(model=model).run(
            [Message.user_text("finish")],
            pause_controller=controller,
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
    result = await AgentLoop(model=model, tools=[EchoTool()]).run([Message.user_text("echo twice")])

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
        tools=[EchoTool()],
        limits=LoopLimits(max_iterations=2),
    ).run([Message.user_text("loop")])

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
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"


@pytest.mark.asyncio
async def test_tool_call_limit_takes_precedence_over_pause_after_model_response() -> None:
    controller = PauseController()
    events = await collect_events(
        AgentLoop(
            model=SelfPausingToolCallModel(controller),
            tools=[EchoTool()],
            limits=LoopLimits(max_total_tool_calls=0),
        ),
        [Message.user_text("echo")],
        pause_controller=controller,
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
    result = await AgentLoop(model=model, tools=[FailTool()]).run([Message.user_text("fail")])

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

    result = await AgentLoop(model=model, tools=[WaitingTool()]).run(
        [Message.user_text("start external work")]
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
    resumed = await AgentLoop(model=resumed_model, tools=[WaitingTool()]).run_snapshot(
        ResumeInput(snapshot=result.snapshot or raise_assertion())
    )

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
        AgentLoop(model=model, tools=[WaitingTool()]),
        [Message.user_text("start external work")],
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
    controller = PauseController()
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
            tools=[EchoTool()],
            hooks=[RequestPauseOnEventHook(EventTypes.TOOL_COMPLETED, controller)],
        ),
        [Message.user_text("run tool")],
        pause_controller=controller,
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
        tools=[ParallelWaitingTool()],
        limits=LoopLimits(max_parallel_tool_calls=2),
    ).run([Message.user_text("start external work")])

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
async def test_external_wait_pause_respects_expired_deadline() -> None:
    now = wall_time()
    context = RuntimeContext(run_id="expired-wait", started_at=now - 2, deadline=now - 1)
    control = RunControlState(
        run_id=context.run_id,
        started_at=context.started_at,
        deadline=context.deadline,
        monotonic_deadline=monotonic() - 1,
    )
    state = AgentState(
        status=AgentStatus.EXECUTING_TOOLS,
        messages=[Message.user_text("wait")],
    )

    events = await PauseExposingLoop(model=ScriptedModel([])).apply_pause_for_test(
        state,
        context,
        control,
        PauseRequest(
            reason="external_callback",
            source="tool",
            wait_id="job-1",
            metadata={},
        ),
    )

    assert state.status is AgentStatus.LIMIT_EXCEEDED
    assert state.error == "timeout_seconds"
    assert state.pause is None
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]


@pytest.mark.asyncio
async def test_stop_on_tool_error() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="fail", arguments={})])]
    )
    result = await AgentLoop(
        model=model,
        tools=[FailTool()],
        limits=LoopLimits(stop_on_tool_error=True),
    ).run([Message.user_text("fail")])

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
            tools=[FailTool()],
            limits=LoopLimits(stop_on_tool_error=True),
        ),
        [Message.user_text("fail")],
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
    result = await AgentLoop(model=model).run([Message.user_text("call missing")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "handled"
    assert "unknown tool" in result.messages[-2].text
    assert result.messages[-2].metadata["is_error"] is True


@pytest.mark.asyncio
async def test_model_timeout_is_hard_limit() -> None:
    result = await AgentLoop(
        model=SlowModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([Message.user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_adapter_timeout_error_is_failure_not_runtime_limit() -> None:
    result = await AgentLoop(model=AdapterTimeoutModel()).run([Message.user_text("slow")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "provider timeout"


@pytest.mark.asyncio
async def test_runtime_timeout_wins_over_converted_cancellation_error() -> None:
    result = await AgentLoop(
        model=CancellationConvertingModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([Message.user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"


@pytest.mark.asyncio
async def test_runtime_timeout_does_not_wait_for_swallowed_cancellation() -> None:
    started_at = monotonic()
    result = await AgentLoop(
        model=CancellationSwallowingModel(),
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([Message.user_text("slow")])
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
        ).run([Message.user_text("slow")])
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
        task = asyncio.create_task(AgentLoop(model=model).run([Message.user_text("cancel")]))
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
    ).run([Message.user_text("slow hook")])
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
        [Message.user_text("slow hook")],
    )

    assert [event.type for event in events] == [
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
        [Message.user_text("raise")],
    )

    assert [event.type for event in events] == [
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
        [Message.user_text("finish")],
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
    ).run([Message.user_text("finish")])

    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_model_completed_event_hook_failure_rolls_back_uncheckpointed_transition() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.MODEL_COMPLETED)],
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert [message.role for message in result.messages] == ["user"]
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED
    transitions = [step for step in result.trace.steps if step.kind == TraceStepKinds.STATE_CHANGED]
    assert transitions[-1].before_status is AgentStatus.PLANNING
    assert transitions[-1].after_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_started_event_hook_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.RUN_STARTED)],
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert result.trace.steps[0].kind == TraceStepKinds.RUN_STARTED
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_checkpoint_event_hook_failure_rolls_back_uncheckpointed_trace() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.CHECKPOINT)],
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED
    transitions = [step for step in result.trace.steps if step.kind == TraceStepKinds.STATE_CHANGED]
    assert transitions == [
        step for step in transitions if step.after_status is not AgentStatus.COMPLETED
    ]


@pytest.mark.asyncio
async def test_failed_run_completed_hook_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ProviderErrorModel(),
        hooks=[RaisingOnEventHook(EventTypes.RUN_COMPLETED)],
    ).run([Message.user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_checkpoint_custom_event_failure_rolls_back_trace_durability() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[FailingQueuedEventAfterHook(EventTypes.CHECKPOINT)],
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.FAILED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.FAILED
    assert result.snapshot is not None
    assert result.snapshot.state.status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_completed_custom_event_failure_trace_still_replays() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[FailingQueuedEventAfterHook(EventTypes.RUN_COMPLETED)],
    ).run([Message.user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_post_paused_event_hook_failure_does_not_rewrite_terminal_checkpoint() -> None:
    controller = PauseController()
    controller.request_pause(reason="manual_pause")
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.RUN_PAUSED)],
        ),
        [Message.user_text("finish")],
        pause_controller=controller,
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

    controller = PauseController()
    controller.request_pause(reason="manual_pause")
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RaisingOnEventHook(EventTypes.RUN_PAUSED)],
    ).run(
        [Message.user_text("finish")],
        pause_controller=controller,
    )

    assert result.trace is not None
    assert replay_trace(result.trace).final_status is AgentStatus.PAUSED


@pytest.mark.asyncio
async def test_pause_request_made_during_run_paused_is_cleared_at_invocation_end() -> None:
    controller = PauseController()
    controller.request_pause(reason="manual_pause")

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RequestPauseOnEventHook(EventTypes.RUN_PAUSED, controller)],
    ).run(
        [Message.user_text("finish")],
        pause_controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert controller.request is None


@pytest.mark.asyncio
async def test_pause_requested_event_hook_failure_clears_controller() -> None:
    controller = PauseController()
    controller.request_pause(reason="manual_pause")

    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[RaisingOnEventHook(EventTypes.PAUSE_REQUESTED)],
        ),
        [Message.user_text("finish")],
        pause_controller=controller,
    )

    assert controller.request is None
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_sync_blocking_hook_timeout_is_runtime_limit() -> None:
    started_at = monotonic()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=0.01),
        hooks=[BlockingAfterModelHook()],
    ).run([Message.user_text("blocking hook")])
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
        tools=[SlowTool()],
        limits=LoopLimits(timeout_seconds=0.01),
    ).run([Message.user_text("slow")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "timeout_seconds"
    assert result.total_tool_calls == 0


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
            tools=[EchoTool()],
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [Message.user_text("echo twice")],
    )

    assert events[-1].data["state"]["pending_tool_call_count"] == 1
    assert events[-1].data["state"]["total_tool_calls"] == 1


@pytest.mark.asyncio
async def test_model_response_checkpoint_rechecks_timeout_after_state_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = False

    def fake_monotonic() -> float:
        return 2.0 if expired else 0.0

    class ExpireAfterModelTransitionHook(RuntimeHook):
        def on_event(
            self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
        ) -> None:
            nonlocal expired
            _ = context, emitter
            if (
                event.type == EventTypes.STATE_CHANGED
                and event.data["from"] == AgentStatus.PLANNING.value
                and event.data["to"] == AgentStatus.EXECUTING_TOOLS.value
            ):
                expired = True

    monkeypatch.setattr(loop_module, "monotonic", fake_monotonic)
    now = wall_time()
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "late"})])]
    )

    events = [
        event
        async for event in AgentLoop(
            model=model,
            tools=[EchoTool()],
            hooks=[ExpireAfterModelTransitionHook()],
        ).run_events(
            [Message.user_text("echo")],
            context=RuntimeContext(started_at=now, deadline=now + 1),
        )
    ]

    checkpoint_statuses = [
        RunSnapshot.from_dict(event.data).state.status
        for event in events
        if event.type == EventTypes.CHECKPOINT
    ]
    assert AgentStatus.EXECUTING_TOOLS not in checkpoint_statuses
    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value
    assert events[-1].data["state"]["error"] == "timeout_seconds"


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
        tools=[WaitingTool(), EchoTool()],
        limits=LoopLimits(max_total_tool_calls=1),
    ).run([Message.user_text("wait then echo")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_total_tool_calls"
    assert result.total_tool_calls == 1
    assert result.snapshot is not None
    assert [call.id for call in result.snapshot.state.pending_tool_calls] == ["call-2"]
    assert result.snapshot.state.pause is None
    assert result.trace is not None
    assert TraceStepKinds.PAUSE_REQUESTED not in [step.kind for step in result.trace.steps]


@pytest.mark.asyncio
async def test_tool_call_limit_wins_over_pause_requested_after_tool_completed() -> None:
    controller = PauseController()
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
            tools=[EchoTool()],
            hooks=[RequestPauseOnEventHook(EventTypes.TOOL_COMPLETED, controller)],
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [Message.user_text("echo twice")],
        pause_controller=controller,
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
        tools=[WaitingTool()],
        limits=LoopLimits(max_iterations=1),
    ).run([Message.user_text("wait")])

    assert result.status is AgentStatus.LIMIT_EXCEEDED
    assert result.error == "max_iterations"
    assert result.snapshot is not None
    assert result.snapshot.state.pause is None
    assert result.trace is not None
    assert TraceStepKinds.PAUSE_REQUESTED not in [step.kind for step in result.trace.steps]


@pytest.mark.asyncio
async def test_tool_final_planning_boundary_applies_pause_before_checkpoint() -> None:
    controller = PauseController()
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
            tools=[EchoTool()],
            hooks=[
                RequestPauseOnStateChangeHook(
                    AgentStatus.EXECUTING_TOOLS,
                    AgentStatus.PLANNING,
                    controller,
                )
            ],
        ),
        [Message.user_text("echo")],
        pause_controller=controller,
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
    controller = PauseController()
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "done"})])]
    )

    events = await collect_events(
        AgentLoop(
            model=model,
            tools=[EchoTool()],
            hooks=[
                RequestPauseOnStateChangeHook(
                    AgentStatus.EXECUTING_TOOLS,
                    AgentStatus.PLANNING,
                    controller,
                )
            ],
            limits=LoopLimits(max_iterations=1),
        ),
        [Message.user_text("echo")],
        pause_controller=controller,
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

    result = await AgentLoop(model=model, tools=[tool]).run([Message.user_text("timed")])

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
        tools=[tool],
        limits=LoopLimits(max_parallel_tool_calls=4),
    ).run([Message.user_text("timed")])
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
        tools=[tool],
        limits=LoopLimits(max_parallel_tool_calls=2),
    ).run([Message.user_text("run")])
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
    agent = AgentLoop(model=model, tools=[EchoTool()])

    first_events = await collect_events(agent, [Message.user_text("first")])
    second_events = await collect_events(agent, [Message.user_text("second")])

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
            tools=[tool],
            limits=LoopLimits(max_parallel_tool_calls=2),
        ),
        [Message.user_text("timed")],
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
            tools=[tool],
            limits=LoopLimits(max_parallel_tool_calls=3),
        ),
        [Message.user_text("timed")],
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
            tools=[tool],
            limits=LoopLimits(max_parallel_tool_calls=2),
        ),
        [Message.user_text("timed")],
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
            tools=[tool],
            limits=LoopLimits(max_parallel_tool_calls=3),
        ),
        [Message.user_text("timed")],
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
            tools=[tool],
            limits=LoopLimits(timeout_seconds=0.02, max_parallel_tool_calls=3),
        ),
        [Message.user_text("timed")],
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
            tools=[tool],
            limits=LoopLimits(timeout_seconds=0.03, max_parallel_tool_calls=3),
        ),
        [Message.user_text("timed")],
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
        tools=[tool],
        limits=LoopLimits(max_parallel_tool_calls=4, stop_on_tool_error=True),
    ).run([Message.user_text("timed")])
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
        tools=[safe, unsafe],
        limits=LoopLimits(max_parallel_tool_calls=4),
    ).run([Message.user_text("timed")])

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
            tools=[tool],
            limits=LoopLimits(max_total_tool_calls=3, max_parallel_tool_calls=4),
        ),
        [Message.user_text("timed")],
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
            tools=[EchoTool()],
            limits=LoopLimits(max_total_tool_calls=1),
        ),
        [Message.user_text("echo twice")],
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
        AgentLoop(model=model, tools=[RecordingTool()]),
        [Message.user_text("record twice")],
    )
    first_checkpoint = next(event for event in events if event.type == EventTypes.CHECKPOINT)
    snapshot = RunSnapshot.from_dict(first_checkpoint.data)
    resumed_model = ScriptedModel([ModelResponse.text("resumed done")])
    resumed_tool = RecordingTool()

    result = await AgentLoop(model=resumed_model, tools=[resumed_tool]).run_snapshot(
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
        [Message.user_text("finish")],
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
    ).run([Message.user_text("hello")])

    assert parts_text(result.final_parts) == "hooked"
    assert hook.events[0] == EventTypes.RUN_STARTED
    assert EventTypes.FINAL in hook.events


@pytest.mark.asyncio
async def test_hooks_can_emit_custom_events() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")]), hooks=[RewritingHook()]),
        [Message.user_text("hello")],
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
        [Message.user_text("hello")],
    )
    result = await AgentLoop(model=ContextInspectingModel()).run(
        [Message.user_text("hello")],
        context=context,
    )

    assert parts_text(result.final_parts) == "acme"
    assert result.run_id == "run-test"
    assert result.snapshot is not None
    assert result.snapshot.context.run_id == "run-test"
    assert result.snapshot.context.metadata == {"tenant": "acme"}
    assert all(event.run_id for event in events)


@pytest.mark.asyncio
async def test_runtime_context_uses_wall_clock_checkpoint_deadline() -> None:
    started_at = wall_time()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=10),
    ).run([Message.user_text("hello")])

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
    events = [
        event async for event in agent.run_events([Message.user_text("hello")], context=context)
    ]

    assert [event.run_id for event in events] == ["stable-run"] * len(events)
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert context.run_id == "stable-run"


@pytest.mark.asyncio
async def test_result_uses_runtime_control_identity_after_context_mutation() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingEventContextHook()],
    ).run([Message.user_text("hello")], context=RuntimeContext(run_id="stable-run"))

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
            messages=[Message.user_text("finish")],
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
        state=AgentState(status=AgentStatus.PLANNING, messages=[Message.user_text("finish")]),
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
    controller = PauseController()
    controller.request_pause(
        reason="external_callback",
        source="tool",
        wait_id="job-1",
        metadata={"tenant": "acme"},
    )
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [Message.user_text("start")],
        pause_controller=controller,
    )
    model = RequestCapturingModel()

    result = await AgentLoop(model=model).run_snapshot(
        ResumeInput(
            snapshot=paused.snapshot or raise_assertion(),
            append_messages=[Message.user_text("callback complete")],
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
    controller = PauseController()
    controller.request_pause(reason="manual_pause", source="tool", wait_id="job-1")

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [Message.user_text("start")],
        pause_controller=controller,
    )

    assert result.status is AgentStatus.PAUSED
    assert result.trace is not None
    pause_steps = [
        step for step in result.trace.steps if step.kind == TraceStepKinds.PAUSE_REQUESTED
    ]
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
        tools=[WaitingTool(), EchoTool()],
    ).run([Message.user_text("start")])

    assert paused.snapshot is not None
    assert paused.snapshot.state.status is AgentStatus.PAUSED
    assert paused.snapshot.state.pause is not None
    assert paused.snapshot.state.pause.resume_status is AgentStatus.EXECUTING_TOOLS
    with pytest.raises(ValueError, match="resumes to planning"):
        ResumeInput(
            snapshot=paused.snapshot,
            append_messages=[Message.user_text("callback complete")],
        )


def test_resume_input_rejects_inconsistent_pending_tool_snapshots() -> None:
    pending_call = ToolCall(id="call-1", name="echo", arguments={})
    second_pending_call = ToolCall(id="call-2", name="echo", arguments={})

    with pytest.raises(ValueError, match="planning.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PLANNING,
                    messages=[Message.user_text("start")],
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
                    messages=[Message.user_text("start")],
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
                        Message.user_text("start"),
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
                        Message.user_text("start"),
                        Message.tool_text("orphan", tool_call_id="call-1"),
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
                    messages=[Message.user_text("start")],
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
                        Message.user_text("start"),
                        Message.assistant([], tool_calls=[pending_call]),
                        Message.user_text("interleaved"),
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
                        Message.user_text("start"),
                        Message.assistant(
                            [],
                            tool_calls=[pending_call, second_pending_call],
                        ),
                        Message.tool_text("second", tool_call_id="call-2"),
                        Message.tool_text("first", tool_call_id="call-1"),
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
                        Message.user_text("start"),
                        Message.assistant(
                            [],
                            tool_calls=[pending_call, second_pending_call],
                        ),
                        Message.tool_text("first", tool_call_id="call-1"),
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
                        Message.user_text("start"),
                        Message.tool_text("orphan", tool_call_id="orphan-call"),
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
                    messages=[Message.user_text("start")],
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
                        Message.user_text("start"),
                        Message.assistant([], tool_calls=[pending_call]),
                    ],
                    pause=PauseState(
                        reason="manual_pause",
                        resume_status=AgentStatus.PLANNING,
                    ),
                ),
                context=RuntimeContext(run_id="paused-planning-orphan-tool-call"),
            ),
            append_messages=[Message.user_text("callback complete")],
        )

    with pytest.raises(ValueError, match="preceding assistant tool_calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[Message.user_text("start")],
                    pause=PauseState(
                        reason="manual_pause",
                        resume_status=AgentStatus.PLANNING,
                    ),
                ),
                context=RuntimeContext(run_id="paused-planning-append-orphan-tool-message"),
            ),
            append_messages=[Message.tool_text("orphan", tool_call_id="call-1")],
        )

    with pytest.raises(ValueError, match="resumes to executing_tools.*pending tool calls"):
        ResumeInput(
            snapshot=RunSnapshot(
                state=AgentState(
                    status=AgentStatus.PAUSED,
                    messages=[Message.user_text("start")],
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
                    messages=[Message.user_text("start")],
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
    controller = PauseController()
    controller.request_pause(reason="manual_pause", wait_id="pause-1")
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [Message.user_text("start")],
        pause_controller=controller,
    )

    with pytest.raises(ValueError, match="does not match"):
        ResumeInput(
            snapshot=paused.snapshot or raise_assertion(),
            expected_pause=PauseSelector(wait_id="other"),
        )


def test_resume_input_rejects_unknown_fields_and_empty_selector_text() -> None:
    snapshot = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[Message.user_text("finish")]),
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
        [Message.user_text("finish")]
    )

    assert result.trace is not None
    replay = replay_trace(result.trace)
    assert replay.valid is True
    assert replay.final_status is AgentStatus.COMPLETED
    assert [step.kind for step in result.trace.steps][-4:] == [
        TraceStepKinds.STATE_CHANGED,
        TraceStepKinds.CHECKPOINT,
        TraceStepKinds.FINAL,
        TraceStepKinds.RUN_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_resume_trace_starts_with_resume_step() -> None:
    controller = PauseController()
    controller.request_pause(reason="manual_pause")
    paused = await AgentLoop(model=ScriptedModel([ModelResponse.text("unused")])).run(
        [Message.user_text("start")],
        pause_controller=controller,
    )

    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
        ResumeInput(snapshot=paused.snapshot or raise_assertion())
    )

    assert result.trace is not None
    assert [step.kind for step in result.trace.steps[:2]] == [
        TraceStepKinds.RESUME,
        TraceStepKinds.RUN_STARTED,
    ]
    assert replay_trace(result.trace).final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_transition_hook_cannot_mutate_live_state() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingTransitionHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.COMPLETED
    assert result.error is None


@pytest.mark.asyncio
async def test_after_model_hook_invalid_response_is_rejected() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[BadAfterModelHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "finish_reason" in (result.error or "")


@pytest.mark.asyncio
async def test_model_completed_event_is_not_emitted_if_after_model_result_is_invalid() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[BadModelResponseShapeHook()],
        ),
        [Message.user_text("hello")],
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
        tools=[EchoTool()],
        hooks=[BadAfterToolHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "is_error" in (result.error or "")


@pytest.mark.asyncio
async def test_tool_completed_event_is_not_emitted_if_after_tool_result_is_invalid() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})])]
    )
    events = await collect_events(
        AgentLoop(model=model, tools=[EchoTool()], hooks=[BadToolResultShapeHook()]),
        [Message.user_text("hello")],
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
        tools=[EchoTool()],
        hooks=[ToolArgumentHook()],
    ).run([Message.user_text("echo")])

    assistant_call = result.messages[1].tool_calls[0]
    assert assistant_call.arguments == {"text": "rewritten"}
    assert result.messages[2].text == "rewritten"


@pytest.mark.asyncio
async def test_event_replacement_cannot_change_core_runtime_events() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[ReplacingEventHook()],
        ),
        [Message.user_text("hello")],
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
        [Message.user_text("hello")],
    )

    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == AgentStatus.FAILED.value


@pytest.mark.asyncio
async def test_custom_hook_events_do_not_pollute_runtime_trace() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RewritingHook()],
    ).run([Message.user_text("hello")])

    assert result.trace is not None
    assert "custom_progress" not in [step.kind for step in result.trace.steps]
    model_call_events = [
        step.references["event_type"]
        for step in result.trace.steps
        if step.kind == TraceStepKinds.MODEL_CALL
    ]
    assert model_call_events == [EventTypes.MODEL_STARTED]
    model_result_events = [
        step.references["event_type"]
        for step in result.trace.steps
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
        AgentLoop(model=model, tools=[EchoTool()]),
        [Message.user_text("echo")],
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
