from __future__ import annotations

import asyncio
import gc
import time as time_module
from collections.abc import AsyncIterator, Sequence
from time import monotonic
from time import time as wall_time
from typing import Any

import pytest

from agent_runtime import (
    AgentEvent,
    AgentLoop,
    AgentState,
    AgentStatus,
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
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    ToolCall,
    ToolChoice,
    ToolResult,
    ToolSpec,
)


class ScriptedModel:
    def __init__(self, steps: Sequence[ModelResponse]) -> None:
        self._steps = list(steps)
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if self.calls >= len(self._steps):
            return ModelResponse.text("fallback")
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


class SlowStreamingModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("stream path should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="partial")
        await asyncio.sleep(1)


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
        response.extra = {"finish_reason": "shadow"}
        return response


class BadAfterToolHook(RuntimeHook):
    def after_tool(self, result: ToolResult, context: RuntimeContext) -> ToolResult:
        _ = context
        result.extra = {"is_error": True}
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


class BlockingAfterModelHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        time_module.sleep(0.05)
        return response


class BadModelResponseExtraHook(RuntimeHook):
    def after_model(self, response: ModelResponse, context: RuntimeContext) -> ModelResponse:
        _ = context
        response.extra = {"role": "user"}
        return response


class BadToolResultExtraHook(RuntimeHook):
    def after_tool(self, result: ToolResult, context: RuntimeContext) -> ToolResult:
        _ = context
        result.extra = {"role": "assistant"}
        return result


async def collect_events(
    agent: AgentLoop, messages: Sequence[Message], *, stream: bool = False
) -> list[AgentEvent]:
    return [event async for event in agent.run_events(messages, stream=stream)]


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
async def test_stream_flag_falls_back_to_complete_for_non_streaming_model() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")])),
        [Message.user_text("stream flag")],
        stream=True,
    )

    assert EventTypes.MODEL_DELTA not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == AgentStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_model_provider_error_is_structured_in_failed_checkpoint() -> None:
    result = await AgentLoop(model=ProviderErrorModel()).run([Message.user_text("fail")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "provider unavailable"
    assert result.state is not None
    assert result.state.extra["model_error"]["provider"] == "test-provider"
    assert result.state.extra["model_error"]["code"] == "rate_limit"
    assert result.state.extra["model_error"]["retryable"] is True


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
    assert checkpoints[-1].state.extra["model_error"]["provider"] == "test-provider"
    assert checkpoints[-1].state.extra["model_error"]["retryable"] is True


@pytest.mark.asyncio
async def test_model_direct_final() -> None:
    model = ScriptedModel([ModelResponse.text("done")])
    result = await AgentLoop(model=model).run([Message.user_text("finish")])

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"
    assert result.iterations == 1
    assert result.total_tool_calls == 0


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
    tool_snapshots = [
        RunSnapshot.from_dict(event.data)
        for event in events
        if event.type == EventTypes.CHECKPOINT
        and RunSnapshot.from_dict(event.data).state.status is AgentStatus.EXECUTING_TOOLS
        and RunSnapshot.from_dict(event.data).state.total_tool_calls > 0
    ]

    assert tool_snapshots
    assert tool_snapshots[0].state.total_tool_calls == 2
    assert tool_snapshots[0].state.pending_tool_calls == []
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

    result = await AgentLoop(model=resumed_model, tools=[resumed_tool]).run_snapshot(snapshot)

    assert snapshot.state.status is AgentStatus.EXECUTING_TOOLS
    assert resumed_tool.calls == ["call-1", "call-2"]
    assert resumed_model.calls == 1
    assert result.status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_model_final_checkpoint_resume_does_not_recall_model() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("first-final")])),
        [Message.user_text("finish")],
    )
    first_checkpoint = next(event for event in events if event.type == EventTypes.CHECKPOINT)
    snapshot = RunSnapshot.from_dict(first_checkpoint.data)
    resumed_model = ScriptedModel([ModelResponse.text("second-final")])

    result = await AgentLoop(model=resumed_model).run_snapshot(snapshot)

    assert snapshot.state.status is AgentStatus.COMPLETED
    assert resumed_model.calls == 0
    assert parts_text(result.final_parts) == "first-final"


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
        extra={"trace": {"id": "trace-1"}},
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
    assert result.context is not None
    assert result.context.run_id == "run-test"
    assert result.context.extra == {"trace": {"id": "trace-1"}}
    assert all(event.run_id for event in events)


@pytest.mark.asyncio
async def test_runtime_context_uses_wall_clock_checkpoint_deadline() -> None:
    started_at = wall_time()
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        limits=LoopLimits(timeout_seconds=10),
    ).run([Message.user_text("hello")])

    assert result.context is not None
    assert result.context.started_at >= started_at
    assert result.context.deadline is not None
    assert result.context.deadline > wall_time()
    assert result.context.deadline - result.context.started_at <= 10.1


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
    assert result.context is not None
    assert result.context.run_id == "stable-run"
    assert result.context.sequence > 0


@pytest.mark.asyncio
async def test_state_round_trip_and_resume() -> None:
    state = AgentState(
        status=AgentStatus.PLANNING,
        messages=[Message.user_text("finish")],
        extra={"checkpoint": {"owner": "host"}},
    )
    restored = AgentState.from_dict(state.to_dict())
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_state(restored)

    assert result.status is AgentStatus.COMPLETED
    assert parts_text(result.final_parts) == "done"
    assert result.state is not None
    assert result.state.to_dict()["status"] == "completed"
    assert result.state.extra == {"checkpoint": {"owner": "host"}}


@pytest.mark.asyncio
async def test_run_snapshot_round_trip_and_resume() -> None:
    snapshot = RunSnapshot(
        state=AgentState(
            status=AgentStatus.PLANNING,
            messages=[Message.user_text("finish")],
            extra={"checkpoint": {"owner": "host"}},
        ),
        context=RuntimeContext(run_id="snapshot-run", metadata={"tenant": "acme"}),
    )
    restored = RunSnapshot.from_dict(snapshot.to_dict())
    result = await AgentLoop(model=ScriptedModel([ModelResponse.text("done")])).run_snapshot(
        restored
    )

    assert result.status is AgentStatus.COMPLETED
    assert result.run_id == "snapshot-run"
    assert result.snapshot is not None
    assert result.snapshot.context.run_id == "snapshot-run"
    assert result.snapshot.state.extra == {"checkpoint": {"owner": "host"}}


@pytest.mark.asyncio
async def test_transition_hook_cannot_mutate_live_state() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[MutatingTransitionHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.COMPLETED
    assert result.error is None


@pytest.mark.asyncio
async def test_after_model_hook_reserved_extra_is_rejected() -> None:
    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[BadAfterModelHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "reserved" in (result.error or "")


@pytest.mark.asyncio
async def test_model_completed_event_is_not_emitted_if_assistant_message_commit_fails() -> None:
    events = await collect_events(
        AgentLoop(
            model=ScriptedModel([ModelResponse.text("done")]),
            hooks=[BadModelResponseExtraHook()],
        ),
        [Message.user_text("hello")],
    )

    assert EventTypes.MODEL_COMPLETED not in [event.type for event in events]
    assert events[-1].data["state"]["status"] == "failed"


@pytest.mark.asyncio
async def test_after_tool_hook_reserved_extra_is_rejected() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})])]
    )
    result = await AgentLoop(
        model=model,
        tools=[EchoTool()],
        hooks=[BadAfterToolHook()],
    ).run([Message.user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert "reserved" in (result.error or "")


@pytest.mark.asyncio
async def test_tool_completed_event_is_not_emitted_if_tool_message_commit_fails() -> None:
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "x"})])]
    )
    events = await collect_events(
        AgentLoop(model=model, tools=[EchoTool()], hooks=[BadToolResultExtraHook()]),
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
async def test_event_replacement_cannot_change_runtime_envelope() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")]), hooks=[ReplacingEventHook()]),
        [Message.user_text("hello")],
    )
    renamed = next(event for event in events if event.type == "renamed_model_started")

    assert renamed.run_id != "bad"
    assert renamed.sequence == 2


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
        EventTypes.CHECKPOINT,
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
