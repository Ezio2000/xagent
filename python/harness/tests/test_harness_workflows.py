from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from harness import AgentHarness, PausedHarnessRun, normalize_messages, waiting_from_events
from kernel import (
    AgentStatus,
    BackgroundTask,
    EventTypes,
    Message,
    ModelCapabilities,
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    PauseSelector,
    RuntimeContext,
    ToolCall,
    ToolObservation,
    ToolSpec,
)
from toolkit import ToolExecutionContext, ToolInvocation, ToolRegistry


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return the input text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.text(str(invocation.arguments["text"]))


class ToolLoopModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.calls += 1
        if self.calls == 1:
            assert request.messages[-1].text == "say hello"
            return ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "hello"})]
            )
        return ModelResponse.text(f"tool said: {request.messages[-1].text}")


class StreamingModel:
    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        raise AssertionError("streaming run should not call complete")

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        yield ModelContentDelta(index=0, text_delta="hel")
        yield ModelContentDelta(index=0, text_delta="lo")


class ExternalWaitTool:
    spec = ToolSpec(
        name="external_wait",
        description="Start external work and pause until a callback arrives.",
        input_schema={
            "type": "object",
            "properties": {"wait_id": {"type": "string"}},
            "required": ["wait_id"],
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.waiting(
            "external work started",
            wait_id=str(invocation.arguments["wait_id"]),
            reason="external_callback",
            pause_metadata={"example": "harness"},
            background_task=BackgroundTask(id="task-1", status="running"),
        )


class BackgroundOnlyTool:
    spec = ToolSpec(
        name="background_only",
        description="Return a background task without pausing.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        return ToolObservation.text(
            "background accepted",
            background_task=BackgroundTask(id="task-2", status="accepted"),
        )


class PauseResumeModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        if not any(message.role == "tool" for message in request.messages):
            return ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="external_wait",
                        arguments={"wait_id": "job-1"},
                    )
                ]
            )
        return ModelResponse.text(f"resumed: {request.messages[-1].text}")


class BackgroundOnlyModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="background_only", arguments={}),
                ]
            )
        assert request.messages[-1].role == "tool"
        return ModelResponse.text("done")


class TextModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        return ModelResponse.text(f"done: {request.messages[-1].text}")


@pytest.mark.asyncio
async def test_agent_harness_runs_tool_loop_with_tool_sequence() -> None:
    model = ToolLoopModel()
    harness = AgentHarness(model=model, tools=[EchoTool()])

    result = await harness.run("say hello")

    assert result.status is AgentStatus.COMPLETED
    assert model.calls == 2
    assert result.final_parts[0].text == "tool said: hello"


@pytest.mark.asyncio
async def test_agent_harness_accepts_existing_tool_registry() -> None:
    harness = AgentHarness(model=ToolLoopModel(), tools=ToolRegistry([EchoTool()]))

    result = await harness.run(Message.user([normalize_messages("say hello")[0].parts[0]]))

    assert result.status is AgentStatus.COMPLETED
    assert result.final_parts[0].text == "tool said: hello"


@pytest.mark.asyncio
async def test_agent_harness_collects_streaming_events() -> None:
    harness = AgentHarness(model=StreamingModel())

    events = await harness.events("stream text", stream=True)

    event_types = [event.type for event in events]
    assert EventTypes.MODEL_DELTA in event_types
    assert event_types[-1] == EventTypes.RUN_COMPLETED


@pytest.mark.asyncio
async def test_agent_harness_pause_resume_trace_and_waiting_state() -> None:
    harness = AgentHarness(model=PauseResumeModel(), tools=[ExternalWaitTool()])

    paused = await harness.run_until_pause("start external job")

    assert isinstance(paused, PausedHarnessRun)
    assert paused.pause.wait_id == "job-1"
    waiting = harness.waiting_state(paused.result)
    assert waiting.pause is not None
    assert waiting.pause.reason == "external_callback"
    assert waiting.background_tasks[0].id == "task-1"

    paused_events = await harness.events("start external job")
    event_waiting = waiting_from_events(paused_events)
    assert event_waiting.pause is not None
    assert event_waiting.background_tasks[0].id == "task-1"

    resumed = await harness.resume(
        paused,
        "job-1 completed",
        expected_pause=PauseSelector(
            source="tool",
            wait_id="job-1",
            metadata={"example": "harness"},
        ),
    )

    assert resumed.status is AgentStatus.COMPLETED
    assert resumed.final_parts[0].text == "resumed: job-1 completed"


@pytest.mark.asyncio
async def test_agent_harness_waiting_state_reads_background_tasks_from_messages() -> None:
    harness = AgentHarness(model=BackgroundOnlyModel(), tools=[BackgroundOnlyTool()])

    result = await harness.run("start background work")

    waiting = harness.waiting_state(result)
    assert result.status is AgentStatus.COMPLETED
    assert waiting.pause is None
    assert waiting.background_tasks[0].id == "task-2"
    assert waiting.background_tasks[0].lifecycle == "started"


@pytest.mark.asyncio
async def test_agent_harness_run_with_trace_replays_trace() -> None:
    harness = AgentHarness(model=TextModel())

    traced = await harness.run_with_trace("hello")

    assert traced.status is AgentStatus.COMPLETED
    assert traced.final_text == "done: hello"
    assert traced.replay.valid
