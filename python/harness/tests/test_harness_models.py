from __future__ import annotations

import pytest
from harness import (
    ContextInspectingModel,
    ControlledStreamingModelDriver,
    FlakyProviderErrorModel,
    ModelStreamPause,
    ModelStreamSleep,
    ProviderErrorModel,
    RequestCapturingModel,
    ScriptedModel,
    StreamingTextModel,
    collect_events,
)
from kernel import (
    AgentLoop,
    ContentPart,
    EventTypes,
    Message,
    ModelContentDelta,
    ModelProviderError,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ModelUsageDelta,
    PauseRequest,
    RunController,
    RuntimeContext,
)


def _model_request(text: str = "hello") -> ModelRequest:
    return ModelRequest(messages=(Message.user([ContentPart.text_part(text)]),))


@pytest.mark.asyncio
async def test_scripted_model_records_requests_and_returns_copies() -> None:
    response = ModelResponse.text("done")
    model = ScriptedModel([response])
    request = _model_request()

    result = await model.complete(request, RuntimeContext(run_id="run-1"))
    request.messages[0].parts[0].text = "mutated request"
    response.parts[0].text = "mutated response"

    assert result.parts[0].text == "done"
    assert model.calls == 1
    assert model.requests[0].messages[0].text == "hello"


@pytest.mark.asyncio
async def test_request_capturing_model_records_defensive_request_copy() -> None:
    model = RequestCapturingModel()
    request = _model_request()

    response = await model.complete(request, RuntimeContext(run_id="run-1"))
    request.messages[0].parts[0].text = "mutated"

    assert response.parts[0].text == "done"
    assert model.request is not None
    assert model.request.messages[0].text == "hello"


@pytest.mark.asyncio
async def test_streaming_text_model_emits_streaming_deltas() -> None:
    model = StreamingTextModel()

    chunks = [
        chunk
        async for chunk in model.stream(
            _model_request(),
            RuntimeContext(run_id="run-1"),
        )
    ]

    assert [chunk.text_delta for chunk in chunks if isinstance(chunk, ModelContentDelta)] == [
        "hel",
        "lo",
    ]
    with pytest.raises(AssertionError, match="stream path"):
        await model.complete(
            _model_request(),
            RuntimeContext(run_id="run-1"),
        )


@pytest.mark.asyncio
async def test_controlled_streaming_model_driver_emits_public_stream_actions() -> None:
    controller = RunController()
    model = ControlledStreamingModelDriver(
        [],
        [
            [
                ModelReasoningDelta(index=0, text_delta="thinking"),
                ModelStreamSleep(0),
                ModelStreamPause(
                    PauseRequest(reason="host_requested", source="model", wait_id="pause-1")
                ),
                ModelContentDelta(index=0, text_delta="visible"),
                ModelUsageDelta(usage=ModelUsage(input_tokens=1, output_tokens=2)),
            ]
        ],
        controller=controller,
    )

    chunks = [
        chunk async for chunk in model.stream(_model_request(), RuntimeContext(run_id="run-1"))
    ]

    assert [type(chunk) for chunk in chunks] == [
        ModelReasoningDelta,
        ModelContentDelta,
        ModelUsageDelta,
    ]
    pause_request = controller.pause_request
    assert pause_request is not None
    assert pause_request.wait_id == "pause-1"


@pytest.mark.asyncio
async def test_provider_error_model_variants() -> None:
    request = _model_request()

    with pytest.raises(ModelProviderError):
        await ProviderErrorModel().complete(request, RuntimeContext(run_id="run-1"))

    flaky = FlakyProviderErrorModel()
    with pytest.raises(ModelProviderError):
        await flaky.complete(request, RuntimeContext(run_id="run-1"))
    response = await flaky.complete(request, RuntimeContext(run_id="run-1"))

    assert flaky.calls == 2
    assert response.parts[0].text == "recovered"


@pytest.mark.asyncio
async def test_context_inspecting_model_returns_context_metadata() -> None:
    response = await ContextInspectingModel("tenant").complete(
        _model_request(),
        RuntimeContext(run_id="run-1", metadata={"tenant": "acme"}),
    )

    assert response.parts[0].text == "acme"


@pytest.mark.asyncio
async def test_collect_events_returns_ordered_event_stream() -> None:
    events = await collect_events(
        AgentLoop(model=ScriptedModel([ModelResponse.text("done")])),
        [Message.user([ContentPart.text_part("hello")])],
    )

    assert events[0].type == EventTypes.RUN_STARTED
    assert events[-1].type == EventTypes.RUN_COMPLETED
