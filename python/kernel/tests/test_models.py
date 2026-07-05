from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import (
    ContentPart,
    Message,
    ModelCapabilities,
    ModelContentDelta,
    ModelOptions,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelStreamAccumulator,
    ModelStreamCompleted,
    ModelStreamEvent,
    ModelStreamStarted,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    model_capabilities,
)


def user_text(text: str) -> Message:
    return Message.user([ContentPart.text_part(text)])


def test_model_request_standard_fields_round_trip() -> None:
    request = ModelRequest(
        messages=(user_text("hello"),),
        options=ModelOptions(
            model="test-model",
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=128,
            stop_sequences=("END",),
            seed=7,
        ),
        tool_choice=ToolChoice(
            mode="tool",
            name="search",
            allow_parallel_tool_calls=True,
        ),
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={"type": "object"},
            strict=True,
        ),
    )

    restored = ModelRequest.from_dict(request.to_dict())

    assert restored.options.model == "test-model"
    assert restored.options.stop_sequences == ("END",)
    assert restored.tool_choice.name == "search"
    assert restored.tool_choice.allow_parallel_tool_calls is True
    assert restored.response_format is not None
    assert restored.response_format.strict is True


def test_model_response_usage_and_ids_round_trip_to_assistant_metadata() -> None:
    response = ModelResponse.text("done")
    response.finish_reason = "end_turn"
    response.usage = ModelUsage(input_tokens=3, output_tokens=4, total_tokens=7)
    response.model = "test-model"
    response.response_id = "resp-1"

    restored = ModelResponse.from_dict(response.to_dict())
    message = restored.to_assistant_message()

    assert restored.usage is not None
    assert restored.usage.total_tokens == 7
    assert message.metadata["finish_reason"] == "end_turn"
    assert message.metadata["usage"]["total_tokens"] == 7
    assert message.metadata["model"] == "test-model"
    assert message.metadata["response_id"] == "resp-1"


def test_model_response_to_assistant_message_strips_provider_metadata() -> None:
    response = ModelResponse(
        parts=[ContentPart.text_part("done", metadata={"secret": "part"})],
        tool_calls=[ToolCall(id="call-1", name="tool", arguments={}, metadata={"secret": "call"})],
        usage=ModelUsage(total_tokens=1, metadata={"secret": "usage"}),
        metadata={"secret": "response"},
    )

    message = response.to_assistant_message()

    assert message.metadata == {"usage": {"total_tokens": 1}}
    assert message.parts[0].metadata == {}
    assert message.tool_calls[0].metadata == {}


def test_model_response_tool_call_ids_must_be_unique() -> None:
    calls = [
        ToolCall(id="call-1", name="tool", arguments={}),
        ToolCall(id="call-1", name="tool", arguments={}),
    ]

    with pytest.raises(ValueError, match="unique"):
        ModelResponse(tool_calls=calls)

    with pytest.raises(ValueError, match="unique"):
        ModelResponse.from_dict({"parts": [], "tool_calls": [call.to_dict() for call in calls]})


def test_invalid_model_standard_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ModelOptions(temperature=-0.1)

    with pytest.raises(ValueError, match="requires name"):
        ToolChoice(mode="tool")

    with pytest.raises(ValueError, match="requires json_schema"):
        ResponseFormat(type="json_schema")

    with pytest.raises(ValueError, match="total_tokens"):
        ModelUsage(total_tokens=-1)

    with pytest.raises(TypeError, match="stop_sequences"):
        ModelOptions(stop_sequences=(cast(Any, 123),))

    with pytest.raises(TypeError, match="temperature"):
        ModelOptions(temperature=cast(Any, True))

    with pytest.raises(TypeError, match="model"):
        ModelOptions(model=cast(Any, 123))

    with pytest.raises(TypeError, match="tool choice name"):
        ToolChoice(mode="tool", name=cast(Any, 123))

    with pytest.raises(TypeError, match="total_tokens"):
        ModelUsage(total_tokens=cast(Any, True))

    with pytest.raises(TypeError, match="strict"):
        ResponseFormat(strict=cast(Any, "yes"))

    with pytest.raises(TypeError, match="streaming"):
        ModelCapabilities(streaming=cast(Any, "yes"))

    with pytest.raises(TypeError, match="finish_reason"):
        ModelResponse(finish_reason=cast(Any, 123))

    with pytest.raises(TypeError, match="response_id"):
        ModelResponse(response_id=cast(Any, 123))

    with pytest.raises(TypeError, match="tool choice mode"):
        ToolChoice(mode=cast(Any, 123))

    with pytest.raises(TypeError, match="response format type"):
        ResponseFormat(type=cast(Any, 123))


def test_model_stream_delta_constructors_reject_invalid_core_types() -> None:
    with pytest.raises(TypeError, match="content delta index"):
        ModelContentDelta(index=cast(Any, True), text_delta="x")

    with pytest.raises(TypeError, match="content delta text"):
        ModelContentDelta(index=0, text_delta=cast(Any, 123))

    with pytest.raises(TypeError, match="tool call delta arguments"):
        ModelToolCallDelta(index=0, arguments_delta=cast(Any, 123))

    with pytest.raises(TypeError, match="usage delta usage"):
        ModelUsageDelta(usage=cast(Any, {}))


def test_model_stream_accumulator_preserves_open_content_part_types() -> None:
    accumulator = ModelStreamAccumulator()

    assert accumulator.apply(ModelContentDelta(index=0, part_type="custom", text_delta="x")) is None
    assert accumulator.apply(ModelContentDelta(index=0, part_type="custom", text_delta="y")) is None

    response = accumulator.response()
    assert response.parts[0].type == "custom"
    assert response.parts[0].text == "xy"


def test_model_stream_accumulator_accumulates_complete_response() -> None:
    accumulator = ModelStreamAccumulator()
    events: list[ModelStreamEvent] = [
        ModelStreamStarted(metadata={"provider": "test"}),
        ModelContentDelta(index=1, part_type="custom", text_delta="B"),
        ModelContentDelta(index=0, text_delta="A"),
        ModelReasoningDelta(index=0, text_delta="hidden"),
        ModelToolCallDelta(index=0, id="call-1", name="search", arguments_delta='{"q"'),
        ModelToolCallDelta(index=0, arguments_delta=':"x"}'),
        ModelUsageDelta(usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5)),
        ModelStreamCompleted(
            ModelResponse(
                parts=[ContentPart.text_part("done")],
                finish_reason="end_turn",
                usage=ModelUsage(total_tokens=5),
                model="test-model",
                response_id="resp-1",
            )
        ),
    ]

    result: ModelResponse | None = None
    for event in events:
        result = accumulator.apply(event)

    assert result is not None
    assert result.finish_reason == "end_turn"
    expected: dict[str, object] = {
        "parts": [
            {"type": "text", "text": "A"},
            {"type": "custom", "text": "B"},
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "name": "search",
                "mode": "execute",
                "arguments": {"q": "x"},
            }
        ],
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        "model": "test-model",
        "response_id": "resp-1",
    }
    assert result.to_dict() == expected
    assert accumulator.response().to_dict() == expected


def test_model_stream_accumulator_uses_completed_response_as_fallback() -> None:
    accumulator = ModelStreamAccumulator()

    result = accumulator.apply(
        ModelStreamCompleted(
            ModelResponse(
                parts=[ContentPart.text_part("done")],
                tool_calls=[
                    ToolCall(id="call-1", name="search", arguments={"q": "x"}),
                ],
                finish_reason="tool_calls",
                usage=ModelUsage(total_tokens=5),
                model="test-model",
                response_id="resp-1",
                metadata={"provider": "test"},
            )
        )
    )

    assert result is not None
    expected: dict[str, object] = {
        "parts": [{"type": "text", "text": "done"}],
        "tool_calls": [
            {
                "id": "call-1",
                "name": "search",
                "mode": "execute",
                "arguments": {"q": "x"},
            }
        ],
        "finish_reason": "tool_calls",
        "usage": {"total_tokens": 5},
        "model": "test-model",
        "response_id": "resp-1",
        "metadata": {"provider": "test"},
    }
    assert result.to_dict() == expected
    assert accumulator.response().to_dict() == expected


def test_model_stream_accumulator_completed_response_does_not_clear_deltas() -> None:
    accumulator = ModelStreamAccumulator()
    accumulator.apply(ModelContentDelta(index=0, text_delta="hel"))
    accumulator.apply(ModelContentDelta(index=0, text_delta="lo"))
    accumulator.apply(ModelUsageDelta(usage=ModelUsage(input_tokens=1, total_tokens=3)))
    result = accumulator.apply(
        ModelStreamCompleted(
            ModelResponse(
                finish_reason="end_turn",
                model="test-model",
                response_id="resp-1",
                metadata={"provider": "test"},
            )
        )
    )

    expected = {
        "parts": [{"type": "text", "text": "hello"}],
        "tool_calls": [],
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 1, "total_tokens": 3},
        "model": "test-model",
        "response_id": "resp-1",
        "metadata": {"provider": "test"},
    }
    assert result is not None
    assert result.to_dict() == expected
    assert accumulator.response().to_dict() == expected


def test_model_stream_accumulator_merges_sparse_usage_snapshots() -> None:
    accumulator = ModelStreamAccumulator()
    accumulator.apply(ModelUsageDelta(usage=ModelUsage(input_tokens=1)))
    accumulator.apply(ModelUsageDelta(usage=ModelUsage(output_tokens=2, total_tokens=3)))
    result = accumulator.apply(
        ModelStreamCompleted(
            ModelResponse(
                usage=ModelUsage(reasoning_tokens=1),
            )
        )
    )

    expected: dict[str, object] = {
        "parts": [],
        "tool_calls": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "reasoning_tokens": 1,
        },
    }
    assert result is not None
    assert result.to_dict() == expected
    assert accumulator.response().to_dict() == expected


def test_model_request_response_constructors_reject_invalid_nested_items() -> None:
    with pytest.raises(TypeError, match="messages items"):
        ModelRequest(messages=(cast(Any, object()),))

    with pytest.raises(TypeError, match="tools items"):
        ModelRequest(messages=(user_text("hello"),), tools=(cast(Any, object()),))

    with pytest.raises(TypeError, match="options"):
        ModelRequest(messages=(user_text("hello"),), options=cast(Any, {}))

    with pytest.raises(TypeError, match="parts items"):
        ModelResponse(parts=[cast(Any, object())])

    with pytest.raises(TypeError, match="tool_calls items"):
        ModelResponse(tool_calls=[cast(Any, object())])

    with pytest.raises(TypeError, match="usage"):
        ModelResponse(usage=cast(Any, {}))


def test_model_request_from_dict_rejects_missing_required_wire_fields() -> None:
    with pytest.raises(KeyError):
        ModelRequest.from_dict({"messages": [], "tools": [], "options": {}})

    with pytest.raises(KeyError):
        ToolChoice.from_dict({})

    with pytest.raises(KeyError):
        ResponseFormat.from_dict({"type": "text"})


def test_model_response_from_dict_rejects_missing_required_wire_fields() -> None:
    with pytest.raises(KeyError):
        ModelResponse.from_dict({"parts": []})

    with pytest.raises(TypeError, match="tool_calls"):
        ModelResponse.from_dict({"parts": [], "tool_calls": None})


def test_model_from_dict_rejects_schema_invalid_optional_fields() -> None:
    with pytest.raises(TypeError, match="model option model"):
        ModelOptions.from_dict({"model": 123})

    with pytest.raises(TypeError, match="max_output_tokens"):
        ModelOptions.from_dict({"max_output_tokens": True})

    with pytest.raises(TypeError, match="total_tokens"):
        ModelUsage.from_dict({"total_tokens": True})

    with pytest.raises(ValueError, match="unknown"):
        ModelOptions.from_dict({"provider_option": True})

    with pytest.raises(ValueError, match="unknown"):
        ModelResponse.from_dict({"parts": [], "tool_calls": [], "provider_state": {}})


def test_model_capabilities_helper_accepts_default_mapping_and_callable() -> None:
    assert model_capabilities(object()) == ModelCapabilities()

    class MappingCapabilitiesModel:
        capabilities = {"streaming": True, "tools": True}

    class CallableCapabilitiesModel:
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(structured_output=True)

    assert model_capabilities(MappingCapabilitiesModel()).streaming is True
    assert model_capabilities(MappingCapabilitiesModel()).tools is True
    assert model_capabilities(CallableCapabilitiesModel()).structured_output is True


def test_model_capabilities_from_dict_rejects_invalid_boolean_fields() -> None:
    with pytest.raises(TypeError, match="streaming"):
        ModelCapabilities.from_dict({"streaming": "true"})

    with pytest.raises(ValueError, match="unknown"):
        ModelCapabilities.from_dict({"audio_input": True})
