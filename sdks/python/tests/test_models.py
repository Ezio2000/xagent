from __future__ import annotations

from typing import Any, cast

import pytest

from agent_runtime import (
    ContentPart,
    Message,
    ModelCapabilities,
    ModelContentDelta,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ResponseFormat,
    ToolCall,
    ToolChoice,
    model_capabilities,
)


def test_model_request_standard_fields_round_trip() -> None:
    request = ModelRequest(
        messages=(Message.user_text("hello"),),
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


def test_model_request_response_constructors_reject_invalid_nested_items() -> None:
    with pytest.raises(TypeError, match="messages items"):
        ModelRequest(messages=(cast(Any, object()),))

    with pytest.raises(TypeError, match="tools items"):
        ModelRequest(messages=(Message.user_text("hello"),), tools=(cast(Any, object()),))

    with pytest.raises(TypeError, match="options"):
        ModelRequest(messages=(Message.user_text("hello"),), options=cast(Any, {}))

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
