from __future__ import annotations

import pytest

from agent_runtime import (
    Message,
    ModelCapabilities,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ResponseFormat,
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


def test_invalid_model_standard_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ModelOptions(temperature=-0.1)

    with pytest.raises(ValueError, match="requires name"):
        ToolChoice(mode="tool")

    with pytest.raises(ValueError, match="requires json_schema"):
        ResponseFormat(type="json_schema")

    with pytest.raises(ValueError, match="total_tokens"):
        ModelUsage(total_tokens=-1)


def test_model_capabilities_helper_accepts_default_mapping_and_callable() -> None:
    assert model_capabilities(object()) == ModelCapabilities()

    class MappingCapabilitiesModel:
        capabilities = {"streaming": True, "tools": True, "audio_input": True}

    class CallableCapabilitiesModel:
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(structured_output=True)

    assert model_capabilities(MappingCapabilitiesModel()).streaming is True
    assert model_capabilities(MappingCapabilitiesModel()).tools is True
    assert model_capabilities(MappingCapabilitiesModel()).extra["audio_input"] is True
    assert ModelCapabilities.from_dict({"audio_input": True}).to_dict()["audio_input"] is True
    assert model_capabilities(CallableCapabilitiesModel()).structured_output is True
