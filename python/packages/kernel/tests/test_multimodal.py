from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import (
    AgentLoop,
    ContentPart,
    Message,
    ModelRequest,
    ModelResponse,
    RuntimeContext,
    ToolCall,
    ToolExecutionContext,
    ToolInvocation,
    ToolObservation,
    ToolSpec,
)


class InspectingModel:
    def __init__(self) -> None:
        self.calls = 0
        self.seen_messages: list[Message] = []

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self.calls += 1
        self.seen_messages = list(request.messages)
        if self.calls == 1:
            assert request.messages[-1].parts[0].type == "text"
            assert request.messages[-1].parts[1].type == "image"
            return ModelResponse(tool_calls=[ToolCall(id="call-1", name="make_file", arguments={})])
        return ModelResponse(
            parts=[
                ContentPart.text_part("Generated file."),
                ContentPart.file_ref(
                    "artifact-1",
                    media_type="text/csv",
                    name="data.csv",
                ),
            ]
        )


class FileTool:
    spec = ToolSpec(
        name="make_file",
        description="Return a generated file artifact.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        return ToolObservation(
            parts=[
                ContentPart.text_part("created"),
                ContentPart.file_ref("artifact-tool-1", media_type="text/csv", name="tool.csv"),
            ]
        )


@pytest.mark.asyncio
async def test_multimodal_user_message_and_file_result() -> None:
    model = InspectingModel()
    result = await AgentLoop(model=model, tools=[FileTool()]).run(
        [
            Message.user(
                [
                    ContentPart.text_part("Analyze this image"),
                    ContentPart.image_uri(
                        "https://example.com/car.png",
                        media_type="image/png",
                        name="car.png",
                    ),
                ]
            )
        ]
    )

    assert result.messages[0].to_dict()["parts"][1]["type"] == "image"
    assert result.messages[-2].role == "tool"
    assert result.messages[-2].parts[1].type == "file"
    assert result.final_parts[1].type == "file"
    assert result.final_parts[1].ref == "artifact-1"


def test_content_part_from_dict_rejects_unknown_wire_fields() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ContentPart.from_dict(
            {
                "type": "audio",
                "ref": "artifact-audio-1",
                "media_type": "audio/wav",
                "provider_cache_control": {"ttl": 60},
            }
        )


def test_protocol_types_do_not_accept_extra_fields() -> None:
    with pytest.raises(TypeError):
        cast(Any, ContentPart.text_part)("hello", extra={"provider_state": {"cursor": "abc"}})

    with pytest.raises(TypeError):
        cast(Any, Message.assistant_text)("hello", extra={"provider_state": {"cursor": "abc"}})

    with pytest.raises(TypeError):
        cast(Any, ModelResponse)(
            parts=[ContentPart.text_part("hello")], extra={"provider_state": {}}
        )

    with pytest.raises(TypeError):
        cast(Any, ToolObservation)(
            parts=[ContentPart.text_part("created")], extra={"artifact_state": {}}
        )

    with pytest.raises(TypeError):
        cast(Any, ModelRequest)(
            messages=(Message.user_text("hello"),), extra={"provider_state": {}}
        )


def test_protocol_instances_do_not_have_extra_slots() -> None:
    response = ModelResponse.text("hello")
    result = ToolObservation.text("created", is_error=False)

    with pytest.raises(AttributeError):
        cast(Any, response).extra = {"provider_state": {"cursor": "abc"}}

    with pytest.raises(AttributeError):
        cast(Any, result).extra = {"artifact_state": {"id": "a1"}}
