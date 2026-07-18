from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    Message,
    Model,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelOptions,
    ModelReasoningDelta,
    ModelRequest,
    ModelUsageDelta,
    ResponseFormat,
    RunContext,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from jharness.models.anthropic import (
    AnthropicCodec,
    AnthropicError,
    AnthropicModel,
    AnthropicProfile,
)
from jharness.models.anthropic.messages_api.stream import AnthropicStreamDecoder


def request() -> ModelRequest:
    return ModelRequest(
        (Message.system("policy"), Message.user("hello")),
        (ToolSpec("search", "search", {"type": "object"}),),
        ModelOptions(max_output_tokens=50),
        ToolChoice("named", "search", False),
        ResponseFormat(
            "json_schema",
            {
                "type": "object",
                "properties": {"x": {"type": "object"}},
                "dependencies": {"x": {"type": "object"}, "property": ["x"]},
                "allOf": [{"type": "object"}],
                "items": {"type": "object"},
                "examples": [{"type": "object"}],
            },
            True,
        ),
    )


def http_model(client: httpx.AsyncClient) -> AnthropicModel:
    return AnthropicModel(
        base_url="https://provider.test",
        api_key="secret",
        model="claude-test",
        client=client,
    )


def test_anthropic_codec_encodes_tools_and_decodes_blocks() -> None:
    codec = AnthropicCodec(model="claude-test")
    payload = codec.encode_request(request())
    tool = cast(dict[str, Any], cast(list[object], payload["tools"])[0])

    assert payload["system"] == "policy"
    assert tool["name"] == "search"
    assert "mode" not in tool
    assert payload["tool_choice"] == {
        "type": "tool",
        "name": "search",
        "disable_parallel_tool_use": True,
    }
    output = cast(dict[str, Any], payload["output_config"])
    schema = cast(dict[str, Any], cast(dict[str, Any], output["format"])["schema"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["x"]["additionalProperties"] is False
    assert schema["dependencies"]["x"]["additionalProperties"] is False
    assert schema["allOf"][0]["additionalProperties"] is False
    assert schema["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema["examples"][0]

    response = codec.decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call-1", "name": "search", "input": {"q": "x"}},
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )
    assert response.parts[0].text == "checking"
    assert response.tool_calls == (ToolCall("call-1", "search", {"q": "x"}),)
    assert response.usage is not None and response.usage.total_tokens == 5


def test_anthropic_stream_decoder_builds_complete_response() -> None:
    decoder = AnthropicStreamDecoder(AnthropicProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "h"}},
    )
    decoder.apply_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "i"}},
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    )
    done, _ = decoder.apply_event("message_stop", {"type": "message_stop"})
    completed = decoder.completed_response()

    assert done is True
    assert completed.parts[0].text == "hi"
    assert completed.usage is not None and completed.usage.total_tokens == 5


def test_anthropic_thinking_deltas_stay_incremental_and_finalize_once() -> None:
    decoder = AnthropicStreamDecoder(AnthropicProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    )
    for _ in range(4096):
        _, deltas = decoder.apply_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "x"},
            },
        )
        assert isinstance(deltas[0], ModelReasoningDelta)
        content_delta = next(delta for delta in deltas if isinstance(delta, ModelContentDelta))
        assert content_delta.text_delta == "x"
        assert content_delta.data == {}
    _, signature_deltas = decoder.apply_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
    )
    signature_delta = cast(ModelContentDelta, signature_deltas[0])
    assert signature_delta.data == {"anthropic": {"type": "thinking", "signature": "sig"}}
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    part = decoder.completed_response().parts[0]
    assert part.text == "x" * 4096
    assert part.data == {
        "anthropic": {
            "type": "thinking",
            "thinking": "x" * 4096,
            "signature": "sig",
        }
    }


def test_anthropic_stream_decoder_accumulates_tool_input() -> None:
    decoder = AnthropicStreamDecoder(AnthropicProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-1",
                "name": "search",
                "input": {},
            },
        },
    )
    decoder.apply_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
        },
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    assert decoder.completed_response().tool_calls == (ToolCall("call-1", "search", {"q": "x"}),)


async def test_anthropic_client_uses_http_transport_and_maps_errors() -> None:
    captured: dict[str, object] = {}

    async def success_handler(raw: httpx.Request) -> httpx.Response:
        captured["api_key"] = raw.headers["x-api-key"]
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(success_handler)) as client:
        model = http_model(client)
        result = await model.invoke(
            ModelRequest((Message.user("hello"),)),
            RunContext("run-1", 1.0),
            stream=False,
            emit_delta=None,
        )
    assert result.parts[0].text == "done"
    assert captured["api_key"] == "secret"
    assert isinstance(model, Model)
    assert not hasattr(model, "complete")
    assert not hasattr(model, "stream")

    async def error_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            json={"error": {"type": "overloaded_error", "message": "busy"}},
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(error_handler)) as client:
        model = http_model(client)
        with pytest.raises(ModelError) as caught:
            await model.invoke(
                ModelRequest((Message.user("hello"),)),
                RunContext("run-1", 1.0),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "overloaded_error"
    assert caught.value.info.retryable is True


async def test_anthropic_stream_overload_keeps_semantic_status_and_retryability() -> None:
    body = (
        "event: error\n"
        'data: {"type":"error","error":'
        '{"type":"overloaded_error","message":"busy"}}\n\n'
    )

    def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as caught:
            await http_model(client).invoke(
                ModelRequest((Message.user("hello"),)),
                RunContext("run-1", 1.0),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "overloaded_error"
    assert caught.value.info.status_code is None
    assert caught.value.info.retryable is True


def test_anthropic_codec_rejects_invalid_envelope() -> None:
    with pytest.raises(AnthropicError, match="type='message'"):
        AnthropicCodec(model="claude-test").decode_response({"type": "error"})


async def test_anthropic_client_decodes_named_sse_stream() -> None:
    body = "".join(
        (
            "event: message_start\n",
            'data: {"type":"message_start","message":{"type":"message",'
            '"role":"assistant","id":"msg-1","model":"claude-test",'
            '"content":[],"usage":{"input_tokens":2,"output_tokens":0}}}\n\n',
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":"hello"}}\n\n',
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n\n',
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":3}}\n\n',
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n\n',
        )
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = http_model(client)
        deltas: list[ModelDelta] = []

        async def emit_delta(delta: ModelDelta, /) -> None:
            deltas.append(delta)

        result = await model.invoke(
            ModelRequest((Message.user("hello"),)),
            RunContext("run-1", 1.0),
            stream=True,
            emit_delta=emit_delta,
        )
        unobserved = await model.invoke(
            ModelRequest((Message.user("hello"),)),
            RunContext("run-2", 1.0),
            stream=True,
            emit_delta=None,
        )

    assert any(isinstance(delta, ModelContentDelta) for delta in deltas)
    assert any(isinstance(delta, ModelUsageDelta) for delta in deltas)
    assert result.parts[0].text == "hello"
    assert result.usage is not None
    assert result.usage.total_tokens == 5
    assert unobserved.parts[0].text == "hello"
