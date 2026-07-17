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
    ModelRequest,
    ResponseFormat,
    RunContext,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from jharness.models.openai import (
    OpenAIChatCompletionsCodec,
    OpenAIChatCompletionsError,
    OpenAIChatCompletionsModel,
    OpenAIChatCompletionsProfile,
)
from jharness.models.openai.chat_completions.stream import OpenAIChatStreamDecoder


def request() -> ModelRequest:
    return ModelRequest(
        (Message.system("policy"), Message.user("hello")),
        (ToolSpec("search", "search", {"type": "object"}),),
        ModelOptions(temperature=0.2, max_output_tokens=50),
        ToolChoice("named", "search", False),
        ResponseFormat("json_schema", {"type": "object"}, True),
    )


def profile() -> OpenAIChatCompletionsProfile:
    return OpenAIChatCompletionsProfile(supports_json_schema=True)


def http_model(client: httpx.AsyncClient) -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(
        base_url="https://provider.test/v1",
        api_key="secret",
        model="gpt-test",
        client=client,
    )


def test_openai_codec_encodes_direct_tool_identity_and_decodes_response() -> None:
    codec = OpenAIChatCompletionsCodec(model="gpt-test", profile=profile())
    payload = codec.encode_request(request())
    tool = cast(dict[str, Any], cast(list[object], payload["tools"])[0])
    function = cast(dict[str, Any], tool["function"])

    assert payload["model"] == "gpt-test"
    assert function["name"] == "search"
    assert "mode" not in function
    assert payload["parallel_tool_calls"] is False
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object"},
            "strict": True,
        },
    }

    response = codec.decode_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    assert response.parts[0].text == "checking"
    assert response.tool_calls == (ToolCall("call-1", "search", {"q": "x"}),)
    assert response.usage is not None and response.usage.total_tokens == 5


def test_openai_stream_decoder_builds_complete_response() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    first = decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hel"},
                    "finish_reason": None,
                }
            ],
        }
    )
    second = decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        }
    )
    completed = decoder.completed_response()

    assert len(first) == 1 and len(second) == 1
    assert completed.parts[0].text == "hello"
    assert completed.finish_reason == "stop"
    assert completed.response_id == "resp-1"


def test_openai_stream_decoder_accumulates_tool_call_arguments() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    decoder.apply_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    decoder.apply_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert decoder.completed_response().tool_calls == (ToolCall("call-1", "search", {"q": "x"}),)


async def test_openai_client_uses_http_transport_and_maps_http_errors() -> None:
    captured: dict[str, object] = {}

    async def success_handler(raw: httpx.Request) -> httpx.Response:
        captured["authorization"] = raw.headers["authorization"]
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
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
    assert captured["authorization"] == "Bearer secret"
    assert isinstance(model, Model)
    assert not hasattr(model, "complete")
    assert not hasattr(model, "stream")

    async def error_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-request-id": "req-1"},
            json={"error": {"message": "rate limited", "code": "rate_limit"}},
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
    assert caught.value.info.code == "rate_limit"
    assert caught.value.info.retryable is True
    assert caught.value.info.request_id == "req-1"


def test_openai_codec_rejects_invalid_choice_shape() -> None:
    codec = OpenAIChatCompletionsCodec(model="gpt-test")
    with pytest.raises(OpenAIChatCompletionsError, match="exactly one choice"):
        codec.decode_response({"choices": []})


async def test_openai_client_decodes_sse_stream() -> None:
    body = (
        'data: {"id":"resp-1","model":"gpt-test","choices":['
        '{"index":0,"delta":{"role":"assistant","content":"hello"},'
        '"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
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

    assert len(deltas) == 1
    assert isinstance(deltas[0], ModelContentDelta)
    assert result.parts[0].text == "hello"
    assert unobserved.parts[0].text == "hello"
