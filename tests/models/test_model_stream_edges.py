from __future__ import annotations

from typing import Any

import pytest

from jharness.kernel import ModelContentDelta, ModelReasoningDelta, ModelUsageDelta, ToolCall
from jharness.models.anthropic import AnthropicError, AnthropicProfile
from jharness.models.anthropic.messages_api.stream import AnthropicStreamDecoder
from jharness.models.openai import OpenAIChatCompletionsError, OpenAIChatCompletionsProfile
from jharness.models.openai.chat_completions.stream import OpenAIChatStreamDecoder


def openai_choice(
    delta: object,
    *,
    finish_reason: object = None,
    index: object = 0,
    **metadata: object,
) -> dict[str, Any]:
    return {
        **metadata,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def test_openai_stream_completion_guards_usage_and_metadata() -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    with pytest.raises(OpenAIChatCompletionsError, match="without a choice"):
        decoder.completed_response()
    usage = decoder.apply_chunk(
        {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }
    )
    assert isinstance(usage[0], ModelUsageDelta)
    decoder.apply_chunk(openai_choice({}, finish_reason="stop"))
    with pytest.raises(OpenAIChatCompletionsError, match="without content"):
        decoder.completed_response()

    reasoning_only = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    reasoning_only.apply_chunk(openai_choice({"reasoning_content": "why"}, finish_reason="stop"))
    with pytest.raises(OpenAIChatCompletionsError, match="without content"):
        reasoning_only.completed_response()

    unfinished = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    unfinished.apply_chunk(openai_choice({"content": "x"}))
    with pytest.raises(OpenAIChatCompletionsError, match="before finish_reason"):
        unfinished.completed_response()

    complete = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    deltas = complete.apply_chunk(
        openai_choice(
            {"role": "assistant", "refusal": "no", "reasoning_content": "why"},
            finish_reason="stop",
            id="response",
            model="model",
            object="chat.completion.chunk",
            created=7,
        )
    )
    assert any(isinstance(delta, ModelContentDelta) for delta in deltas)
    response = complete.completed_response()
    assert response.parts[0].type == "refusal"
    assert response.metadata["object"] == "chat.completion.chunk"
    assert response.metadata["created"] == 7


@pytest.mark.parametrize(
    "chunk,pattern",
    [
        ({"unused": None}, "requires choices or usage"),
        ({"choices": 1}, "choices must be an array"),
        ({"choices": list[object]()}, "empty choices require usage"),
        (
            {"choices": [{"unused": None}, {"unused": None}]},
            "exactly one choice",
        ),
        ({"choices": [1]}, "choice must be an object"),
        (openai_choice({"unused": None}, index=True), "index must be an integer"),
        (openai_choice({"unused": None}, index=1), "index must be 0"),
        (openai_choice(1), "delta must be an object"),
        (openai_choice({"unused": None}, finish_reason=""), "finish_reason must be"),
        (openai_choice({"role": "user"}), "role must be 'assistant'"),
        (openai_choice({"content": 1}), "content delta must be"),
        (openai_choice({"reasoning_content": 1}), "reasoning delta must be"),
        (openai_choice({"refusal": 1}), "refusal delta must be"),
        (openai_choice({"tool_calls": 1}), "tool_calls must be an array"),
        (openai_choice({"tool_calls": [1]}), "tool call must be an object"),
        (
            openai_choice({"tool_calls": [{"type": "other"}]}),
            "unsupported chat completion stream tool call type",
        ),
        (
            openai_choice({"tool_calls": [{"function": 1}]}),
            "tool function must be an object",
        ),
        (
            openai_choice({"tool_calls": [{"id": 1}]}),
            "expected string or null",
        ),
        (
            openai_choice({"tool_calls": [{"index": True, "id": "call"}]}),
            "tool call index must be an integer",
        ),
        (
            openai_choice({"tool_calls": [{"index": -1, "id": "call"}]}),
            "tool call index must be >= 0",
        ),
        (openai_choice({"unused": None}, id=1), "id must be a string or null"),
        (openai_choice({"unused": None}, id=""), "id must not be empty"),
        (
            openai_choice({"unused": None}, created=True),
            "created must be an integer or null",
        ),
    ],
)
def test_openai_stream_rejects_invalid_chunks(chunk: dict[str, Any], pattern: str) -> None:
    with pytest.raises(OpenAIChatCompletionsError, match=pattern):
        OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile()).apply_chunk(chunk)


def test_openai_stream_rejects_metadata_changes_and_post_finish_choices() -> None:
    changed = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    changed.apply_chunk(openai_choice({"content": "a"}, id="one"))
    with pytest.raises(OpenAIChatCompletionsError, match="id changed"):
        changed.apply_chunk(openai_choice({"content": "b"}, id="two"))

    finished = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    finished.apply_chunk(openai_choice({"content": "a"}, finish_reason="stop"))
    with pytest.raises(OpenAIChatCompletionsError, match="after finish_reason"):
        finished.apply_chunk(openai_choice({"content": "b"}))

    empty_call = OpenAIChatStreamDecoder(OpenAIChatCompletionsProfile())
    assert empty_call.apply_chunk(openai_choice({"tool_calls": [{"unused": None}]})) == []


def test_openai_stream_round_trips_reasoning_with_distinct_content_indexes() -> None:
    decoder = OpenAIChatStreamDecoder(
        OpenAIChatCompletionsProfile(reasoning_content_mode="round_trip")
    )
    deltas = decoder.apply_chunk(
        openai_choice(
            {
                "role": "assistant",
                "reasoning_content": "why",
                "content": "answer",
                "refusal": "no",
            },
            finish_reason="stop",
        )
    )

    assert [
        (type(delta), getattr(delta, "index", None), getattr(delta, "part_type", None))
        for delta in deltas
    ] == [
        (ModelReasoningDelta, 0, None),
        (ModelContentDelta, 0, "reasoning"),
        (ModelContentDelta, 1, "text"),
        (ModelContentDelta, 2, "refusal"),
    ]
    assert [(part.type, part.text) for part in decoder.completed_response().parts] == [
        ("reasoning", "why"),
        ("text", "answer"),
        ("refusal", "no"),
    ]

    reasoning_only = OpenAIChatStreamDecoder(
        OpenAIChatCompletionsProfile(reasoning_content_mode="round_trip")
    )
    reasoning_only.apply_chunk(openai_choice({"reasoning_content": "only"}, finish_reason="stop"))
    assert [(part.type, part.text) for part in reasoning_only.completed_response().parts] == [
        ("reasoning", "only")
    ]


@pytest.mark.parametrize(
    ("wire_field", "part_type"),
    (("content", "text"), ("refusal", "refusal")),
)
def test_openai_stream_round_trip_uses_compact_indexes_without_reasoning(
    wire_field: str,
    part_type: str,
) -> None:
    decoder = OpenAIChatStreamDecoder(
        OpenAIChatCompletionsProfile(reasoning_content_mode="round_trip")
    )
    deltas = decoder.apply_chunk(openai_choice({wire_field: "only"}, finish_reason="stop"))

    content_delta = next(delta for delta in deltas if isinstance(delta, ModelContentDelta))
    assert content_delta.index == 0
    assert [(part.type, part.text) for part in decoder.completed_response().parts] == [
        (part_type, "only")
    ]


def test_openai_stream_round_trip_rejects_reasoning_after_content() -> None:
    decoder = OpenAIChatStreamDecoder(
        OpenAIChatCompletionsProfile(reasoning_content_mode="round_trip")
    )
    decoder.apply_chunk(openai_choice({"content": "answer"}))

    with pytest.raises(OpenAIChatCompletionsError, match="reasoning after a later content part"):
        decoder.apply_chunk(openai_choice({"reasoning_content": "late"}))


def test_openai_stream_requires_reasoning_for_round_trip_tool_calls() -> None:
    profile = OpenAIChatCompletionsProfile(reasoning_content_mode="required_with_tools")
    missing = OpenAIChatStreamDecoder(profile)
    missing.apply_chunk(
        openai_choice(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ]
            },
            finish_reason="tool_calls",
        )
    )
    with pytest.raises(OpenAIChatCompletionsError, match="requires non-empty reasoning"):
        missing.completed_response()

    complete = OpenAIChatStreamDecoder(profile)
    complete.apply_chunk(openai_choice({"reasoning_content": "why"}))
    complete.apply_chunk(
        openai_choice(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ]
            },
            finish_reason="tool_calls",
        )
    )
    response = complete.completed_response()
    assert response.parts[0].type == "reasoning"
    assert response.tool_calls == (ToolCall("call-1", "search", {}),)


def anthropic_started(*, profile: AnthropicProfile | None = None) -> AnthropicStreamDecoder:
    decoder = AnthropicStreamDecoder(AnthropicProfile() if profile is None else profile)
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "content": [],
                "id": "message",
                "model": "model",
            },
        },
    )
    return decoder


def anthropic_start_block(
    decoder: AnthropicStreamDecoder,
    block: dict[str, Any],
    *,
    index: object = 0,
) -> None:
    decoder.apply_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def anthropic_delta(
    decoder: AnthropicStreamDecoder,
    delta: dict[str, Any],
    *,
    index: int = 0,
) -> None:
    decoder.apply_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def anthropic_stop(decoder: AnthropicStreamDecoder, *, index: int = 0) -> None:
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": index})


def test_anthropic_stream_event_envelope_and_start_guards() -> None:
    decoder = AnthropicStreamDecoder(AnthropicProfile())
    assert decoder.apply_event("ping", {"type": "ping"}) == (False, [])
    with pytest.raises(AnthropicError, match="before message_stop"):
        decoder.completed_response()
    for event_name, value, pattern in (
        (None, {"unused": None}, "requires a type"),
        ("ping", {"type": "message_start"}, "name must match"),
        ("error", {"type": "error"}, "stream error event"),
        ("other", {"type": "other"}, "unsupported Anthropic stream event"),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"unused": None},
            },
            "requires message_start",
        ),
        ("message_stop", {"type": "message_stop"}, "requires message_start"),
    ):
        with pytest.raises(AnthropicError, match=pattern):
            AnthropicStreamDecoder(AnthropicProfile()).apply_event(event_name, value)

    started = anthropic_started()
    with pytest.raises(AnthropicError, match="more than once"):
        started.apply_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"type": "message", "role": "assistant", "content": []},
            },
        )


@pytest.mark.parametrize(
    "message,pattern",
    [
        (1, "message must be an object"),
        ({"type": "other", "role": "assistant", "content": []}, "type='message'"),
        ({"type": "message", "role": "user", "content": []}, "role='assistant'"),
        ({"type": "message", "role": "assistant", "content": ""}, "content must be an array"),
        (
            {"type": "message", "role": "assistant", "content": [{"unused": None}]},
            "content must be empty",
        ),
        ({"type": "message", "role": "assistant", "content": [], "id": ""}, "must not be empty"),
    ],
)
def test_anthropic_stream_rejects_invalid_message_start(message: object, pattern: str) -> None:
    with pytest.raises(AnthropicError, match=pattern):
        AnthropicStreamDecoder(AnthropicProfile()).apply_event(
            "message_start", {"type": "message_start", "message": message}
        )


@pytest.mark.parametrize(
    "block,index,pattern",
    [
        ({"type": "text", "text": "x"}, None, "requires an index"),
        ({"type": "text", "text": "x"}, True, "index must be an integer"),
        ({"type": "text", "text": "x"}, -1, "index must be >= 0"),
        ({"unused": None}, 0, "requires non-empty type"),
        ({"type": "other"}, 0, "unsupported Anthropic stream content block"),
        ({"type": "text", "text": 1}, 0, "text block requires text"),
        ({"type": "thinking", "thinking": 1}, 0, "thinking block requires thinking"),
        ({"type": "redacted_thinking", "data": ""}, 0, "requires non-empty data"),
        ({"type": "tool_use", "id": "", "name": "tool"}, 0, "id must not be empty"),
        ({"type": "tool_use", "id": "call", "name": ""}, 0, "name must not be empty"),
        (
            {"type": "tool_use", "id": "call", "name": "tool", "input": 1},
            0,
            "input must be an object",
        ),
    ],
)
def test_anthropic_stream_rejects_invalid_block_starts(
    block: dict[str, Any], index: object, pattern: str
) -> None:
    decoder = anthropic_started()
    value: dict[str, Any] = {"type": "content_block_start", "content_block": block}
    if index is not None:
        value["index"] = index
    with pytest.raises(AnthropicError, match=pattern):
        decoder.apply_event("content_block_start", value)


def test_anthropic_stream_rejects_duplicate_and_empty_blocks() -> None:
    duplicate = anthropic_started()
    anthropic_start_block(duplicate, {"type": "text", "text": "x"})
    with pytest.raises(AnthropicError, match="started more than once"):
        anthropic_start_block(duplicate, {"type": "text", "text": "y"})

    for block, pattern in (
        ({"type": "text", "text": ""}, "requires non-empty text"),
        ({"type": "thinking", "thinking": ""}, "requires non-empty thinking"),
    ):
        decoder = anthropic_started()
        anthropic_start_block(decoder, block)
        with pytest.raises(AnthropicError, match=pattern):
            anthropic_stop(decoder)


def test_anthropic_stream_rejects_events_for_a_closed_block() -> None:
    decoder = anthropic_started()
    anthropic_start_block(decoder, {"type": "text", "text": "x"})
    anthropic_stop(decoder)

    with pytest.raises(AnthropicError, match="requires an open index"):
        anthropic_delta(decoder, {"type": "text_delta", "text": "y"})
    with pytest.raises(AnthropicError, match="requires an open index"):
        anthropic_stop(decoder)


def test_anthropic_stream_interleaves_blocks_and_keeps_monotonic_tool_order() -> None:
    decoder = anthropic_started()
    anthropic_start_block(
        decoder,
        {"type": "tool_use", "id": "call-1", "name": "first", "input": {}},
        index=4,
    )
    anthropic_start_block(decoder, {"type": "text", "text": "a"}, index=1)
    anthropic_delta(decoder, {"type": "input_json_delta", "partial_json": '{"x":1}'}, index=4)
    anthropic_delta(decoder, {"type": "text_delta", "text": "b"}, index=1)
    anthropic_stop(decoder, index=1)
    anthropic_stop(decoder, index=4)
    anthropic_start_block(
        decoder,
        {"type": "tool_use", "id": "call-2", "name": "second", "input": {"y": 2}},
        index=9,
    )
    anthropic_stop(decoder, index=9)
    decoder.apply_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    response = decoder.completed_response()
    assert response.parts[0].text == "ab"
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call-1", "first", {"x": 1}),
        ("call-2", "second", {"y": 2}),
    ]


@pytest.mark.parametrize(
    "block,delta,pattern",
    [
        ({"type": "text", "text": "x"}, {"type": "other"}, "unsupported"),
        (
            {"type": "text", "text": "x"},
            {"type": "thinking_delta", "thinking": "x"},
            "does not match text",
        ),
        ({"type": "text", "text": "x"}, {"type": "text_delta", "text": 1}, "requires text"),
        (
            {"type": "thinking", "thinking": "x"},
            {"type": "thinking_delta", "thinking": 1},
            "requires thinking text",
        ),
        (
            {"type": "thinking", "thinking": "x"},
            {"type": "signature_delta", "signature": 1},
            "requires signature",
        ),
        (
            {"type": "tool_use", "id": "call", "name": "tool"},
            {"type": "input_json_delta", "partial_json": 1},
            "requires partial_json",
        ),
    ],
)
def test_anthropic_stream_rejects_invalid_block_deltas(
    block: dict[str, Any], delta: dict[str, Any], pattern: str
) -> None:
    decoder = anthropic_started()
    anthropic_start_block(decoder, block)
    with pytest.raises(AnthropicError, match=pattern):
        anthropic_delta(decoder, delta)


def test_anthropic_stream_terminal_guards_and_disabled_usage() -> None:
    no_open = anthropic_started()
    with pytest.raises(AnthropicError, match="requires an open index"):
        anthropic_delta(no_open, {"type": "text_delta", "text": "x"})
    with pytest.raises(AnthropicError, match="requires an open index"):
        anthropic_stop(no_open)
    with pytest.raises(AnthropicError, match="requires a terminal message_delta"):
        no_open.apply_event("message_stop", {"type": "message_stop"})

    open_block = anthropic_started()
    anthropic_start_block(open_block, {"type": "text", "text": "x"})
    with pytest.raises(AnthropicError, match="all content blocks to stop"):
        open_block.apply_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )
    with pytest.raises(AnthropicError, match="open content block indexes"):
        open_block.apply_event("message_stop", {"type": "message_stop"})

    no_data = anthropic_started()
    no_data.apply_event(
        "message_delta",
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    )
    with pytest.raises(AnthropicError, match="completed without content"):
        no_data.apply_event("message_stop", {"type": "message_stop"})
    with pytest.raises(AnthropicError, match="appeared more than once"):
        no_data.apply_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )

    no_usage = anthropic_started(profile=AnthropicProfile(stream_usage=False))
    anthropic_start_block(no_usage, {"type": "text", "text": "x"})
    anthropic_stop(no_usage)
    _, usage = no_usage.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    )
    assert usage == []
    no_usage.apply_event("message_stop", {"type": "message_stop"})
    no_usage.completed_response()
    with pytest.raises(AnthropicError, match="after message_stop"):
        no_usage.apply_event("ping", {"type": "ping"})
