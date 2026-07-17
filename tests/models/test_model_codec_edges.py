from __future__ import annotations

import base64
from typing import Any

import pytest

from jharness.kernel import ArtifactRef, ContentPart, Message, ToolCall, ToolChoice, ToolSpec
from jharness.models.anthropic import AnthropicError, AnthropicProfile
from jharness.models.anthropic.messages_api.messages import (
    decode_content_blocks,
    encode_system_message,
    encode_tool_result_content,
    encode_user_content_part,
)
from jharness.models.anthropic.messages_api.messages import (
    encode_message as encode_anthropic_message,
)
from jharness.models.anthropic.messages_api.messages import (
    encode_message_content as encode_anthropic_content,
)
from jharness.models.anthropic.messages_api.messages import (
    encode_messages as encode_anthropic_messages,
)
from jharness.models.anthropic.messages_api.tools import (
    decode_tool_uses,
    encode_assistant_tool_uses,
)
from jharness.models.anthropic.messages_api.tools import (
    encode_tool_choice as encode_anthropic_choice,
)
from jharness.models.anthropic.messages_api.tools import (
    encode_tools as encode_anthropic_tools,
)
from jharness.models.openai import OpenAIChatCompletionsError, OpenAIChatCompletionsProfile
from jharness.models.openai.chat_completions.messages import (
    decode_message_content,
    decode_message_refusal,
    encode_chat_message,
    encode_content_part,
)
from jharness.models.openai.chat_completions.messages import (
    encode_message_content as encode_openai_content,
)
from jharness.models.openai.chat_completions.tools import (
    decode_tool_calls,
    encode_assistant_tool_calls,
)
from jharness.models.openai.chat_completions.tools import (
    encode_tool_choice as encode_openai_choice,
)
from jharness.models.openai.chat_completions.tools import (
    encode_tools as encode_openai_tools,
)


def test_openai_content_edges_and_incremental_native_parts() -> None:
    profile = OpenAIChatCompletionsProfile(
        supports_image_input=True,
        supports_video_input=True,
        supports_file_input=True,
    )
    call = ToolCall("call", "lookup")
    assert encode_chat_message(Message.assistant(tool_calls=(call,)), profile)["content"] is None
    assert encode_openai_content((), "user", profile) == ""
    assert encode_openai_content((ContentPart.text_part("ok"),), "tool", profile) == "ok"
    assert decode_message_content(None) == []
    assert decode_message_content("") == []
    assert decode_message_content([{"type": "text", "text": ""}]) == []
    assert decode_message_refusal(None) == []
    assert (
        encode_content_part(ContentPart("file", uri="data:,x"), profile)["file"]["filename"]
        == "file"
    )
    assert encode_assistant_tool_calls((call,))[0]["function"]["arguments"] == "{}"

    bare_refusal = ContentPart("refusal", text="no")
    assert encode_chat_message(Message.assistant((bare_refusal,)), profile)["content"] == [
        {"type": "refusal", "refusal": "no"}
    ]
    native_refusal = ContentPart(
        "refusal",
        text="no",
        data={"openai": {"type": "refusal", "refusal": "native"}},
    )
    assert encode_chat_message(Message.assistant((native_refusal,)), profile)["content"] == [
        {"type": "refusal", "refusal": "native"}
    ]


@pytest.mark.parametrize(
    "part,pattern",
    [
        (ContentPart("image"), "requires a uri"),
        (ContentPart("video"), "requires a uri"),
        (ContentPart("file"), "requires a uri"),
        (ContentPart("audio", uri="https://x/audio"), "unsupported content"),
    ],
)
def test_openai_content_rejects_missing_or_unsupported_sources(
    part: ContentPart, pattern: str
) -> None:
    profile = OpenAIChatCompletionsProfile(
        supports_image_input=True,
        supports_video_input=True,
        supports_file_input=True,
    )
    with pytest.raises(OpenAIChatCompletionsError, match=pattern):
        encode_content_part(part, profile)


def test_openai_content_rejects_disabled_capabilities_and_bad_native_data() -> None:
    disabled = OpenAIChatCompletionsProfile(
        supports_image_input=False,
        supports_video_input=False,
        supports_file_input=False,
    )
    for part, pattern in (
        (ContentPart("image", uri="https://x/image"), "image input"),
        (ContentPart("video", uri="https://x/video"), "video input"),
        (ContentPart("file", uri="https://x/file"), "file input"),
    ):
        with pytest.raises(OpenAIChatCompletionsError, match=pattern):
            encode_content_part(part, disabled)

    profile = OpenAIChatCompletionsProfile(supports_image_input=True)
    with pytest.raises(OpenAIChatCompletionsError, match="only support text"):
        encode_openai_content((ContentPart("image", uri="https://x/image"),), "system", profile)
    for part, pattern in (
        (ContentPart("audio", uri="x"), "unsupported assistant"),
        (ContentPart("refusal", text="no", data={"openai": "bad"}), "must be an object"),
        (
            ContentPart(
                "refusal",
                text="no",
                data={"openai": {"type": "other", "refusal": "no"}},
            ),
            "unsupported OpenAI-native",
        ),
        (
            ContentPart(
                "refusal",
                text="no",
                data={"openai": {"type": "refusal", "refusal": ""}},
            ),
            "non-empty refusal",
        ),
    ):
        with pytest.raises(OpenAIChatCompletionsError, match=pattern):
            encode_chat_message(Message.assistant((part,)), profile)


@pytest.mark.parametrize(
    "value,pattern",
    [
        ([1], "must be an object"),
        ([{"other": None}], "non-empty type"),
        ([{"type": "image"}], "unsupported"),
        ([{"type": "text", "text": 1}], "text string"),
        ([{"type": "refusal", "refusal": ""}], "non-empty refusal"),
    ],
)
def test_openai_content_decoder_rejects_invalid_blocks(value: object, pattern: str) -> None:
    with pytest.raises(OpenAIChatCompletionsError, match=pattern):
        decode_message_content(value)


def test_openai_tool_codec_edges() -> None:
    spec = ToolSpec("lookup", "lookup", {"type": "object"})
    profile = OpenAIChatCompletionsProfile()
    assert encode_openai_tools((), profile) == []
    with pytest.raises(OpenAIChatCompletionsError, match="does not support tools"):
        encode_openai_tools((spec,), OpenAIChatCompletionsProfile(supports_tools=False))
    assert encode_openai_choice(ToolChoice(), tool_names=set(), profile=profile) is None
    no_choice = OpenAIChatCompletionsProfile(supports_tool_choice=False)
    assert encode_openai_choice(ToolChoice(), tool_names={"lookup"}, profile=no_choice) is None
    with pytest.raises(OpenAIChatCompletionsError, match="does not support tool_choice"):
        encode_openai_choice(ToolChoice("none"), tool_names={"lookup"}, profile=no_choice)
    assert (
        encode_openai_choice(ToolChoice("required"), tool_names={"lookup"}, profile=profile)
        == "required"
    )
    assert decode_tool_calls(None) == []


@pytest.mark.parametrize(
    "value,pattern",
    [
        (1, "must be an array"),
        ([1], "must be an object"),
        ([{"type": "other"}], "unsupported"),
        ([{"id": "call", "function": 1}], "function must be an object"),
        ([{"id": 1, "function": {"name": "tool"}}], "id must be a string"),
        ([{"id": "", "function": {"name": "tool"}}], "id must not be empty"),
        ([{"id": "call", "function": {"name": ""}}], "name must not be empty"),
        (
            [{"id": "call", "function": {"name": "tool", "arguments": 1}}],
            "arguments must be a string",
        ),
        (
            [{"id": "call", "function": {"name": "tool", "arguments": "[]"}}],
            "decode to an object",
        ),
    ],
)
def test_openai_tool_decoder_rejects_invalid_values(value: object, pattern: str) -> None:
    with pytest.raises(OpenAIChatCompletionsError, match=pattern):
        decode_tool_calls(value)


def test_anthropic_message_grouping_and_mid_conversation_system_edges() -> None:
    blocks = AnthropicProfile(system_content_mode="blocks")
    system, messages = encode_anthropic_messages((Message.system("policy"),), blocks)
    assert system == [{"type": "text", "text": "policy"}]
    assert messages == []
    assert encode_system_message(Message.system("policy"), blocks) == {
        "role": "system",
        "content": [{"type": "text", "text": "policy"}],
    }
    with pytest.raises(AnthropicError, match="unsupported Anthropic message role"):
        encode_anthropic_message(Message.system("policy"), blocks)

    enabled = AnthropicProfile(supports_mid_conversation_system=True)
    _, final_system = encode_anthropic_messages(
        (Message.user("one"), Message.system("instruction")), enabled
    )
    assert final_system[-1] == {"role": "system", "content": "instruction"}
    _, before_assistant = encode_anthropic_messages(
        (
            Message.user("one"),
            Message.system("instruction"),
            Message.assistant((ContentPart.text_part("two"),)),
        ),
        enabled,
    )
    assert [message["role"] for message in before_assistant] == ["user", "system", "assistant"]
    with pytest.raises(AnthropicError, match="must follow a user"):
        encode_anthropic_messages(
            (
                Message.user("one"),
                Message.assistant((ContentPart.text_part("two"),)),
                Message.system("late"),
            ),
            enabled,
        )
    with pytest.raises(AnthropicError, match="precede an assistant"):
        encode_anthropic_messages(
            (Message.user("one"), Message.system("middle"), Message.user("two")),
            enabled,
        )


def test_anthropic_content_shapes_and_native_metadata() -> None:
    profile = AnthropicProfile()
    call = ToolCall("call", "lookup")
    assert encode_anthropic_content((), "user", profile) == ""
    assert encode_anthropic_content((ContentPart.text_part("a"),), "assistant", profile) == "a"
    assert encode_anthropic_message(Message.assistant(tool_calls=(call,)), profile)["content"] == [
        {"type": "tool_use", "id": "call", "name": "lookup", "input": {}}
    ]
    assert encode_tool_result_content((), profile) == ""
    assert encode_tool_result_content((ContentPart.text_part("x"),), profile) == "x"
    assert (
        encode_user_content_part(
            ContentPart.artifact_part(ArtifactRef("file", name="report.pdf")), profile
        )["title"]
        == "report.pdf"
    )
    assert encode_user_content_part(ContentPart("image", uri="https://x/image"), profile)[
        "source"
    ] == {"type": "url", "url": "https://x/image"}

    thinking = ContentPart(
        "thinking",
        text="reason",
        metadata={"anthropic": {"signature": "sig"}},
    )
    redacted = ContentPart(
        "redacted_thinking",
        metadata={"anthropic": {"data": "secret"}},
    )
    encoded = encode_anthropic_message(Message.assistant((thinking, redacted)), profile)
    assert encoded["content"] == [
        {"type": "thinking", "thinking": "reason", "signature": "sig"},
        {"type": "redacted_thinking", "data": "secret"},
    ]
    with pytest.raises(AnthropicError, match="require anthropic metadata"):
        encode_anthropic_message(Message.assistant((ContentPart("redacted_thinking"),)), profile)
    with pytest.raises(AnthropicError, match="metadata signature must be a string"):
        encode_anthropic_message(
            Message.assistant(
                (
                    ContentPart(
                        "thinking",
                        text="reason",
                        metadata={"anthropic": {"signature": 1}},
                    ),
                )
            ),
            profile,
        )


@pytest.mark.parametrize(
    "block,pattern",
    [
        (1, "must be an object"),
        ({}, "non-empty type"),
        ({"type": "other"}, "unsupported"),
        ({"type": "text", "text": ""}, "non-empty text"),
        ({"type": "thinking", "thinking": ""}, "non-empty thinking"),
        ({"type": "thinking", "thinking": "ok", "signature": 1}, "signature must be"),
        ({"type": "redacted_thinking", "data": ""}, "non-empty data"),
    ],
)
def test_anthropic_content_decoder_rejects_invalid_blocks(block: object, pattern: str) -> None:
    with pytest.raises(AnthropicError, match=pattern):
        decode_content_blocks([block])


def test_anthropic_native_blocks_validate_role_shape_and_capabilities() -> None:
    profile = AnthropicProfile(system_content_mode="blocks")
    native_text = ContentPart(
        "opaque",
        data={"anthropic": {"type": "text", "text": "policy"}},
    )
    system, _ = encode_anthropic_messages((Message("system", (native_text,)),), profile)
    assert system == [{"type": "text", "text": "policy"}]
    with pytest.raises(AnthropicError, match="require system_content_mode='blocks'"):
        encode_anthropic_messages(
            (Message("system", (native_text,)),), AnthropicProfile(system_content_mode="string")
        )

    cases = (
        (
            ContentPart("opaque", data={"anthropic": {}}),
            profile,
            "non-empty type",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "thinking", "thinking": "x"}}),
            profile,
            "not allowed for user",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "image", "source": {}}}),
            AnthropicProfile(supports_image_input=False),
            "does not support image",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "document", "source": {}}}),
            AnthropicProfile(supports_file_input=False),
            "does not support file",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "image", "source": 1}}),
            profile,
            "source must be an object",
        ),
    )
    for part, selected_profile, pattern in cases:
        with pytest.raises(AnthropicError, match=pattern):
            encode_user_content_part(part, selected_profile)


@pytest.mark.parametrize(
    "part,pattern",
    [
        (ContentPart("file"), "requires a uri or artifact"),
        (
            ContentPart("file", uri="data:application/json;base64,e30="),
            "must use application/pdf or text/plain",
        ),
        (ContentPart("file", uri="https://x/file", media_type="text/plain"), "must be PDFs"),
        (ContentPart("file", uri="data:text/plain,abc"), "must use base64"),
        (ContentPart("file", uri="data:text/plain;base64,"), "requires base64 data"),
        (ContentPart("file", uri="data:text/plain;base64,***"), "invalid base64"),
        (
            ContentPart(
                "file",
                uri="data:text/plain;base64," + base64.b64encode(b"\xff").decode(),
            ),
            "decode as UTF-8",
        ),
    ],
)
def test_anthropic_media_rejects_invalid_sources(part: ContentPart, pattern: str) -> None:
    with pytest.raises(AnthropicError, match=pattern):
        encode_user_content_part(part, AnthropicProfile())


def test_anthropic_tool_codec_edges() -> None:
    spec = ToolSpec("lookup", "lookup", {"type": "object"})
    profile = AnthropicProfile()
    call = ToolCall("call", "lookup", {"x": 1})
    assert encode_anthropic_tools((), profile) == []
    with pytest.raises(AnthropicError, match="does not support tools"):
        encode_anthropic_tools((spec,), AnthropicProfile(supports_tools=False))
    assert encode_anthropic_choice(ToolChoice(), tool_names=set(), profile=profile) is None
    no_choice = AnthropicProfile(supports_tool_choice=False)
    assert encode_anthropic_choice(ToolChoice(), tool_names={"lookup"}, profile=no_choice) is None
    with pytest.raises(AnthropicError, match="does not support tool_choice"):
        encode_anthropic_choice(ToolChoice("none"), tool_names={"lookup"}, profile=no_choice)
    assert encode_anthropic_choice(ToolChoice("none"), tool_names={"lookup"}, profile=profile) == {
        "type": "none"
    }
    assert encode_assistant_tool_uses((call,))[0]["input"] == {"x": 1}
    assert decode_tool_uses([{"id": "call", "name": "lookup", "input": None}]) == [
        ToolCall("call", "lookup")
    ]
    assert decode_tool_uses([{"id": "call", "name": "lookup", "input": '{"x":1}'}]) == [call]


@pytest.mark.parametrize(
    "block,pattern",
    [
        ({"id": 1, "name": "tool"}, "id must be a string"),
        ({"id": "", "name": "tool"}, "id must not be empty"),
        ({"id": "call", "name": ""}, "name must not be empty"),
        ({"id": "call", "name": "tool", "input": 1}, "object or JSON string"),
        ({"id": "call", "name": "tool", "input": "{"}, "invalid JSON"),
        ({"id": "call", "name": "tool", "input": "[]"}, "decode to an object"),
    ],
)
def test_anthropic_tool_decoder_rejects_invalid_values(block: dict[str, Any], pattern: str) -> None:
    with pytest.raises(AnthropicError, match=pattern):
        decode_tool_uses([block])
