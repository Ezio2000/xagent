from __future__ import annotations

import base64

import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    Message,
    ToolCall,
    ToolChoice,
    ToolFailure,
    ToolSpec,
    ToolSuccess,
)
from jharness.models.anthropic import AnthropicError, AnthropicProfile
from jharness.models.anthropic.messages_api.messages import (
    decode_content_blocks,
    encode_message,
    encode_messages,
    encode_user_content_part,
)
from jharness.models.anthropic.messages_api.tools import (
    decode_tool_uses,
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
from jharness.models.openai.chat_completions.tools import (
    decode_tool_calls,
)
from jharness.models.openai.chat_completions.tools import (
    encode_tool_choice as encode_openai_choice,
)
from jharness.models.openai.chat_completions.tools import (
    encode_tools as encode_openai_tools,
)


def tool_message(call_id: str, *, failure: bool = False) -> Message:
    parts = (ContentPart.text_part("result"),)
    outcome = (
        ToolFailure.from_error("failed", "tool failed")
        if failure
        else ToolSuccess(parts, {"value": 1})
    )
    return Message.tool(call_id, outcome)


def test_openai_message_codec_covers_roles_multimodal_and_native_parts() -> None:
    profile = OpenAIChatCompletionsProfile(
        supports_image_input=True,
        supports_video_input=True,
        supports_file_input=True,
        system_content_mode="parts",
    )
    call = ToolCall("call-1", "search", {"q": "x"})

    assert encode_chat_message(Message.external("callback"), profile) == {
        "role": "user",
        "content": "callback",
    }
    assert encode_chat_message(Message.system("policy"), profile)["content"] == [
        {"type": "text", "text": "policy"}
    ]
    assistant = Message.assistant(
        (
            ContentPart.text_part("cannot"),
            ContentPart(type="refusal", text="no"),
        ),
        tool_calls=(call,),
    )
    encoded_assistant = encode_chat_message(assistant, profile)
    assert encoded_assistant["tool_calls"][0]["id"] == "call-1"
    assert encode_chat_message(tool_message("call-1"), profile) == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result",
    }

    assert (
        encode_content_part(ContentPart("image", uri="https://x/image.png"), profile)["type"]
        == "image_url"
    )
    assert (
        encode_content_part(ContentPart("video", uri="https://x/video.mp4"), profile)["type"]
        == "video_url"
    )
    assert encode_content_part(ContentPart.artifact_part(ArtifactRef("file-1")), profile) == {
        "type": "file",
        "file": {"file_id": "file-1"},
    }
    assert (
        encode_content_part(
            ContentPart("file", uri="data:text/plain;base64,SGk=", name="note.txt"), profile
        )["file"]["filename"]
        == "note.txt"
    )

    decoded = decode_message_content(
        [
            {"type": "text", "text": "hello"},
            {"type": "refusal", "refusal": "no"},
        ]
    )
    assert [part.type for part in decoded] == ["text", "refusal"]
    assert decode_message_refusal("blocked")[0].type == "refusal"


def test_openai_tool_codec_validates_choices_and_arguments() -> None:
    spec = ToolSpec("search", "search", {"type": "object"})
    profile = OpenAIChatCompletionsProfile()

    assert encode_openai_tools((spec,), profile)[0]["function"]["name"] == "search"
    assert encode_openai_choice(ToolChoice(), tool_names={"search"}, profile=profile) == "auto"
    assert encode_openai_choice(
        ToolChoice("named", "search"), tool_names={"search"}, profile=profile
    ) == {"type": "function", "function": {"name": "search"}}
    calls = decode_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": {"q": "x"}},
            },
            {
                "id": "call-2",
                "function": {"name": "search", "arguments": ""},
            },
        ]
    )
    assert calls == [
        ToolCall("call-1", "search", {"q": "x"}),
        ToolCall("call-2", "search", {}),
    ]

    with pytest.raises(OpenAIChatCompletionsError, match="requires at least one"):
        encode_openai_choice(ToolChoice("required"), tool_names=set(), profile=profile)
    with pytest.raises(OpenAIChatCompletionsError, match="unavailable"):
        encode_openai_choice(ToolChoice("named", "other"), tool_names={"search"}, profile=profile)
    with pytest.raises(OpenAIChatCompletionsError, match="invalid JSON"):
        decode_tool_calls([{"id": "call", "function": {"name": "search", "arguments": "{"}}])


def test_openai_message_codec_rejects_unsupported_content() -> None:
    profile = OpenAIChatCompletionsProfile(
        supports_image_input=False,
        supports_video_input=False,
        supports_file_input=False,
    )
    with pytest.raises(OpenAIChatCompletionsError, match="image input"):
        encode_content_part(ContentPart("image", uri="https://x/image"), profile)
    with pytest.raises(OpenAIChatCompletionsError, match="unsupported content"):
        encode_content_part(ContentPart("audio", uri="https://x/audio"), profile)
    with pytest.raises(OpenAIChatCompletionsError, match="string, array, or null"):
        decode_message_content(3)
    with pytest.raises(OpenAIChatCompletionsError, match="non-empty"):
        decode_message_refusal("")


def test_anthropic_message_codec_covers_system_tools_and_native_parts() -> None:
    profile = AnthropicProfile(system_content_mode="blocks")
    call = ToolCall("call-1", "search", {"q": "x"})
    thinking = ContentPart(
        type="thinking",
        text="reason",
        data={"anthropic": {"type": "thinking", "thinking": "reason", "signature": "sig"}},
    )
    redacted = ContentPart(
        type="redacted_thinking",
        data={"anthropic": {"type": "redacted_thinking", "data": "secret"}},
    )
    system, messages = encode_messages(
        (
            Message.system("policy"),
            Message.user("hello"),
            Message.assistant((thinking, redacted), tool_calls=(call,)),
            tool_message("call-1", failure=True),
            Message.external("callback"),
        ),
        profile,
    )

    assert system == [{"type": "text", "text": "policy"}]
    assert messages[1]["content"][-1] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "search",
        "input": {"q": "x"},
    }
    assert messages[2]["content"][0]["is_error"] is True
    assert messages[-1] == {"role": "user", "content": "callback"}

    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "why", "signature": "sig"},
        {"type": "redacted_thinking", "data": "secret"},
        {"type": "tool_use", "id": "call-2", "name": "search", "input": {}},
    ]
    parts, uses = decode_content_blocks(blocks)
    assert [part.type for part in parts] == ["text", "thinking", "redacted_thinking"]
    assert decode_tool_uses(uses) == [ToolCall("call-2", "search", {})]
    assert encode_message(Message.external("callback"), profile)["role"] == "user"


def test_anthropic_media_and_tool_choice_codec() -> None:
    profile = AnthropicProfile()
    image = ContentPart(
        "image",
        uri="data:image/png;base64,aGVsbG8=",
        media_type="image/png",
    )
    pdf = ContentPart(
        "file",
        uri="data:application/pdf;base64,aGVsbG8=",
        media_type="application/pdf",
    )
    text = ContentPart(
        "file",
        uri=f"data:text/plain;base64,{base64.b64encode(b'hello').decode()}",
        media_type="text/plain",
    )

    assert encode_user_content_part(image, profile)["source"]["type"] == "base64"
    default_media = encode_user_content_part(
        ContentPart("image", uri="data:;base64,aGVsbG8="),
        profile,
    )
    assert default_media["source"]["media_type"] == "application/octet-stream"
    assert encode_user_content_part(pdf, profile)["source"]["media_type"] == "application/pdf"
    assert encode_user_content_part(text, profile)["source"]["data"] == "hello"
    assert encode_user_content_part(ContentPart.artifact_part(ArtifactRef("file-1")), profile)[
        "source"
    ] == {
        "type": "file",
        "file_id": "file-1",
    }
    spec = ToolSpec("search", "search", {"type": "object"})
    assert encode_anthropic_tools((spec,), profile)[0]["name"] == "search"
    assert encode_anthropic_choice(
        ToolChoice("required", allow_parallel_tool_calls=False),
        tool_names={"search"},
        profile=profile,
    ) == {"type": "any", "disable_parallel_tool_use": True}


def test_anthropic_codec_rejects_invalid_roles_media_and_blocks() -> None:
    profile = AnthropicProfile(supports_image_input=False, supports_file_input=False)
    with pytest.raises(AnthropicError, match="mid-conversation"):
        encode_messages(
            (Message.user("hello"), Message.system("late")),
            profile,
        )
    with pytest.raises(AnthropicError, match="image input"):
        encode_user_content_part(ContentPart("image", uri="https://x/image"), profile)
    with pytest.raises(AnthropicError, match="video input"):
        encode_user_content_part(ContentPart("video", uri="https://x/video"), profile)
    with pytest.raises(AnthropicError, match="non-empty text"):
        decode_content_blocks([{"type": "text", "text": ""}])
    with pytest.raises(AnthropicError, match="invalid JSON"):
        decode_tool_uses([{"id": "call", "name": "tool", "input": "{"}])
