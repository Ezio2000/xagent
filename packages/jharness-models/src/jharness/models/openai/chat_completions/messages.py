"""Message conversion for OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import ContentPart, Message
from jharness.models.openai.errors import OPENAI_JSON, OpenAIChatCompletionsError
from jharness.models.openai.profiles import OpenAIChatCompletionsProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]

_ASSISTANT_NATIVE_PART_TYPES = {"refusal"}


def encode_chat_message(
    message: Message,
    profile: OpenAIChatCompletionsProfile,
) -> JsonObject:
    role = "user" if message.role == "external" else message.role
    if role == "tool":
        if message.tool_call_id is None or message.outcome is None:
            raise OpenAIChatCompletionsError("tool messages require tool_call_id and outcome")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text_only_content(message.outcome.parts, "tool"),
        }

    reasoning_content = None
    content_parts = message.parts
    if role == "assistant":
        reasoning_content, content_parts = _extract_assistant_reasoning(message.parts, profile)
    content = encode_message_content(content_parts, role, profile)
    data: JsonObject = {"role": role, "content": content}
    if role == "assistant":
        if reasoning_content is not None:
            data["reasoning_content"] = reasoning_content
        if (
            message.tool_calls
            and profile.reasoning_content_mode == "required_with_tools"
            and reasoning_content is None
        ):
            raise OpenAIChatCompletionsError(
                f"{profile.name} requires non-empty reasoning content for assistant tool calls"
            )
        if message.tool_calls:
            from jharness.models.openai.chat_completions.tools import (
                encode_assistant_tool_calls,
            )

            data["tool_calls"] = encode_assistant_tool_calls(message.tool_calls)
            if content == "" and not profile.requires_assistant_content_for_tool_calls:
                data["content"] = None
    return data


def encode_message_content(
    parts: Sequence[ContentPart],
    role: str,
    profile: OpenAIChatCompletionsProfile,
) -> str | list[JsonObject]:
    if role == "system" and profile.system_content_mode == "string":
        return _text_only_content(parts, "system")
    if role == "assistant":
        return _encode_assistant_content(parts)
    if role == "tool":
        return _text_only_content(parts, role)
    if not parts:
        return ""
    if all(part.type == "text" for part in parts) and not (
        role == "system" and profile.system_content_mode == "parts"
    ):
        return "".join(part.text or "" for part in parts)
    return [encode_content_part(part, profile) for part in parts]


def encode_content_part(part: ContentPart, profile: OpenAIChatCompletionsProfile) -> JsonObject:
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "image":
        if not profile.supports_image_input:
            raise OpenAIChatCompletionsError(f"{profile.name} does not support image input")
        uri = _required_uri(part, "image")
        return {"type": "image_url", "image_url": {"url": uri}}
    if part.type == "video":
        if not profile.supports_video_input:
            raise OpenAIChatCompletionsError(f"{profile.name} does not support video input")
        uri = _required_uri(part, "video")
        return {"type": "video_url", "video_url": {"url": uri}}
    if part.type in {"artifact", "file"}:
        if not profile.supports_file_input:
            raise OpenAIChatCompletionsError(f"{profile.name} does not support file input")
        if part.artifact is not None:
            return {"type": "file", "file": {"file_id": part.artifact.ref}}
        uri = _required_uri(part, "file")
        return {
            "type": "file",
            "file": {
                "file_data": uri,
                "filename": part.name or "file",
            },
        }
    raise OpenAIChatCompletionsError(
        f"unsupported content part type for Chat Completions: {part.type}"
    )


def decode_message_content(value: object) -> list[ContentPart]:
    if value is None:
        return []
    if isinstance(value, str):
        return [ContentPart.text_part(value)] if value else []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise OpenAIChatCompletionsError(
            "chat completion message content must be a string, array, or null"
        )
    decoded = (_decode_message_content_part(item) for item in cast(Sequence[object], value))
    return [part for part in decoded if part is not None]


def _decode_message_content_part(value: object) -> ContentPart | None:
    mapping = OPENAI_JSON.mapping(value, "chat completion content part")
    part_type = mapping.get("type")
    if part_type == "text":
        return _decode_text_content_part(mapping)
    if not isinstance(part_type, str) or not part_type:
        raise OpenAIChatCompletionsError("chat completion content part requires non-empty type")
    if part_type not in _ASSISTANT_NATIVE_PART_TYPES:
        raise OpenAIChatCompletionsError(
            f"unsupported chat completion assistant content part: {part_type}"
        )
    return _decode_refusal_content_part(mapping)


def _decode_text_content_part(mapping: Mapping[str, Any]) -> ContentPart | None:
    text = mapping.get("text", "")
    if not isinstance(text, str):
        raise OpenAIChatCompletionsError("chat completion text part must contain text string")
    return ContentPart.text_part(text) if text else None


def _decode_refusal_content_part(mapping: Mapping[str, Any]) -> ContentPart:
    refusal = mapping.get("refusal")
    if not isinstance(refusal, str) or not refusal:
        raise OpenAIChatCompletionsError(
            "chat completion refusal part requires non-empty refusal text"
        )
    return ContentPart(
        type="refusal",
        text=refusal,
        data={"openai": dict(mapping)},
    )


def decode_message_refusal(value: object) -> list[ContentPart]:
    if value is None:
        return []
    if not isinstance(value, str) or not value:
        raise OpenAIChatCompletionsError(
            "chat completion message refusal must be a non-empty string or null"
        )
    block = {"type": "refusal", "refusal": value}
    return [ContentPart(type="refusal", text=value, data={"openai": block})]


def _encode_assistant_content(
    parts: Sequence[ContentPart],
) -> str | list[JsonObject]:
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    blocks: list[JsonObject] = []
    for part in parts:
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text or ""})
            continue
        if part.type != "refusal":
            raise OpenAIChatCompletionsError(
                f"unsupported assistant content part for Chat Completions: {part.type}"
            )
        wire_block = _wire_block(part)
        if wire_block is None:
            refusal = part.text
            if not isinstance(refusal, str) or not refusal:
                raise OpenAIChatCompletionsError(
                    "assistant refusal parts require non-empty text or OpenAI data"
                )
            wire_block = {"type": "refusal", "refusal": refusal}
        blocks.append(wire_block)
    return blocks


def _extract_assistant_reasoning(
    parts: Sequence[ContentPart],
    profile: OpenAIChatCompletionsProfile,
) -> tuple[str | None, tuple[ContentPart, ...]]:
    reasoning_parts = [part for part in parts if part.type == "reasoning"]
    if not reasoning_parts:
        return None, tuple(parts)
    if profile.reasoning_content_mode == "live_only":
        raise OpenAIChatCompletionsError(
            f"{profile.name} does not support reasoning content in assistant messages"
        )
    chunks: list[str] = []
    for part in reasoning_parts:
        if not isinstance(part.text, str):
            raise OpenAIChatCompletionsError("assistant reasoning parts require a text string")
        chunks.append(part.text)
    reasoning_content = "".join(chunks)
    return (
        reasoning_content or None,
        tuple(part for part in parts if part.type != "reasoning"),
    )


def _wire_block(part: ContentPart) -> JsonObject | None:
    raw = part.data.get("openai")
    if raw is None:
        return None
    block = OPENAI_JSON.mapping(raw, "OpenAI-native content part")
    block_type = block.get("type")
    if block_type not in _ASSISTANT_NATIVE_PART_TYPES:
        raise OpenAIChatCompletionsError(
            f"unsupported OpenAI-native assistant content part: {block_type}"
        )
    if block_type != part.type:
        raise OpenAIChatCompletionsError(
            "OpenAI-native content type must match the ContentPart type"
        )
    refusal = block.get("refusal")
    if not isinstance(refusal, str) or not refusal:
        raise OpenAIChatCompletionsError(
            "OpenAI-native refusal parts require non-empty refusal text"
        )
    return dict(block)


def _text_only_content(parts: Sequence[ContentPart], role: str) -> str:
    unsupported = [part.type for part in parts if part.type != "text"]
    if unsupported:
        raise OpenAIChatCompletionsError(f"{role} messages only support text content parts")
    return "".join(part.text or "" for part in parts)


def _required_uri(part: ContentPart, label: str) -> str:
    if part.uri is None:
        raise OpenAIChatCompletionsError(f"{label} input requires a uri")
    return part.uri
