"""Message conversion for Anthropic Messages."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from jharness.kernel import ContentPart, Message
from jharness.models.anthropic.errors import ANTHROPIC_JSON, AnthropicError
from jharness.models.anthropic.profiles import AnthropicProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]

_NATIVE_BLOCK_TYPES_BY_ROLE = {
    "system": {"text"},
    "user": {"text", "image", "document"},
    "assistant": {"thinking", "redacted_thinking"},
}


def encode_messages(
    messages: Sequence[Message],
    profile: AnthropicProfile,
) -> tuple[str | list[JsonObject] | None, list[JsonObject]]:
    message_list = list(messages)
    system_blocks: list[JsonObject] = []
    encoded_messages: list[JsonObject] = []
    tool_result_blocks: list[JsonObject] = []
    conversation_started = False
    for index, message in enumerate(message_list):
        if message.role == "system":
            if tool_result_blocks:
                encoded_messages.append({"role": "user", "content": tool_result_blocks})
                tool_result_blocks = []
            if not conversation_started:
                system_blocks.extend(_encode_system_parts(message.parts, profile))
                continue
            if not profile.supports_mid_conversation_system:
                raise AnthropicError(
                    f"{profile.name} does not support mid-conversation system messages"
                )
            _validate_mid_conversation_system_position(
                encoded_messages,
                _next_message(message_list, index),
            )
            encoded_messages.append(encode_system_message(message, profile))
            continue
        if message.role == "tool":
            conversation_started = True
            tool_result_blocks.append(encode_tool_result_block(message, profile))
            continue
        if tool_result_blocks:
            encoded_messages.append({"role": "user", "content": tool_result_blocks})
            tool_result_blocks = []
        encoded_messages.append(encode_message(message, profile))
        conversation_started = True
    if tool_result_blocks:
        encoded_messages.append({"role": "user", "content": tool_result_blocks})
    return _system_value(system_blocks, profile), encoded_messages


def _next_message(messages: Sequence[Message], index: int) -> Message | None:
    if index + 1 >= len(messages):
        return None
    return messages[index + 1]


def _validate_mid_conversation_system_position(
    encoded_messages: Sequence[JsonObject],
    next_message: Message | None,
) -> None:
    if not encoded_messages or encoded_messages[-1].get("role") != "user":
        raise AnthropicError("mid-conversation system messages must follow a user turn")
    if next_message is None:
        return
    if next_message.role != "assistant":
        raise AnthropicError(
            "mid-conversation system messages must be final or precede an assistant turn"
        )


def encode_message(
    message: Message,
    profile: AnthropicProfile,
) -> JsonObject:
    if message.role == "tool":
        return {"role": "user", "content": [encode_tool_result_block(message, profile)]}

    role = "user" if message.role == "external" else message.role
    if role not in {"user", "assistant"}:
        raise AnthropicError(f"unsupported Anthropic message role: {message.role}")
    content = encode_message_content(message.parts, role, profile)
    if role == "assistant" and message.tool_calls:
        blocks = _content_blocks(content)
        from jharness.models.anthropic.messages_api.tools import encode_assistant_tool_uses

        blocks.extend(encode_assistant_tool_uses(message.tool_calls))
        content = blocks
    return {"role": role, "content": content}


def encode_system_message(message: Message, profile: AnthropicProfile) -> JsonObject:
    content = _system_value(_encode_system_parts(message.parts, profile), profile)
    if content is None:
        content = ""
    return {"role": "system", "content": content}


def encode_tool_result_block(message: Message, profile: AnthropicProfile) -> JsonObject:
    if message.tool_call_id is None or message.outcome is None:
        raise AnthropicError("tool messages require tool_call_id and outcome")
    block: JsonObject = {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": encode_tool_result_content(message.outcome.parts, profile),
    }
    if message.outcome.kind == "failure":
        block["is_error"] = True
    return block


def encode_message_content(
    parts: Sequence[ContentPart],
    role: str,
    profile: AnthropicProfile,
) -> str | list[JsonObject]:
    if role == "assistant":
        blocks = [_encode_assistant_part(part, profile) for part in parts]
        if not blocks:
            return ""
        if all(block.get("type") == "text" for block in blocks):
            return "".join(cast(str, block.get("text", "")) for block in blocks)
        return blocks
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    return [encode_user_content_part(part, profile) for part in parts]


def encode_user_content_part(part: ContentPart, profile: AnthropicProfile) -> JsonObject:
    wire_block = _wire_block(part, role="user", profile=profile)
    if wire_block is not None:
        return wire_block
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "image":
        if not profile.supports_image_input:
            raise AnthropicError(f"{profile.name} does not support image input")
        return {"type": "image", "source": _encode_media_source(part, "image")}
    if part.type in {"artifact", "file"}:
        if not profile.supports_file_input:
            raise AnthropicError(f"{profile.name} does not support file input")
        block: JsonObject = {"type": "document", "source": _encode_media_source(part, "file")}
        name = part.artifact.name if part.artifact is not None else part.name
        if name:
            block["title"] = name
        return block
    if part.type == "video":
        raise AnthropicError(f"{profile.name} does not support video input")
    raise AnthropicError(f"unsupported content part type for Anthropic Messages: {part.type}")


def encode_tool_result_content(
    parts: Sequence[ContentPart], profile: AnthropicProfile
) -> str | list[JsonObject]:
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    return [encode_user_content_part(part, profile) for part in parts]


def decode_content_blocks(value: object) -> tuple[list[ContentPart], list[Mapping[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise AnthropicError("Anthropic response content must be an array")
    parts: list[ContentPart] = []
    tool_uses: list[Mapping[str, Any]] = []
    for item in cast(Sequence[object], value):
        block = ANTHROPIC_JSON.mapping(item, "Anthropic content block")
        block_type = _content_block_type(block)
        if block_type == "tool_use":
            tool_uses.append(block)
            continue
        decoder = _CONTENT_BLOCK_DECODERS.get(block_type)
        if decoder is None:
            raise AnthropicError(f"unsupported Anthropic assistant content block: {block_type}")
        parts.append(decoder(block))
    return parts, tool_uses


def _content_block_type(block: Mapping[str, Any]) -> str:
    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type:
        raise AnthropicError("Anthropic content block requires non-empty type")
    return block_type


def _decode_text_block(block: Mapping[str, Any]) -> ContentPart:
    text = block.get("text")
    if not isinstance(text, str) or not text:
        raise AnthropicError("Anthropic text block requires non-empty text")
    return ContentPart.text_part(text)


def _decode_thinking_block(block: Mapping[str, Any]) -> ContentPart:
    thinking = block.get("thinking")
    if not isinstance(thinking, str) or not thinking:
        raise AnthropicError("Anthropic thinking block requires non-empty thinking text")
    signature = block.get("signature")
    if signature is not None and not isinstance(signature, str):
        raise AnthropicError("Anthropic thinking signature must be a string")
    return ContentPart(
        type="thinking",
        text=thinking,
        data={"anthropic": dict(block)},
    )


def _decode_redacted_thinking_block(block: Mapping[str, Any]) -> ContentPart:
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise AnthropicError("Anthropic redacted_thinking block requires non-empty data")
    return ContentPart(
        type="redacted_thinking",
        data={"anthropic": dict(block)},
    )


_CONTENT_BLOCK_DECODERS: Mapping[str, Callable[[Mapping[str, Any]], ContentPart]] = {
    "redacted_thinking": _decode_redacted_thinking_block,
    "text": _decode_text_block,
    "thinking": _decode_thinking_block,
}


def _encode_system_parts(
    parts: Sequence[ContentPart], profile: AnthropicProfile
) -> list[JsonObject]:
    blocks: list[JsonObject] = []
    for part in parts:
        wire_block = _wire_block(part, role="system", profile=profile)
        if wire_block is not None:
            blocks.append(wire_block)
            continue
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text or ""})
            continue
        raise AnthropicError(f"system messages do not support {part.type!r} content parts")
    if profile.system_content_mode == "string" and any(
        block.get("type") != "text" for block in blocks
    ):
        raise AnthropicError("system_content_mode='string' only supports text system blocks")
    return blocks


def _system_value(
    blocks: Sequence[JsonObject], profile: AnthropicProfile
) -> str | list[JsonObject] | None:
    if not blocks:
        return None
    if profile.system_content_mode == "blocks":
        return [dict(block) for block in blocks]
    return "\n\n".join(cast(str, block.get("text", "")) for block in blocks)


def _encode_assistant_part(part: ContentPart, profile: AnthropicProfile) -> JsonObject:
    wire_block = _wire_block(part, role="assistant", profile=profile)
    if wire_block is not None:
        return wire_block
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "thinking":
        block: JsonObject = {"type": "thinking", "thinking": part.text or ""}
        signature = _anthropic_metadata_str(part, "signature")
        if signature is not None:
            block["signature"] = signature
        return block
    if part.type == "redacted_thinking":
        if not profile.supports_redacted_thinking:
            raise AnthropicError(f"{profile.name} does not support redacted_thinking")
        data = _anthropic_metadata_str(part, "data")
        if data is None:
            raise AnthropicError("redacted_thinking parts require anthropic metadata data")
        return {"type": "redacted_thinking", "data": data}
    raise AnthropicError("assistant messages only support text or Anthropic-native parts")


def _wire_block(
    part: ContentPart,
    *,
    role: str,
    profile: AnthropicProfile,
) -> JsonObject | None:
    value = part.data.get("anthropic")
    if value is None:
        return None
    mapping = ANTHROPIC_JSON.mapping(value, "Anthropic-native content part")
    block_type = mapping.get("type")
    if not isinstance(block_type, str) or not block_type:
        raise AnthropicError("Anthropic-native content part requires a non-empty type")
    allowed_types = _NATIVE_BLOCK_TYPES_BY_ROLE[role]
    if block_type not in allowed_types:
        raise AnthropicError(
            f"Anthropic-native {block_type!r} blocks are not allowed for {role} messages"
        )
    if role == "system" and profile.system_content_mode != "blocks":
        raise AnthropicError("Anthropic-native system blocks require system_content_mode='blocks'")
    if block_type == "image" and not profile.supports_image_input:
        raise AnthropicError(f"{profile.name} does not support image input")
    if block_type == "document" and not profile.supports_file_input:
        raise AnthropicError(f"{profile.name} does not support file input")
    if block_type == "redacted_thinking" and not profile.supports_redacted_thinking:
        raise AnthropicError(f"{profile.name} does not support redacted_thinking")
    _validate_wire_block(mapping, block_type)
    return dict(mapping)


def _validate_wire_block(block: Mapping[str, Any], block_type: str) -> None:
    if block_type == "text":
        if not isinstance(block.get("text"), str):
            raise AnthropicError("Anthropic-native text blocks require text")
        return
    if block_type in {"image", "document"}:
        ANTHROPIC_JSON.mapping(block.get("source"), f"Anthropic-native {block_type} source")
        return
    if block_type == "thinking":
        if not isinstance(block.get("thinking"), str):
            raise AnthropicError("Anthropic-native thinking blocks require thinking text")
        signature = block.get("signature")
        if signature is not None and not isinstance(signature, str):
            raise AnthropicError("Anthropic-native thinking signature must be a string")
        return
    if block_type == "redacted_thinking":
        data = block.get("data")
        if not isinstance(data, str) or not data:
            raise AnthropicError("Anthropic-native redacted_thinking blocks require non-empty data")
        return
    raise AnthropicError(f"unsupported Anthropic-native content block: {block_type}")


def _anthropic_metadata_str(part: ContentPart, key: str) -> str | None:
    metadata = part.metadata.get("anthropic")
    if not isinstance(metadata, Mapping):
        return None
    metadata_mapping = cast(Mapping[str, object], metadata)
    value = metadata_mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnthropicError(f"Anthropic metadata {key} must be a string")
    return value


def _content_blocks(value: str | list[JsonObject]) -> list[JsonObject]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    return [dict(block) for block in value]


def _encode_media_source(part: ContentPart, label: str) -> JsonObject:
    if part.artifact is not None:
        return {"type": "file", "file_id": part.artifact.ref}
    if part.uri is None:
        raise AnthropicError(f"{label} input requires a uri or artifact")
    if part.uri.startswith("data:"):
        media_type, data = _parse_data_url(part.uri)
        media_type = part.media_type or media_type
        if label == "file":
            if media_type == "application/pdf":
                return {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                }
            if media_type == "text/plain":
                return {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": _decode_base64_text(data),
                }
            raise AnthropicError(
                "Anthropic document data URLs must use application/pdf or text/plain"
            )
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    if label == "file" and part.media_type is not None and part.media_type != "application/pdf":
        raise AnthropicError("Anthropic document URL inputs must be PDFs")
    return {"type": "url", "url": part.uri}


def _parse_data_url(uri: str) -> tuple[str, str]:
    header, separator, data = uri.partition(",")
    if separator != "," or not header.startswith("data:"):
        raise AnthropicError("data URL input must contain media type and base64 data")
    metadata = header.removeprefix("data:")
    values = metadata.split(";")
    media_type = values[0] or "application/octet-stream"
    if "base64" not in values[1:]:
        raise AnthropicError("data URL input must use base64 encoding")
    if not data:
        raise AnthropicError("data URL input requires base64 data")
    return media_type, data


def _decode_base64_text(data: str) -> str:
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AnthropicError("text/plain data URL contains invalid base64 data") from exc
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnthropicError("text/plain data URL must decode as UTF-8") from exc
