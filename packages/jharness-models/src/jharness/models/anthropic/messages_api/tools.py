"""Tool conversion for Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

from jharness.kernel import ToolCall, ToolChoice, ToolSpec, thaw_json_value
from jharness.models.anthropic.errors import ANTHROPIC_JSON, AnthropicError
from jharness.models.anthropic.profiles import AnthropicProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]


def encode_tools(tools: Sequence[ToolSpec], profile: AnthropicProfile) -> list[JsonObject]:
    if not tools:
        return []
    if not profile.supports_tools:
        raise AnthropicError(f"{profile.name} does not support tools")
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": thaw_json_value(tool.input_schema),
        }
        for tool in tools
    ]


def encode_tool_choice(
    choice: ToolChoice,
    *,
    tool_names: Collection[str],
    profile: AnthropicProfile,
) -> JsonObject | None:
    if not tool_names:
        if choice.type in {"required", "named"}:
            raise AnthropicError(f"tool_choice={choice.type!r} requires at least one tool")
        return None
    if not profile.supports_tool_choice:
        if choice.type == "auto":
            return None
        raise AnthropicError(f"{profile.name} does not support tool_choice")
    if choice.type == "named":
        if choice.name is None or choice.name not in tool_names:
            raise AnthropicError(f"tool_choice names an unavailable tool: {choice.name}")
        value: JsonObject = {"type": "tool", "name": choice.name}
    else:
        value = {
            "type": {
                "auto": "auto",
                "none": "none",
                "required": "any",
            }[choice.type]
        }
    if choice.type != "none" and profile.supports_parallel_tool_call_control:
        value["disable_parallel_tool_use"] = not choice.allow_parallel_tool_calls
    return value


def encode_assistant_tool_uses(calls: Sequence[ToolCall]) -> list[JsonObject]:
    return [
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": thaw_json_value(call.arguments),
        }
        for call in calls
    ]


def decode_tool_uses(blocks: Sequence[Mapping[str, Any]]) -> list[ToolCall]:
    return [
        ToolCall(
            id=ANTHROPIC_JSON.required_string(block.get("id"), "Anthropic tool_use id"),
            name=ANTHROPIC_JSON.required_string(block.get("name"), "Anthropic tool_use name"),
            arguments=_decode_input(block.get("input")),
        )
        for block in blocks
    ]


def _decode_input(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if not isinstance(value, str):
        raise AnthropicError("Anthropic tool_use input must be an object or JSON string")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnthropicError("Anthropic tool_use input is invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise AnthropicError("Anthropic tool_use input must decode to an object")
    return cast(Mapping[str, Any], parsed)
