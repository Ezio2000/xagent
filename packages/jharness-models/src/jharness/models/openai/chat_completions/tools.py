"""Tool conversion for OpenAI Chat Completions."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

from jharness.kernel import ToolCall, ToolChoice, ToolSpec, thaw_json_value
from jharness.models.openai.errors import OPENAI_JSON, OpenAIChatCompletionsError
from jharness.models.openai.profiles import OpenAIChatCompletionsProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]


def encode_tools(
    tools: Sequence[ToolSpec],
    profile: OpenAIChatCompletionsProfile,
) -> list[JsonObject]:
    if not tools:
        return []
    if not profile.supports_tools:
        raise OpenAIChatCompletionsError(f"{profile.name} does not support tools")
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": thaw_json_value(tool.input_schema),
            },
        }
        for tool in tools
    ]


def encode_tool_choice(
    choice: ToolChoice,
    *,
    tool_names: Collection[str],
    profile: OpenAIChatCompletionsProfile,
) -> str | JsonObject | None:
    if not tool_names:
        if choice.type in {"required", "named"}:
            raise OpenAIChatCompletionsError(
                f"tool_choice={choice.type!r} requires at least one tool"
            )
        return None
    if not profile.supports_tool_choice:
        if choice.type == "auto":
            return None
        raise OpenAIChatCompletionsError(f"{profile.name} does not support tool_choice")
    if choice.type in {"auto", "none", "required"}:
        return choice.type
    if choice.name is None or choice.name not in tool_names:
        raise OpenAIChatCompletionsError(f"tool_choice names an unavailable tool: {choice.name}")
    return {"type": "function", "function": {"name": choice.name}}


def encode_assistant_tool_calls(calls: Sequence[ToolCall]) -> list[JsonObject]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(
                    thaw_json_value(call.arguments),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            },
        }
        for call in calls
    ]


def decode_tool_calls(value: object) -> list[ToolCall]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise OpenAIChatCompletionsError("chat completion tool_calls must be an array")
    calls: list[ToolCall] = []
    for item in cast(Sequence[object], value):
        mapping = OPENAI_JSON.mapping(item, "chat completion tool call")
        call_type = mapping.get("type", "function")
        if call_type != "function":
            raise OpenAIChatCompletionsError(
                f"unsupported chat completion tool call type: {call_type}"
            )
        function = OPENAI_JSON.mapping(mapping.get("function"), "chat completion tool function")
        calls.append(
            ToolCall(
                id=OPENAI_JSON.required_string(mapping.get("id"), "chat completion tool call id"),
                name=OPENAI_JSON.required_string(
                    function.get("name"),
                    "chat completion tool function name",
                ),
                arguments=_decode_arguments(function.get("arguments")),
            )
        )
    return calls


def _decode_arguments(value: object) -> Mapping[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if not isinstance(value, str):
        raise OpenAIChatCompletionsError("chat completion tool function arguments must be a string")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OpenAIChatCompletionsError(
            "chat completion tool function arguments are invalid JSON"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise OpenAIChatCompletionsError(
            "chat completion tool function arguments must decode to an object"
        )
    return cast(Mapping[str, Any], parsed)
