"""Request and response codec for OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ResponseFormat,
    ToolCall,
    thaw_json_value,
)
from jharness.models.openai.chat_completions.messages import (
    decode_message_content,
    decode_message_refusal,
    encode_chat_message,
)
from jharness.models.openai.chat_completions.tools import (
    decode_tool_calls,
    encode_tool_choice,
    encode_tools,
)
from jharness.models.openai.errors import OPENAI_JSON, OpenAIChatCompletionsError
from jharness.models.openai.profiles import OpenAIChatCompletionsProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]

_RESERVED_REQUEST_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "stream",
    "stream_options",
    "n",
}


class OpenAIChatCompletionsCodec:
    """Translate between kernel model DTOs and Chat Completions JSON."""

    def __init__(
        self,
        *,
        model: str,
        profile: OpenAIChatCompletionsProfile | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.profile = profile or OpenAIChatCompletionsProfile()

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        tools = encode_tools(request.tools, self.profile)
        payload: JsonObject = {
            "model": request.options.model or self.model,
            "messages": [
                encode_chat_message(message, self.profile) for message in request.messages
            ],
        }
        self._add_model_options(payload, request)
        self._add_tool_options(payload, request, tools)
        self._add_response_format(payload, request)
        self._add_stream_options(payload, stream=stream)
        self._add_extra_request_body(payload)
        return payload

    def _add_model_options(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.options.temperature is not None:
            payload["temperature"] = request.options.temperature
        if request.options.top_p is not None:
            payload["top_p"] = request.options.top_p
        if request.options.max_output_tokens is not None:
            payload[self.profile.max_tokens_field] = request.options.max_output_tokens
        if request.options.stop:
            payload["stop"] = list(request.options.stop)
        if request.options.seed is not None:
            if not self.profile.supports_seed:
                raise OpenAIChatCompletionsError(f"{self.profile.name} does not support seed")
            payload["seed"] = request.options.seed

    def _add_tool_options(
        self,
        payload: JsonObject,
        request: ModelRequest,
        tools: list[JsonObject],
    ) -> None:
        if tools:
            payload["tools"] = tools
        tool_choice = encode_tool_choice(
            request.tool_choice,
            tool_names={tool.name for tool in request.tools},
            profile=self.profile,
        )
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if (
            tools
            and request.tool_choice.type != "none"
            and self.profile.supports_parallel_tool_call_control
        ):
            payload["parallel_tool_calls"] = request.tool_choice.allow_parallel_tool_calls

    def _add_response_format(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.response_format is not None:
            payload["response_format"] = self._encode_response_format(request.response_format)

    def _add_stream_options(self, payload: JsonObject, *, stream: bool) -> None:
        if stream:
            if not self.profile.supports_streaming:
                raise OpenAIChatCompletionsError(f"{self.profile.name} does not support streaming")
            payload["stream"] = True
            if self.profile.stream_include_usage:
                payload["stream_options"] = {"include_usage": True}

    def _add_extra_request_body(self, payload: JsonObject) -> None:
        for key, value in self.profile.extra_request_body.items():
            if key in _RESERVED_REQUEST_FIELDS:
                raise OpenAIChatCompletionsError(
                    f"extra_request_body cannot set reserved request field: {key}"
                )
            payload[key] = value

    def decode_response(
        self,
        value: Mapping[str, Any],
    ) -> ModelResponse:
        choice, message = _decode_assistant_choice(value)
        parts, tool_calls = _decode_assistant_payload(message, self.profile)
        finish_reason = OPENAI_JSON.required_string(
            choice.get("finish_reason"),
            "chat completion finish_reason",
        )
        return ModelResponse(
            parts=tuple(parts),
            tool_calls=tuple(tool_calls),
            finish_reason=self.profile.finish_reason(finish_reason),
            usage=decode_usage(value.get("usage")),
            model_id=OPENAI_JSON.optional_string(value.get("model")),
            response_id=OPENAI_JSON.optional_string(value.get("id")),
            metadata=_response_metadata(value, self.profile.name),
        )

    def _encode_response_format(self, response_format: ResponseFormat) -> JsonObject:
        if response_format.type == "text":
            return {"type": "text"}
        if response_format.type == "json_object":
            if not self.profile.supports_json_object:
                raise OpenAIChatCompletionsError(
                    f"{self.profile.name} does not support JSON object mode"
                )
            return {"type": "json_object"}
        if response_format.type == "json_schema":
            if not self.profile.supports_json_schema:
                raise OpenAIChatCompletionsError(
                    f"{self.profile.name} does not support JSON schema output"
                )
            if response_format.schema is None:
                raise OpenAIChatCompletionsError("JSON schema response format requires schema")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.profile.json_schema_name,
                    "schema": thaw_json_value(response_format.schema),
                    "strict": response_format.strict,
                },
            }
        raise OpenAIChatCompletionsError(
            f"unsupported response format type: {response_format.type}"
        )


def _decode_assistant_choice(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if "error" in value:
        raise OpenAIChatCompletionsError(
            "chat completion response must not contain an error envelope"
        )
    raw_choices = value.get("choices")
    if not isinstance(raw_choices, list):
        raise OpenAIChatCompletionsError("chat completion response requires exactly one choice")
    choices = cast(list[object], raw_choices)
    if len(choices) != 1:
        raise OpenAIChatCompletionsError("chat completion response requires exactly one choice")
    choice = OPENAI_JSON.mapping(choices[0], "chat completion choice")
    if _choice_index(choice) != 0:
        raise OpenAIChatCompletionsError("chat completion response choice index must be 0")
    message = OPENAI_JSON.mapping(choice.get("message"), "chat completion choice message")
    if message.get("role") != "assistant":
        raise OpenAIChatCompletionsError("chat completion response requires role='assistant'")
    return choice, message


def _decode_assistant_payload(
    message: Mapping[str, Any],
    profile: OpenAIChatCompletionsProfile,
) -> tuple[list[ContentPart], list[ToolCall]]:
    reasoning_parts = _decode_message_reasoning(
        message.get("reasoning_content"),
        profile,
    )
    parts = decode_message_content(message.get("content"))
    refusal_parts = decode_message_refusal(message.get("refusal"))
    if refusal_parts and any(part.type == "refusal" for part in parts):
        raise OpenAIChatCompletionsError(
            "chat completion response must not duplicate refusal content"
        )
    parts.extend(refusal_parts)
    tool_calls = decode_tool_calls(message.get("tool_calls"))
    if (
        tool_calls
        and profile.reasoning_content_mode == "required_with_tools"
        and not reasoning_parts
    ):
        raise OpenAIChatCompletionsError(
            f"{profile.name} requires non-empty reasoning content for assistant tool calls"
        )
    parts[:0] = reasoning_parts
    if not parts and not tool_calls:
        raise OpenAIChatCompletionsError(
            "chat completion assistant message requires content, refusal, or tool_calls"
        )
    return parts, tool_calls


def _decode_message_reasoning(
    value: object,
    profile: OpenAIChatCompletionsProfile,
) -> list[ContentPart]:
    if profile.reasoning_content_mode == "live_only" or value is None:
        return []
    if not isinstance(value, str):
        raise OpenAIChatCompletionsError(
            "chat completion message reasoning_content must be a string or null"
        )
    return [ContentPart(type="reasoning", text=value)] if value else []


def _response_metadata(value: Mapping[str, Any], provider: str) -> JsonObject:
    metadata: JsonObject = {"provider": provider, "choice_count": 1}
    object_value = value.get("object")
    if object_value is not None:
        metadata["object"] = OPENAI_JSON.required_string(object_value, "chat completion object")
    created = value.get("created")
    if created is not None:
        if not isinstance(created, int) or isinstance(created, bool):
            raise OpenAIChatCompletionsError("chat completion created must be an integer")
        metadata["created"] = created
    return metadata


def decode_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    usage = OPENAI_JSON.mapping(value, "chat completion usage")
    prompt_tokens = OPENAI_JSON.optional_integer(usage.get("prompt_tokens"))
    completion_tokens = OPENAI_JSON.optional_integer(usage.get("completion_tokens"))
    total_tokens = OPENAI_JSON.optional_integer(usage.get("total_tokens"))
    completion_details = usage.get("completion_tokens_details")
    prompt_details = usage.get("prompt_tokens_details")
    reasoning_tokens = None
    cache_read_tokens = None
    if isinstance(completion_details, Mapping):
        completion_details_mapping = cast(Mapping[str, object], completion_details)
        reasoning_tokens = OPENAI_JSON.optional_integer(
            completion_details_mapping.get("reasoning_tokens")
        )
    if isinstance(prompt_details, Mapping):
        prompt_details_mapping = cast(Mapping[str, object], prompt_details)
        cache_read_tokens = OPENAI_JSON.optional_integer(
            prompt_details_mapping.get("cached_tokens")
        )
    if cache_read_tokens is None:
        cache_read_tokens = OPENAI_JSON.optional_integer(usage.get("prompt_cache_hit_tokens"))
    return ModelUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def _choice_index(choice: Mapping[str, Any]) -> int:
    if "index" not in choice:
        raise OpenAIChatCompletionsError("chat completion response choice requires an index")
    index = choice["index"]
    if not isinstance(index, int) or isinstance(index, bool):
        raise OpenAIChatCompletionsError("chat completion response choice index must be an integer")
    return index
