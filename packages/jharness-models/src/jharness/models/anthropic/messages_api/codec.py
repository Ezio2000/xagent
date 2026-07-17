"""Request and response codec for Anthropic Messages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from jharness.kernel import ModelRequest, ModelResponse, ModelUsage, ResponseFormat, thaw_json_value
from jharness.models.anthropic.errors import ANTHROPIC_JSON, AnthropicError
from jharness.models.anthropic.messages_api.messages import (
    decode_content_blocks,
    encode_messages,
)
from jharness.models.anthropic.messages_api.tools import (
    decode_tool_uses,
    encode_tool_choice,
    encode_tools,
)
from jharness.models.anthropic.profiles import AnthropicProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]

_RESERVED_REQUEST_FIELDS = {
    "model",
    "max_tokens",
    "messages",
    "system",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
    "tools",
    "tool_choice",
    "output_config",
}
_MAPPING_SUBSCHEMA_KEYWORDS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
)
_SEQUENCE_SUBSCHEMA_KEYWORDS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SINGLE_SUBSCHEMA_KEYWORDS = (
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)


class AnthropicCodec:
    """Translate between kernel model DTOs and Anthropic Messages JSON."""

    def __init__(self, *, model: str, profile: AnthropicProfile | None = None) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.profile = profile or AnthropicProfile()

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        system, messages = encode_messages(request.messages, self.profile)
        payload: JsonObject = {
            "model": request.options.model or self.model,
            "max_tokens": request.options.max_output_tokens or self.profile.default_max_tokens,
            "messages": messages,
        }
        if system is not None:
            payload["system"] = system
        self._add_generation_options(payload, request)
        self._add_tool_options(payload, request)
        self._add_output_options(payload, request)
        self._add_stream_option(payload, stream=stream)
        self._add_extra_request_body(payload)
        return payload

    def _add_generation_options(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.options.temperature is not None:
            payload["temperature"] = request.options.temperature
        if request.options.top_p is not None:
            payload["top_p"] = request.options.top_p
        if request.options.stop:
            payload["stop_sequences"] = list(request.options.stop)
        self._add_seed(payload, request.options.seed)

    def _add_seed(self, payload: JsonObject, seed: int | None) -> None:
        if seed is None:
            return
        seed_field = self.profile.seed_field
        if not seed_field:
            raise AnthropicError(f"{self.profile.name} does not support seed")
        if seed_field in _RESERVED_REQUEST_FIELDS or seed_field in payload:
            raise AnthropicError(f"seed_field conflicts with reserved request field: {seed_field}")
        payload[seed_field] = seed

    def _add_tool_options(
        self,
        payload: JsonObject,
        request: ModelRequest,
    ) -> None:
        tools = encode_tools(request.tools, self.profile)
        if tools:
            payload["tools"] = tools
        tool_choice = encode_tool_choice(
            request.tool_choice,
            tool_names={tool.name for tool in request.tools},
            profile=self.profile,
        )
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    def _add_output_options(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.response_format is not None:
            output_config = self._encode_response_format(request.response_format)
            payload.update(self._merge_output_config(payload, output_config))
        elif self.profile.extra_output_config:
            payload["output_config"] = dict(self.profile.extra_output_config)

    def _add_stream_option(self, payload: JsonObject, *, stream: bool) -> None:
        if stream:
            if not self.profile.supports_streaming:
                raise AnthropicError(f"{self.profile.name} does not support streaming")
            payload["stream"] = True

    def _add_extra_request_body(self, payload: JsonObject) -> None:
        extra_request_body = cast(Mapping[object, Any], self.profile.extra_request_body)
        for raw_key, value in extra_request_body.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise AnthropicError("extra_request_body keys must be non-empty strings")
            key = raw_key
            if key in _RESERVED_REQUEST_FIELDS or key in payload:
                raise AnthropicError(f"extra_request_body cannot set reserved request field: {key}")
            if key == self.profile.seed_field:
                raise AnthropicError(f"extra_request_body cannot set reserved request field: {key}")
            payload[key] = value

    def decode_response(
        self,
        value: Mapping[str, Any],
    ) -> ModelResponse:
        if "error" in value:
            raise AnthropicError("Anthropic response must not contain an error envelope")
        response_type = value.get("type")
        if response_type != "message":
            raise AnthropicError("Anthropic response requires type='message'")
        role = value.get("role")
        if role != "assistant":
            raise AnthropicError("Anthropic response requires role='assistant'")
        if "content" not in value or value["content"] is None:
            raise AnthropicError("Anthropic response requires content")
        parts, tool_blocks = decode_content_blocks(value["content"])
        if not parts and not tool_blocks:
            raise AnthropicError("Anthropic assistant response requires content or tool_use")
        stop_reason = ANTHROPIC_JSON.required_string(
            value.get("stop_reason"), "Anthropic stop_reason"
        )
        usage = decode_usage(value.get("usage"))
        metadata: JsonObject = {"provider": self.profile.name}
        metadata["type"] = response_type
        metadata["role"] = role
        return ModelResponse(
            parts=tuple(parts),
            tool_calls=tuple(decode_tool_uses(tool_blocks)),
            finish_reason=self.profile.finish_reason(stop_reason),
            usage=usage,
            model_id=ANTHROPIC_JSON.optional_string(value.get("model")),
            response_id=ANTHROPIC_JSON.optional_string(value.get("id")),
            metadata=metadata,
        )

    def _encode_response_format(self, response_format: ResponseFormat) -> JsonObject:
        if response_format.type == "text":
            return {}
        if response_format.type == "json_object":
            if not self.profile.supports_json_object:
                raise AnthropicError(f"{self.profile.name} does not support JSON object mode")
            return {
                "format": {
                    "type": "json_schema",
                    "schema": dict(self.profile.json_object_schema),
                }
            }
        if response_format.type == "json_schema":
            if not self.profile.supports_json_schema:
                raise AnthropicError(f"{self.profile.name} does not support JSON schema output")
            if response_format.schema is None:
                raise AnthropicError("JSON schema response format requires schema")
            schema = thaw_json_value(response_format.schema)
            if response_format.strict and isinstance(schema, Mapping):
                schema = _strict_json_schema(schema)
            return {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }
        raise AnthropicError(f"unsupported response format type: {response_format.type}")

    def _merge_output_config(
        self,
        payload: JsonObject,
        output_config: JsonObject,
    ) -> JsonObject:
        if not output_config and not self.profile.extra_output_config:
            return {}
        merged = dict(self.profile.extra_output_config)
        for key, value in output_config.items():
            if key in merged:
                raise AnthropicError(f"extra_output_config cannot set response format field: {key}")
            merged[key] = value
        if "output_config" in payload:
            raise AnthropicError("output_config is already set")
        return {"output_config": merged}


def decode_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    usage = ANTHROPIC_JSON.mapping(value, "Anthropic usage")
    input_tokens = ANTHROPIC_JSON.optional_integer(usage.get("input_tokens"))
    output_tokens = ANTHROPIC_JSON.optional_integer(usage.get("output_tokens"))
    output_details = usage.get("output_tokens_details")
    reasoning_tokens = None
    if isinstance(output_details, Mapping):
        output_details_mapping = cast(Mapping[str, object], output_details)
        reasoning_tokens = ANTHROPIC_JSON.optional_integer(
            output_details_mapping.get("thinking_tokens")
        )
    cache_read_tokens = ANTHROPIC_JSON.optional_integer(usage.get("cache_read_input_tokens"))
    cache_write_tokens = ANTHROPIC_JSON.optional_integer(usage.get("cache_creation_input_tokens"))
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _strict_json_schema(schema: Mapping[str, Any]) -> JsonObject:
    value = deepcopy(dict(schema))
    _apply_strict_json_schema(value)
    return value


def _apply_strict_json_schema(value: object) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if not isinstance(current, Mapping):
            continue
        mapping = cast(dict[str, Any], current)
        _enforce_strict_object_schema(mapping)
        pending.extend(reversed(tuple(_subschemas(mapping))))


def _subschemas(schema: Mapping[str, object]) -> Iterator[object]:
    for keyword in _MAPPING_SUBSCHEMA_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, Mapping):
            yield from cast(Mapping[object, object], value).values()
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, Mapping):
        for child in cast(Mapping[object, object], dependencies).values():
            if isinstance(child, Mapping):
                yield cast(object, child)
    for keyword in _SEQUENCE_SUBSCHEMA_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            yield from cast(Sequence[object], value)
    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            yield cast(object, child)


def _enforce_strict_object_schema(schema: JsonObject) -> None:
    schema_type = schema.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if not is_object and "properties" not in schema:
        return
    additional = schema.get("additionalProperties")
    if additional is not None and additional is not False:
        raise AnthropicError("strict JSON schema object additionalProperties must be false")
    schema["additionalProperties"] = False
