"""Streaming conversion for Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from jharness.kernel import (
    ContentPart,
    ModelContentDelta,
    ModelDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
)
from jharness.models._stream import DeltaAccumulator
from jharness.models.anthropic.errors import ANTHROPIC_JSON, AnthropicError
from jharness.models.anthropic.messages_api.codec import decode_usage
from jharness.models.anthropic.profiles import AnthropicProfile

_BLOCK_TYPE_BY_DELTA_TYPE = {
    "input_json_delta": "tool_use",
    "signature_delta": "thinking",
    "text_delta": "text",
    "thinking_delta": "thinking",
}


@dataclass(slots=True)
class _BlockState:
    block_type: str
    populated: bool
    tool_index: int | None = None
    closed: bool = False


class AnthropicStreamDecoder:
    """Decode Anthropic Messages SSE events into kernel stream events."""

    def __init__(
        self,
        profile: AnthropicProfile,
    ) -> None:
        self._profile = profile
        self._accumulator = DeltaAccumulator(AnthropicError)
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._response_id: str | None = None
        self._usage: ModelUsage | None = None
        self._blocks: dict[int, _BlockState] = {}
        self._next_tool_index = 0
        self._phase: Literal["initial", "active", "delta_seen", "stopped"] = "initial"

    def apply_event(
        self,
        event_name: str | None,
        value: Mapping[str, Any],
    ) -> tuple[bool, list[ModelDelta]]:
        if self._phase == "stopped":
            raise AnthropicError("Anthropic stream emitted an event after message_stop")
        event_type = _event_type(event_name, value)
        if event_type == "ping":
            return False, []
        if event_type == "message_start":
            deltas = self._message_start_events(value)
        elif event_type == "content_block_start":
            deltas = self._content_block_start_events(value)
        elif event_type == "content_block_delta":
            deltas = self._content_block_delta_events(value)
        elif event_type == "content_block_stop":
            self._content_block_stop(value)
            return False, []
        elif event_type == "message_delta":
            deltas = self._message_delta_events(value)
        elif event_type == "message_stop":
            self._message_stop()
            return True, []
        elif event_type == "error":
            raise AnthropicError("Anthropic stream error event")
        else:
            raise AnthropicError(f"unsupported Anthropic stream event type: {event_type}")
        self._accumulate(deltas)
        return False, deltas

    def completed_response(self) -> ModelResponse:
        if self._phase != "stopped":
            raise AnthropicError("Anthropic stream completed before message_stop")
        response = self._accumulator.response(
            finish_reason=self._finish_reason,
            model_id=self._model,
            response_id=self._response_id,
            metadata={
                "provider": self._profile.name,
                "type": "message",
                "role": "assistant",
            },
        )
        return replace(
            response,
            parts=tuple(_complete_native_thinking(part) for part in response.parts),
        )

    def _message_start_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        if self._phase != "initial":
            raise AnthropicError("Anthropic stream message_start appeared more than once")
        self._phase = "active"
        message = ANTHROPIC_JSON.mapping(value.get("message"), "Anthropic stream message")
        if message.get("type") != "message":
            raise AnthropicError("Anthropic stream message_start requires type='message'")
        if message.get("role") != "assistant":
            raise AnthropicError("Anthropic stream message_start requires role='assistant'")
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
            raise AnthropicError("Anthropic stream message_start content must be an array")
        if content:
            raise AnthropicError("Anthropic stream message_start content must be empty")
        message_id = message.get("id")
        model = message.get("model")
        if message_id is not None:
            self._response_id = ANTHROPIC_JSON.required_string(
                message_id, "Anthropic stream message id"
            )
        if model is not None:
            self._model = ANTHROPIC_JSON.required_string(model, "Anthropic stream model")
        return self._usage_events(message.get("usage"))

    def _content_block_start_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_message_started("content_block_start")
        index = _event_index(value)
        if index in self._blocks:
            raise AnthropicError(f"Anthropic content block index started more than once: {index}")
        block = ANTHROPIC_JSON.mapping(value.get("content_block"), "Anthropic content block")
        block_type = _required_type(
            block.get("type"),
            "Anthropic content block requires non-empty type",
        )
        tool_index: int | None = None
        if block_type in {"text", "thinking"}:
            field = "text" if block_type == "text" else "thinking"
            suffix = "text" if block_type == "text" else "thinking text"
            events = _text_events(
                index,
                block.get(field),
                part_type=block_type,
                error_message=f"Anthropic {block_type} block requires {suffix}",
            )
        elif block_type == "redacted_thinking":
            events = self._redacted_thinking_start_events(index, block)
        elif block_type == "tool_use":
            tool_index = self._next_tool_index
            events = self._tool_use_start_events(tool_index, block)
            self._next_tool_index += 1
        else:
            raise AnthropicError(f"unsupported Anthropic stream content block: {block_type}")
        self._blocks[index] = _BlockState(block_type, bool(events), tool_index)
        return events

    def _redacted_thinking_start_events(
        self,
        index: int,
        block: Mapping[str, Any],
    ) -> list[ModelDelta]:
        data = block.get("data")
        if not isinstance(data, str) or not data:
            raise AnthropicError("Anthropic redacted_thinking block requires non-empty data")
        return [
            ModelContentDelta(
                index=index,
                text_delta="",
                part_type="redacted_thinking",
                data={"anthropic": {"type": "redacted_thinking", "data": data}},
            )
        ]

    def _tool_use_start_events(
        self,
        call_index: int,
        block: Mapping[str, Any],
    ) -> list[ModelDelta]:
        call_id = ANTHROPIC_JSON.required_string(block.get("id"), "Anthropic tool_use id")
        name = ANTHROPIC_JSON.required_string(block.get("name"), "Anthropic tool_use name")
        raw_input = block.get("input", {})
        if not isinstance(raw_input, Mapping):
            raise AnthropicError("Anthropic tool_use input must be an object")
        deltas: list[ModelDelta] = [
            ModelToolCallDelta(index=call_index, arguments_delta="", id=call_id, name=name)
        ]
        if raw_input:
            deltas.append(
                ModelToolCallDelta(
                    index=call_index,
                    arguments_delta=json.dumps(
                        cast(Mapping[str, Any], raw_input), separators=(",", ":"), sort_keys=True
                    ),
                )
            )
        return deltas

    def _content_block_delta_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_message_started("content_block_delta")
        index = _event_index(value)
        state = self._blocks.get(index)
        if state is None or state.closed:
            raise AnthropicError(f"Anthropic content block delta requires an open index: {index}")
        delta = ANTHROPIC_JSON.mapping(value.get("delta"), "Anthropic content block delta")
        delta_type = _required_type(
            delta.get("type"),
            "Anthropic content block delta requires non-empty type",
        )
        expected_block_type = _BLOCK_TYPE_BY_DELTA_TYPE.get(delta_type)
        if expected_block_type is None:
            raise AnthropicError(f"unsupported Anthropic content block delta type: {delta_type}")
        if state.block_type != expected_block_type:
            raise AnthropicError(
                f"Anthropic {delta_type} does not match {state.block_type} content block"
            )
        if delta_type in {"text_delta", "thinking_delta"}:
            part_type = "text" if delta_type == "text_delta" else "thinking"
            field = "text" if part_type == "text" else "thinking"
            suffix = "text" if part_type == "text" else "thinking text"
            events = _text_events(
                index,
                delta.get(field),
                part_type=part_type,
                error_message=f"Anthropic {part_type} delta requires {suffix}",
            )
        elif delta_type == "signature_delta":
            events = self._signature_delta_events(index, delta)
        else:
            if state.tool_index is None:
                raise AnthropicError("Anthropic tool input delta requires a tool call index")
            events = self._input_json_delta_events(state.tool_index, delta)
        if events and delta_type != "signature_delta":
            state.populated = True
        return events

    def _signature_delta_events(
        self,
        index: int,
        delta: Mapping[str, Any],
    ) -> list[ModelDelta]:
        signature = delta.get("signature")
        if not isinstance(signature, str):
            raise AnthropicError("Anthropic signature delta requires signature")
        if not signature:
            return []
        return [
            ModelContentDelta(
                index=index,
                text_delta="",
                part_type="thinking",
                data={"anthropic": {"type": "thinking", "signature": signature}},
            )
        ]

    def _input_json_delta_events(
        self,
        call_index: int,
        delta: Mapping[str, Any],
    ) -> list[ModelDelta]:
        partial = delta.get("partial_json")
        if not isinstance(partial, str):
            raise AnthropicError("Anthropic input JSON delta requires partial_json")
        return [ModelToolCallDelta(index=call_index, arguments_delta=partial)] if partial else []

    def _accumulate(self, deltas: Sequence[ModelDelta]) -> None:
        for delta in deltas:
            self._accumulator.apply(delta)

    def _message_delta_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_message_started("message_delta")
        if self._phase == "delta_seen":
            raise AnthropicError("Anthropic stream message_delta appeared more than once")
        if any(not state.closed for state in self._blocks.values()):
            raise AnthropicError("Anthropic message_delta requires all content blocks to stop")
        delta = ANTHROPIC_JSON.mapping(value.get("delta"), "Anthropic message delta")
        stop_reason = ANTHROPIC_JSON.required_string(
            delta.get("stop_reason"),
            "Anthropic message_delta stop_reason",
        )
        self._finish_reason = self._profile.finish_reason(stop_reason)
        self._phase = "delta_seen"
        return self._usage_events(value.get("usage"))

    def _content_block_stop(self, value: Mapping[str, Any]) -> None:
        self._require_message_started("content_block_stop")
        index = _event_index(value)
        state = self._blocks.get(index)
        if state is None or state.closed:
            raise AnthropicError(f"Anthropic content block stop requires an open index: {index}")
        self._validate_populated_content_block(state)
        state.closed = True

    @staticmethod
    def _validate_populated_content_block(state: _BlockState) -> None:
        if state.populated:
            return
        block_type = state.block_type
        if block_type == "text":
            raise AnthropicError("Anthropic text content block requires non-empty text before stop")
        if block_type == "thinking":
            raise AnthropicError(
                "Anthropic thinking content block requires non-empty thinking before stop"
            )
        raise AnthropicError(f"Anthropic {block_type} content block completed without data")

    def _message_stop(self) -> None:
        self._require_message_started("message_stop")
        open_indexes = sorted(index for index, state in self._blocks.items() if not state.closed)
        if open_indexes:
            indexes = ", ".join(str(index) for index in open_indexes)
            raise AnthropicError(
                f"Anthropic message_stop has open content block indexes: {indexes}"
            )
        if self._phase != "delta_seen" or self._finish_reason is None:
            raise AnthropicError("Anthropic message_stop requires a terminal message_delta")
        if not self._accumulator.has_output:
            raise AnthropicError(
                "Anthropic stream completed without content, thinking, or tool_use"
            )
        self._phase = "stopped"

    def _require_message_started(self, event_type: str) -> None:
        if self._phase == "initial":
            raise AnthropicError(f"Anthropic {event_type} requires message_start")

    def _usage_events(self, value: object) -> list[ModelDelta]:
        if not self._profile.stream_usage:
            return []
        usage = decode_usage(value)
        if usage is None:
            return []
        self._usage = _merge_usage(self._usage, usage)
        return [ModelUsageDelta(usage=self._usage)]


def _text_events(
    index: int,
    value: object,
    *,
    part_type: str,
    error_message: str,
) -> list[ModelDelta]:
    if not isinstance(value, str):
        raise AnthropicError(error_message)
    if not value:
        return []
    content = ModelContentDelta(index=index, text_delta=value, part_type=part_type)
    if part_type == "text":
        return [content]
    return [ModelReasoningDelta(index=index, text_delta=value), content]


def _merge_usage(existing: ModelUsage | None, update: ModelUsage) -> ModelUsage:
    if existing is None:
        return update
    input_tokens = _prefer(update.input_tokens, existing.input_tokens)
    output_tokens = _prefer(update.output_tokens, existing.output_tokens)
    total_tokens = _prefer(update.total_tokens, existing.total_tokens)
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=_prefer(update.reasoning_tokens, existing.reasoning_tokens),
        cache_read_tokens=_prefer(update.cache_read_tokens, existing.cache_read_tokens),
        cache_write_tokens=_prefer(update.cache_write_tokens, existing.cache_write_tokens),
    )


def _complete_native_thinking(part: ContentPart) -> ContentPart:
    if part.type != "thinking":
        return part
    raw_block = part.data.get("anthropic")
    block = dict(cast(Mapping[str, Any], raw_block)) if isinstance(raw_block, Mapping) else {}
    block.update(type="thinking", thinking=part.text or "")
    data = dict(part.data)
    data["anthropic"] = block
    return replace(part, data=data)


def _prefer(updated: int | None, existing: int | None) -> int | None:
    return updated if updated is not None else existing


def _event_type(event_name: str | None, value: Mapping[str, Any]) -> str:
    raw_type = value.get("type")
    if not isinstance(raw_type, str) or not raw_type:
        raise AnthropicError("Anthropic stream event payload requires a type")
    if event_name is not None and event_name != raw_type:
        raise AnthropicError("Anthropic stream event name must match the payload type")
    return raw_type


def _event_index(value: Mapping[str, Any]) -> int:
    if "index" not in value:
        raise AnthropicError("Anthropic stream content block event requires an index")
    index = value["index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise AnthropicError("Anthropic stream event index must be an integer")
    if index < 0:
        raise AnthropicError("Anthropic stream event index must be >= 0")
    return index


def _required_type(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnthropicError(error_message)
    return value
