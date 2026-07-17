"""Streaming conversion for OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from jharness.kernel import (
    ModelContentDelta,
    ModelDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
)
from jharness.models._stream import DeltaAccumulator
from jharness.models.openai.chat_completions.codec import decode_usage
from jharness.models.openai.errors import OPENAI_JSON, OpenAIChatCompletionsError
from jharness.models.openai.profiles import OpenAIChatCompletionsProfile

_MetadataValue = TypeVar("_MetadataValue")


class OpenAIChatStreamDecoder:
    """Decode Chat Completions stream chunks into kernel stream events."""

    def __init__(
        self,
        profile: OpenAIChatCompletionsProfile,
    ) -> None:
        self._profile = profile
        self._accumulator = DeltaAccumulator(OpenAIChatCompletionsError)
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._response_id: str | None = None
        self._usage: ModelUsage | None = None
        self._object: str | None = None
        self._created: int | None = None
        self._phase: Literal["initial", "active", "finished"] = "initial"

    def apply_chunk(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._capture_chunk_metadata(value)
        usage_event = self._capture_usage(value.get("usage"))
        deltas: list[ModelDelta] = [] if usage_event is None else [usage_event]
        choice = self._decode_choice(value.get("choices"), has_usage=usage_event is not None)
        if choice is not None:
            deltas.extend(self._apply_choice(choice))
        for delta in deltas:
            self._accumulator.apply(delta)
        return deltas

    def _capture_usage(self, value: object) -> ModelUsageDelta | None:
        usage = decode_usage(value)
        if usage is None:
            return None
        self._usage = usage if self._usage is None else self._usage.merge_snapshot(usage)
        return ModelUsageDelta(usage=self._usage)

    @staticmethod
    def _decode_choice(value: object, *, has_usage: bool) -> Mapping[str, Any] | None:
        if value is None:
            if not has_usage:
                raise OpenAIChatCompletionsError(
                    "chat completion stream chunk requires choices or usage"
                )
            return None
        if not isinstance(value, list):
            raise OpenAIChatCompletionsError("chat completion stream choices must be an array")
        raw_choices = cast(list[object], value)
        if not raw_choices:
            if not has_usage:
                raise OpenAIChatCompletionsError(
                    "chat completion stream empty choices require usage"
                )
            return None
        if len(raw_choices) != 1:
            raise OpenAIChatCompletionsError(
                "chat completion stream requires exactly one choice per chunk"
            )
        return OPENAI_JSON.mapping(raw_choices[0], "chat completion stream choice")

    def _apply_choice(self, choice: Mapping[str, Any]) -> list[ModelDelta]:
        if self._phase == "finished":
            raise OpenAIChatCompletionsError(
                "chat completion stream emitted a choice after finish_reason"
            )
        if _choice_index(choice) != 0:
            raise OpenAIChatCompletionsError("chat completion stream choice index must be 0")
        self._phase = "active"
        delta = OPENAI_JSON.mapping(choice.get("delta"), "chat completion stream delta")
        deltas = self._deltas_from_wire(delta)
        self._capture_finish_reason(choice.get("finish_reason"))
        return deltas

    def _capture_finish_reason(self, finish_reason: object) -> None:
        if finish_reason is None:
            return
        if not isinstance(finish_reason, str) or not finish_reason:
            raise OpenAIChatCompletionsError(
                "chat completion stream finish_reason must be a non-empty string or null"
            )
        self._finish_reason = self._profile.finish_reason(finish_reason)
        self._phase = "finished"

    def completed_response(self) -> ModelResponse:
        if self._phase == "initial":
            raise OpenAIChatCompletionsError("chat completion stream completed without a choice")
        if not self._accumulator.has_output:
            raise OpenAIChatCompletionsError(
                "chat completion stream completed without content, refusal, or tool_calls"
            )
        if self._phase != "finished":
            raise OpenAIChatCompletionsError(
                "chat completion stream completed before finish_reason"
            )
        metadata: dict[str, Any] = {
            "provider": self._profile.name,
            "choice_count": 1,
        }
        if self._object is not None:
            metadata["object"] = self._object
        if self._created is not None:
            metadata["created"] = self._created
        return self._accumulator.response(
            finish_reason=self._finish_reason,
            model_id=self._model,
            response_id=self._response_id,
            metadata=metadata,
        )

    def _capture_chunk_metadata(self, value: Mapping[str, Any]) -> None:
        self._response_id = _consistent_metadata_value(
            self._response_id,
            _optional_metadata_str(value.get("id"), "id"),
            "id",
        )
        self._model = _consistent_metadata_value(
            self._model,
            _optional_metadata_str(value.get("model"), "model"),
            "model",
        )
        self._object = _consistent_metadata_value(
            self._object,
            _optional_metadata_str(value.get("object"), "object"),
            "object",
        )
        self._created = _consistent_metadata_value(
            self._created,
            _optional_metadata_int(value.get("created"), "created"),
            "created",
        )

    def _deltas_from_wire(self, delta: Mapping[str, Any]) -> list[ModelDelta]:
        _validate_delta_role(delta.get("role"))
        deltas: list[ModelDelta] = [
            item
            for item in (
                self._content_event(delta.get("content")),
                _reasoning_event(delta.get("reasoning_content")),
                self._refusal_event(delta.get("refusal")),
            )
            if item is not None
        ]
        deltas.extend(self._tool_call_events(delta.get("tool_calls")))
        return deltas

    def _content_event(self, value: object) -> ModelContentDelta | None:
        content = _optional_delta_text(value, "content")
        if not content:
            return None
        return ModelContentDelta(index=0, text_delta=content)

    def _refusal_event(self, value: object) -> ModelContentDelta | None:
        refusal = _optional_delta_text(value, "refusal")
        if not refusal:
            return None
        return ModelContentDelta(index=0, text_delta=refusal, part_type="refusal")

    def _tool_call_events(self, value: object) -> list[ModelToolCallDelta]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise OpenAIChatCompletionsError("chat completion stream tool_calls must be an array")
        decoded = (self._tool_call_event(raw_call) for raw_call in cast(list[object], value))
        return [event for event in decoded if event is not None]

    def _tool_call_event(self, value: object) -> ModelToolCallDelta | None:
        call = OPENAI_JSON.mapping(value, "chat completion stream tool call")
        call_type = call.get("type")
        if call_type is not None and call_type != "function":
            raise OpenAIChatCompletionsError(
                f"unsupported chat completion stream tool call type: {call_type}"
            )
        call_index = _tool_call_index(call)
        function = call.get("function")
        function_mapping = (
            OPENAI_JSON.mapping(function, "chat completion stream tool function")
            if function is not None
            else cast(Mapping[str, Any], {})
        )
        call_id = OPENAI_JSON.optional_string(call.get("id"))
        name = OPENAI_JSON.optional_string(function_mapping.get("name"))
        arguments_delta = OPENAI_JSON.optional_string(function_mapping.get("arguments"))
        if call_id is None and name is None and arguments_delta is None:
            return None
        return ModelToolCallDelta(
            index=call_index,
            id=call_id,
            name=name,
            arguments_delta=arguments_delta or "",
        )


def _validate_delta_role(role: object) -> None:
    if role is not None and role != "assistant":
        raise OpenAIChatCompletionsError("chat completion stream delta role must be 'assistant'")


def _reasoning_event(value: object) -> ModelReasoningDelta | None:
    reasoning = _optional_delta_text(value, "reasoning")
    if not reasoning:
        return None
    return ModelReasoningDelta(index=0, text_delta=reasoning)


def _optional_delta_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIChatCompletionsError(
            f"chat completion stream {label} delta must be a string or null"
        )
    return value


def _choice_index(choice: Mapping[str, Any]) -> int:
    if "index" not in choice:
        raise OpenAIChatCompletionsError("chat completion stream choice requires an index")
    index = choice["index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise OpenAIChatCompletionsError("chat completion stream choice index must be an integer")
    return index


def _tool_call_index(call: Mapping[str, Any]) -> int:
    index = call.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        raise OpenAIChatCompletionsError(
            "chat completion stream tool call index must be an integer"
        )
    if index < 0:
        raise OpenAIChatCompletionsError("chat completion stream tool call index must be >= 0")
    return index


def _optional_metadata_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIChatCompletionsError(f"chat completion stream {label} must be a string or null")
    if not value:
        raise OpenAIChatCompletionsError(f"chat completion stream {label} must not be empty")
    return value


def _optional_metadata_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAIChatCompletionsError(
            f"chat completion stream {label} must be an integer or null"
        )
    return value


def _consistent_metadata_value(
    existing: _MetadataValue | None,
    update: _MetadataValue | None,
    label: str,
) -> _MetadataValue | None:
    if update is None:
        return existing
    if existing is not None and update != existing:
        raise OpenAIChatCompletionsError(f"chat completion stream {label} changed between chunks")
    return update
