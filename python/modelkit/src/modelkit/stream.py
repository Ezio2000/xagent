"""Helpers for model adapters and adapter tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from kernel import (
    AgentError,
    ContentPart,
    ModelCapabilities,
    ModelContentDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamStarted,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ToolCall,
)


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(value))


@dataclass(slots=True)
class _ContentBuffer:
    part_type: str
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(slots=True)
class _ToolCallBuffer:
    id: str | None = None
    name: str | None = None
    mode: str | None = None
    arguments_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)


class ModelStreamAccumulator:
    """Accumulate provider-neutral stream deltas into a complete ModelResponse."""

    __slots__ = ("_content", "_finish_reason", "_model", "_response_id", "_tool_calls", "_usage")

    _content: dict[int, _ContentBuffer]
    _finish_reason: str | None
    _model: str | None
    _response_id: str | None
    _tool_calls: dict[int, _ToolCallBuffer]
    _usage: ModelUsage | None

    def __init__(self) -> None:
        self._content = {}
        self._tool_calls = {}
        self._usage = None
        self._finish_reason = None
        self._model = None
        self._response_id = None

    def apply(self, event: ModelStreamEvent) -> ModelResponse | None:
        if isinstance(event, ModelStreamStarted | ModelReasoningDelta):
            return None
        if isinstance(event, ModelContentDelta):
            buffer = self._content.setdefault(
                event.index,
                _ContentBuffer(event.part_type, metadata=_copy_mapping(event.metadata)),
            )
            if buffer.part_type != event.part_type:
                raise AgentError("stream content part_type changed for the same index")
            buffer.text += event.text_delta
            return None
        if isinstance(event, ModelToolCallDelta):
            buffer = self._tool_calls.setdefault(
                event.index, _ToolCallBuffer(metadata=_copy_mapping(event.metadata))
            )
            if event.id is not None:
                if buffer.id is not None and buffer.id != event.id:
                    raise AgentError("stream tool call id changed for the same index")
                buffer.id = event.id
            if event.name is not None:
                if buffer.name is not None and buffer.name != event.name:
                    raise AgentError("stream tool call name changed for the same index")
                buffer.name = event.name
            if event.mode is not None:
                if buffer.mode is not None and buffer.mode != event.mode:
                    raise AgentError("stream tool call mode changed for the same index")
                buffer.mode = event.mode
            if event.arguments_delta is not None:
                buffer.arguments_text += event.arguments_delta
            if event.metadata:
                buffer.metadata = _copy_mapping(event.metadata)
            return None
        if isinstance(event, ModelUsageDelta):
            self._usage = ModelUsage.from_dict(event.usage.to_dict())
            return None
        response = ModelResponse.from_dict(event.response.to_dict())
        self._finish_reason = response.finish_reason
        self._usage = response.usage
        self._model = response.model
        self._response_id = response.response_id
        return response

    def response(self) -> ModelResponse:
        parts: list[ContentPart] = []
        for index in sorted(self._content):
            buffer = self._content[index]
            if buffer.part_type == "text":
                parts.append(ContentPart.text_part(buffer.text, metadata=buffer.metadata))
            else:
                parts.append(
                    ContentPart(
                        type=buffer.part_type,
                        text=buffer.text,
                        metadata=buffer.metadata,
                    )
                )

        tool_calls: list[ToolCall] = []
        for index in sorted(self._tool_calls):
            buffer = self._tool_calls[index]
            if not buffer.id or not buffer.name:
                raise AgentError("stream tool call requires id and name")
            raw_arguments = buffer.arguments_text or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise AgentError("stream tool call arguments are not valid JSON") from exc
            if not isinstance(arguments, Mapping):
                raise AgentError("stream tool call arguments must decode to an object")
            tool_calls.append(
                ToolCall(
                    id=buffer.id,
                    name=buffer.name,
                    mode=buffer.mode or "execute",
                    arguments=cast(Mapping[str, Any], arguments),
                    metadata=buffer.metadata,
                )
            )

        return ModelResponse(
            parts=parts,
            tool_calls=tool_calls,
            finish_reason=self._finish_reason,
            usage=self._usage,
            model=self._model,
            response_id=self._response_id,
        )


def model_capabilities(client: object) -> ModelCapabilities:
    """Return capabilities advertised by a model client, or the empty default."""

    value = getattr(client, "capabilities", None)
    if value is None:
        return ModelCapabilities()
    if isinstance(value, ModelCapabilities):
        return value
    if isinstance(value, Mapping):
        return ModelCapabilities.from_dict(cast(Mapping[str, Any], value))
    if callable(value):
        result = value()
        if isinstance(result, ModelCapabilities):
            return result
        if isinstance(result, Mapping):
            return ModelCapabilities.from_dict(cast(Mapping[str, Any], result))
    raise TypeError("model capabilities must be ModelCapabilities, mapping, or callable")
