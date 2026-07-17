"""Private provider-owned accumulation for portable model deltas."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, NoReturn, cast

from jharness.kernel import (
    ContentPart,
    ModelContentDelta,
    ModelDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ToolCall,
)


@dataclass(slots=True)
class _ContentBuffer:
    part_type: str
    text_chunks: list[str] = field(default_factory=list[str])
    data: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(slots=True)
class _ToolCallBuffer:
    id: str | None = None
    name: str | None = None
    argument_chunks: list[str] = field(default_factory=list[str])


class DeltaAccumulator:
    """Build one final response inside a provider adapter."""

    __slots__ = ("_content", "_error_factory", "_tools", "_usage")

    def __init__(self, error_factory: Callable[[str], Exception]) -> None:
        self._error_factory = error_factory
        self._content: dict[int, _ContentBuffer] = {}
        self._tools: dict[int, _ToolCallBuffer] = {}
        self._usage: ModelUsage | None = None

    def apply(self, delta: ModelDelta) -> None:
        if isinstance(delta, ModelContentDelta):
            self._apply_content(delta)
        elif isinstance(delta, ModelToolCallDelta):
            self._apply_tool_call(delta)
        elif isinstance(delta, ModelUsageDelta):
            self._usage = (
                delta.usage if self._usage is None else self._usage.merge_snapshot(delta.usage)
            )
        elif not isinstance(cast(object, delta), ModelReasoningDelta):
            self._raise("provider stream produced an unsupported model delta")

    @property
    def has_output(self) -> bool:
        """Whether accumulated deltas can produce response content or tool calls."""

        return bool(self._content or self._tools)

    def response(
        self,
        *,
        finish_reason: str | None,
        model_id: str | None,
        response_id: str | None,
        metadata: Mapping[str, Any],
    ) -> ModelResponse:
        try:
            return ModelResponse(
                parts=tuple(
                    ContentPart(
                        type=buffer.part_type,
                        text="".join(buffer.text_chunks),
                        data=buffer.data,
                    )
                    for _, buffer in sorted(self._content.items())
                ),
                tool_calls=tuple(
                    self._build_tool_call(buffer) for _, buffer in sorted(self._tools.items())
                ),
                finish_reason=finish_reason,
                usage=self._usage,
                model_id=model_id,
                response_id=response_id,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            self._raise(f"provider stream produced an invalid response: {exc}", cause=exc)

    def _apply_content(self, delta: ModelContentDelta) -> None:
        current = self._content.setdefault(delta.index, _ContentBuffer(delta.part_type))
        if current.part_type != delta.part_type:
            self._raise("content delta part_type changed for one index")
        current.text_chunks.append(delta.text_delta)
        current.data.update(delta.data)

    def _apply_tool_call(self, delta: ModelToolCallDelta) -> None:
        current = self._tools.setdefault(delta.index, _ToolCallBuffer())
        current.id = self._consistent_value(current.id, delta.id, "id")
        current.name = self._consistent_value(current.name, delta.name, "name")
        current.argument_chunks.append(delta.arguments_delta)

    def _build_tool_call(self, buffer: _ToolCallBuffer) -> ToolCall:
        if buffer.id is None or buffer.name is None:
            self._raise("streamed tool call requires id and name")
        try:
            arguments: object = json.loads("".join(buffer.argument_chunks) or "{}")
        except json.JSONDecodeError as exc:
            self._raise("streamed tool arguments must be valid JSON", cause=exc)
        if not isinstance(arguments, Mapping):
            self._raise("streamed tool arguments must be a JSON object")
        return ToolCall(buffer.id, buffer.name, cast(Mapping[str, Any], arguments))

    def _consistent_value(
        self,
        current: str | None,
        update: str | None,
        label: str,
    ) -> str | None:
        if update is None:
            return current
        if current is not None and current != update:
            self._raise(f"tool call delta {label} changed for one index")
        return update

    def _raise(self, message: str, *, cause: Exception | None = None) -> NoReturn:
        error = self._error_factory(message)
        if cause is None:
            raise error
        raise error from cause
