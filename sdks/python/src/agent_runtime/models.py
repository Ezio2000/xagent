"""Model client protocol and normalized model data types."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, cast

from agent_runtime.errors import AgentError
from agent_runtime.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_parts_summary,
)
from agent_runtime.runtime import RuntimeContext
from agent_runtime.tools import ToolSpec

ToolChoiceMode = Literal["auto", "none", "required", "tool"]
ResponseFormatType = Literal["text", "json_object", "json_schema"]


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _empty_parts() -> list[ContentPart]:
    return []


def _empty_tool_calls() -> list[ToolCall]:
    return []


def _empty_stop_sequences() -> tuple[str, ...]:
    return ()


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value or {}))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _copy_extra(value: Mapping[str, Any] | None, reserved: set[str], label: str) -> dict[str, Any]:
    extra = _copy_mapping(value)
    conflicts = reserved & extra.keys()
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"{label} extra cannot override reserved field(s): {names}")
    return extra


@dataclass(slots=True)
class ModelOptions:
    """Common provider-neutral model call options."""

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: tuple[str, ...] = field(default_factory=_empty_stop_sequences)
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.model is not None and not self.model:
            raise ValueError("model option model must not be empty")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be > 0 and <= 1")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        self.stop_sequences = tuple(str(item) for item in self.stop_sequences)
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {
                "model",
                "temperature",
                "top_p",
                "max_output_tokens",
                "stop_sequences",
                "seed",
                "metadata",
            },
            "model options",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelOptions:
        known = {
            "model",
            "temperature",
            "top_p",
            "max_output_tokens",
            "stop_sequences",
            "seed",
            "metadata",
        }
        return cls(
            model=cast(str | None, value.get("model")),
            temperature=cast(float | None, value.get("temperature")),
            top_p=cast(float | None, value.get("top_p")),
            max_output_tokens=cast(int | None, value.get("max_output_tokens")),
            stop_sequences=tuple(str(item) for item in value.get("stop_sequences") or ()),
            seed=cast(int | None, value.get("seed")),
            metadata=_expect_mapping(value.get("metadata") or {}, "model options metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.model is not None:
            data["model"] = self.model
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.max_output_tokens is not None:
            data["max_output_tokens"] = self.max_output_tokens
        if self.stop_sequences:
            data["stop_sequences"] = list(self.stop_sequences)
        if self.seed is not None:
            data["seed"] = self.seed
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {
                    "model",
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "stop_sequences",
                    "seed",
                    "metadata",
                },
                "model options",
            )
        )
        return data


@dataclass(slots=True)
class ToolChoice:
    """Provider-neutral tool-use preference for a model call."""

    mode: ToolChoiceMode = "auto"
    name: str | None = None
    allow_parallel_tool_calls: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "required", "tool"}:
            raise ValueError(f"unsupported tool choice mode: {self.mode}")
        if self.mode == "tool" and not self.name:
            raise ValueError("tool choice mode 'tool' requires name")
        if self.mode != "tool" and self.name is not None:
            raise ValueError("tool choice name is only valid when mode is 'tool'")
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"mode", "name", "allow_parallel_tool_calls", "metadata"},
            "tool choice",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolChoice:
        known = {"mode", "name", "allow_parallel_tool_calls", "metadata"}
        return cls(
            mode=cast(ToolChoiceMode, str(value.get("mode", "auto"))),
            name=cast(str | None, value.get("name")),
            allow_parallel_tool_calls=cast(bool | None, value.get("allow_parallel_tool_calls")),
            metadata=_expect_mapping(value.get("metadata") or {}, "tool choice metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode}
        if self.name is not None:
            data["name"] = self.name
        if self.allow_parallel_tool_calls is not None:
            data["allow_parallel_tool_calls"] = self.allow_parallel_tool_calls
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {"mode", "name", "allow_parallel_tool_calls", "metadata"},
                "tool choice",
            )
        )
        return data


@dataclass(slots=True)
class ResponseFormat:
    """Provider-neutral response formatting request."""

    type: ResponseFormatType = "text"
    json_schema: Mapping[str, Any] | None = None
    strict: bool = False
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.type not in {"text", "json_object", "json_schema"}:
            raise ValueError(f"unsupported response format type: {self.type}")
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("json_schema response format requires json_schema")
        if self.type != "json_schema" and self.json_schema is not None:
            raise ValueError("json_schema is only valid for json_schema response format")
        if self.json_schema is not None:
            self.json_schema = _copy_mapping(self.json_schema)
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"type", "json_schema", "strict", "metadata"},
            "response format",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResponseFormat:
        known = {"type", "json_schema", "strict", "metadata"}
        raw_schema = value.get("json_schema")
        return cls(
            type=cast(ResponseFormatType, str(value.get("type", "text"))),
            json_schema=None
            if raw_schema is None
            else _expect_mapping(raw_schema, "response format json_schema"),
            strict=bool(value.get("strict", False)),
            metadata=_expect_mapping(value.get("metadata") or {}, "response format metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type, "strict": self.strict}
        if self.json_schema is not None:
            data["json_schema"] = _copy_mapping(self.json_schema)
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {"type", "json_schema", "strict", "metadata"},
                "response format",
            )
        )
        return data


@dataclass(slots=True)
class ModelUsage:
    """Provider-neutral token usage information."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "metadata",
            },
            "model usage",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelUsage:
        known = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "metadata",
        }
        return cls(
            input_tokens=cast(int | None, value.get("input_tokens")),
            output_tokens=cast(int | None, value.get("output_tokens")),
            total_tokens=cast(int | None, value.get("total_tokens")),
            reasoning_tokens=cast(int | None, value.get("reasoning_tokens")),
            cache_read_tokens=cast(int | None, value.get("cache_read_tokens")),
            cache_write_tokens=cast(int | None, value.get("cache_write_tokens")),
            metadata=_expect_mapping(value.get("metadata") or {}, "model usage metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "reasoning_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "metadata",
                },
                "model usage",
            )
        )
        return data


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    """Capabilities advertised by a model adapter."""

    streaming: bool = False
    tools: bool = False
    tool_choice: bool = False
    parallel_tool_calls: bool = False
    multimodal_input: bool = False
    multimodal_output: bool = False
    structured_output: bool = False
    json_mode: bool = False
    usage: bool = False
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))
        object.__setattr__(
            self,
            "extra",
            _copy_extra(
                self.extra,
                {
                    "streaming",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "multimodal_input",
                    "multimodal_output",
                    "structured_output",
                    "json_mode",
                    "usage",
                    "metadata",
                },
                "model capabilities",
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelCapabilities:
        known = {
            "streaming",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "multimodal_input",
            "multimodal_output",
            "structured_output",
            "json_mode",
            "usage",
            "metadata",
        }
        return cls(
            streaming=bool(value.get("streaming", False)),
            tools=bool(value.get("tools", False)),
            tool_choice=bool(value.get("tool_choice", False)),
            parallel_tool_calls=bool(value.get("parallel_tool_calls", False)),
            multimodal_input=bool(value.get("multimodal_input", False)),
            multimodal_output=bool(value.get("multimodal_output", False)),
            structured_output=bool(value.get("structured_output", False)),
            json_mode=bool(value.get("json_mode", False)),
            usage=bool(value.get("usage", False)),
            metadata=_expect_mapping(value.get("metadata") or {}, "model capabilities metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "streaming": self.streaming,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "multimodal_input": self.multimodal_input,
            "multimodal_output": self.multimodal_output,
            "structured_output": self.structured_output,
            "json_mode": self.json_mode,
            "usage": self.usage,
        }
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {
                    "streaming",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "multimodal_input",
                    "multimodal_output",
                    "structured_output",
                    "json_mode",
                    "usage",
                    "metadata",
                },
                "model capabilities",
            )
        )
        return data


@dataclass(slots=True)
class ModelRequest:
    """Model-neutral request passed to model adapters."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    options: ModelOptions = field(default_factory=ModelOptions)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    response_format: ResponseFormat | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.messages = tuple(Message.from_dict(message.to_dict()) for message in self.messages)
        self.tools = tuple(ToolSpec.from_dict(tool.to_dict()) for tool in self.tools)
        self.options = ModelOptions.from_dict(self.options.to_dict())
        self.tool_choice = ToolChoice.from_dict(self.tool_choice.to_dict())
        if self.response_format is not None:
            self.response_format = ResponseFormat.from_dict(self.response_format.to_dict())
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {"messages", "tools", "options", "tool_choice", "response_format", "metadata"},
            "model request",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelRequest:
        known = {"messages", "tools", "options", "tool_choice", "response_format", "metadata"}
        raw_response_format = value.get("response_format")
        return cls(
            messages=tuple(
                Message.from_dict(_expect_mapping(message, "model request message"))
                for message in cast(Sequence[object], value.get("messages") or ())
            ),
            tools=tuple(
                ToolSpec.from_dict(_expect_mapping(tool, "model request tool"))
                for tool in cast(Sequence[object], value.get("tools") or ())
            ),
            options=ModelOptions.from_dict(
                _expect_mapping(value.get("options") or {}, "model request options")
            ),
            tool_choice=ToolChoice.from_dict(
                _expect_mapping(value.get("tool_choice") or {}, "model request tool_choice")
            ),
            response_format=None
            if raw_response_format is None
            else ResponseFormat.from_dict(
                _expect_mapping(raw_response_format, "model request response_format")
            ),
            metadata=_expect_mapping(value.get("metadata") or {}, "model request metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "messages": [message.to_dict() for message in self.messages],
            "tools": [tool.to_dict() for tool in self.tools],
            "options": self.options.to_dict(),
            "tool_choice": self.tool_choice.to_dict(),
        }
        if self.response_format is not None:
            data["response_format"] = self.response_format.to_dict()
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {"messages", "tools", "options", "tool_choice", "response_format", "metadata"},
                "model request",
            )
        )
        return data


@dataclass(slots=True)
class ModelResponse:
    """Normalized model response returned by model adapters."""

    parts: list[ContentPart] = field(default_factory=_empty_parts)
    tool_calls: list[ToolCall] = field(default_factory=_empty_tool_calls)
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    model: str | None = None
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    extra: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.parts = [ContentPart.from_dict(part.to_dict()) for part in self.parts]
        self.tool_calls = [ToolCall.from_dict(call.to_dict()) for call in self.tool_calls]
        if self.finish_reason is not None and not self.finish_reason:
            raise ValueError("finish_reason must not be empty")
        if self.usage is not None:
            self.usage = ModelUsage.from_dict(self.usage.to_dict())
        if self.model is not None and not self.model:
            raise ValueError("model response model must not be empty")
        if self.response_id is not None and not self.response_id:
            raise ValueError("model response response_id must not be empty")
        self.metadata = _copy_mapping(self.metadata)
        self.extra = _copy_extra(
            self.extra,
            {
                "parts",
                "tool_calls",
                "finish_reason",
                "usage",
                "model",
                "response_id",
                "metadata",
            },
            "model response",
        )

    @classmethod
    def text(cls, text: str, tool_calls: Sequence[ToolCall] | None = None) -> ModelResponse:
        return cls(parts=[ContentPart.text_part(text)], tool_calls=list(tool_calls or ()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelResponse:
        known = {
            "parts",
            "tool_calls",
            "finish_reason",
            "usage",
            "model",
            "response_id",
            "metadata",
        }
        raw_usage = value.get("usage")
        return cls(
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "model response part"))
                for part in cast(Sequence[object], value.get("parts") or ())
            ],
            tool_calls=[
                ToolCall.from_dict(_expect_mapping(call, "model response tool call"))
                for call in cast(Sequence[object], value.get("tool_calls") or ())
            ],
            finish_reason=cast(str | None, value.get("finish_reason")),
            usage=None
            if raw_usage is None
            else ModelUsage.from_dict(_expect_mapping(raw_usage, "model response usage")),
            model=cast(str | None, value.get("model")),
            response_id=cast(str | None, value.get("response_id")),
            metadata=_expect_mapping(value.get("metadata") or {}, "model response metadata"),
            extra={key: deepcopy(item) for key, item in value.items() if key not in known},
        )

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def text_content(self) -> str:
        return "".join(part.text or "" for part in self.parts if part.type == "text")

    def to_assistant_message(self) -> Message:
        metadata = _copy_mapping(self.metadata)
        if self.finish_reason is not None:
            metadata["finish_reason"] = self.finish_reason
        if self.usage is not None:
            metadata["usage"] = self.usage.to_dict()
        if self.model is not None:
            metadata["model"] = self.model
        if self.response_id is not None:
            metadata["response_id"] = self.response_id
        return Message.assistant(
            self.parts,
            self.tool_calls,
            metadata=metadata,
            extra=self.extra,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "parts": [part.to_dict() for part in self.parts],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
        if self.model is not None:
            data["model"] = self.model
        if self.response_id is not None:
            data["response_id"] = self.response_id
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        data.update(
            _copy_extra(
                self.extra,
                {
                    "parts",
                    "tool_calls",
                    "finish_reason",
                    "usage",
                    "model",
                    "response_id",
                    "metadata",
                },
                "model response",
            )
        )
        return data

    def summary(self) -> dict[str, Any]:
        data = content_parts_summary(self.parts) | {
            "tool_call_count": len(self.tool_calls),
            "has_tool_calls": bool(self.tool_calls),
        }
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        if self.usage is not None:
            data["usage"] = self.usage.to_dict()
        if self.model is not None:
            data["model"] = self.model
        if self.response_id is not None:
            data["response_id"] = self.response_id
        return data


@dataclass(slots=True, frozen=True)
class ModelStreamStarted:
    """A model stream has started."""

    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True, frozen=True)
class ModelContentDelta:
    """Incremental visible model content."""

    index: int
    text_delta: str
    part_type: str = "text"
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("content delta index must be >= 0")
        if not self.part_type:
            raise ValueError("content delta part_type must not be empty")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True, frozen=True)
class ModelReasoningDelta:
    """Incremental model reasoning content."""

    index: int
    text_delta: str
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("reasoning delta index must be >= 0")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True, frozen=True)
class ModelToolCallDelta:
    """Incremental model tool-call construction."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("tool call delta index must be >= 0")
        if self.id is not None and not self.id:
            raise ValueError("tool call delta id must not be empty")
        if self.name is not None and not self.name:
            raise ValueError("tool call delta name must not be empty")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True, frozen=True)
class ModelUsageDelta:
    """Incremental or final model usage data."""

    usage: ModelUsage
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", ModelUsage.from_dict(self.usage.to_dict()))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True, frozen=True)
class ModelStreamCompleted:
    """A model stream has completed with a normalized response."""

    response: ModelResponse

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", ModelResponse.from_dict(self.response.to_dict()))


ModelStreamEvent: TypeAlias = (
    ModelStreamStarted
    | ModelContentDelta
    | ModelReasoningDelta
    | ModelToolCallDelta
    | ModelUsageDelta
    | ModelStreamCompleted
)


@dataclass(slots=True)
class _ContentBuffer:
    part_type: str
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)


@dataclass(slots=True)
class _ToolCallBuffer:
    id: str | None = None
    name: str | None = None
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
            if buffer.part_type != "text":
                raise AgentError(f"unsupported streamed content part type: {buffer.part_type}")
            parts.append(ContentPart.text_part(buffer.text, metadata=buffer.metadata))

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


def stream_event_to_delta_payload(event: ModelStreamEvent) -> dict[str, Any] | None:
    """Convert a model stream event into a public model_delta payload."""

    if isinstance(event, ModelStreamStarted | ModelStreamCompleted):
        return None
    if isinstance(event, ModelContentDelta):
        data: dict[str, Any] = {
            "kind": "text_delta",
            "index": event.index,
            "text_delta": event.text_delta,
            "part_type": event.part_type,
        }
        if event.metadata:
            data["metadata"] = _copy_mapping(event.metadata)
        return data
    if isinstance(event, ModelReasoningDelta):
        data = {
            "kind": "reasoning_delta",
            "index": event.index,
            "text_delta": event.text_delta,
        }
        if event.metadata:
            data["metadata"] = _copy_mapping(event.metadata)
        return data
    if isinstance(event, ModelToolCallDelta):
        data = {"kind": "tool_call_delta", "index": event.index}
        if event.id is not None:
            data["id"] = event.id
        if event.name is not None:
            data["name"] = event.name
        if event.arguments_delta is not None:
            data["arguments_delta"] = event.arguments_delta
        if event.metadata:
            data["metadata"] = _copy_mapping(event.metadata)
        return data
    data = {"kind": "usage_delta", "usage": event.usage.to_dict()}
    if event.metadata:
        data["metadata"] = _copy_mapping(event.metadata)
    return data


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


class ModelClient(Protocol):
    """Protocol implemented by model adapters."""

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        """Return the next model decision."""
        ...


class StreamingModelClient(ModelClient, Protocol):
    """Optional protocol implemented by streaming model adapters."""

    def stream(
        self, request: ModelRequest, context: RuntimeContext
    ) -> AsyncIterator[ModelStreamEvent]:
        """Yield provider-neutral model stream events."""
        ...
