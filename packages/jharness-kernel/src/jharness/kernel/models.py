"""Immutable provider-neutral model protocol values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, runtime_checkable

from jharness.kernel._validation import (
    expect_bool,
    expect_instance,
    expect_instance_tuple,
    expect_int,
    expect_non_empty_str,
    expect_nonnegative_int,
    expect_number,
    expect_optional_int,
    expect_str,
    freeze_mapping,
)
from jharness.kernel.context import RunContext
from jharness.kernel.messages import ContentPart, Message, ToolCall

if TYPE_CHECKING:
    from jharness.kernel.tools import ToolSpec


@dataclass(frozen=True, slots=True)
class ModelOptions:
    """Portable model sampling and output options."""

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        if self.model is not None:
            expect_str(self.model, "model")
        if self.temperature is not None:
            expect_number(self.temperature, "temperature")
        if self.top_p is not None:
            expect_number(self.top_p, "top_p")
        if (
            self.max_output_tokens is not None
            and expect_int(self.max_output_tokens, "max_output_tokens") < 1
        ):
            raise ValueError("max_output_tokens must be >= 1")
        if self.stop is not None:
            object.__setattr__(self, "stop", expect_instance_tuple(self.stop, str, "stop"))
        expect_optional_int(self.seed, "seed")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "model metadata"))


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Portable model-side tool selection intent."""

    type: str = "auto"
    name: str | None = None
    allow_parallel_tool_calls: bool = True

    def __post_init__(self) -> None:
        choice_type = expect_str(self.type, "tool choice type")
        if choice_type not in {"auto", "none", "required", "named"}:
            raise ValueError(f"unsupported tool choice: {choice_type}")
        if choice_type == "named":
            if self.name is None or not expect_str(self.name, "tool choice name"):
                raise ValueError("named tool choice requires name")
        elif self.name is not None:
            raise ValueError("only named tool choice may carry name")
        expect_bool(self.allow_parallel_tool_calls, "allow_parallel_tool_calls")


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Portable final response-format request."""

    type: str
    schema: Mapping[str, Any] | bool | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        format_type = expect_str(self.type, "response format type")
        expect_bool(self.strict, "response format strict")
        if format_type not in {"text", "json_object", "json_schema"}:
            raise ValueError(f"unsupported response format: {format_type}")
        if format_type == "json_schema":
            if not isinstance(self.schema, Mapping | bool):
                raise TypeError("json_schema response format requires schema")
            if isinstance(self.schema, Mapping):
                object.__setattr__(self, "schema", freeze_mapping(self.schema, "response schema"))
        elif self.schema is not None:
            raise ValueError("only json_schema response format may carry schema")
        elif self.strict:
            raise ValueError("only json_schema response format may be strict")


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Standard cumulative model usage fields."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in _USAGE_FIELDS:
            value = cast(int | None, getattr(self, name))
            if value is not None:
                expect_nonnegative_int(value, name)

    def add(self, other: ModelUsage | None) -> ModelUsage:
        """Add independently reported usage counters field by field."""

        if other is None:
            return self
        expect_instance(other, ModelUsage, "model usage")
        values: dict[str, int | None] = {}
        for name in _USAGE_FIELDS:
            current = cast(int | None, getattr(self, name))
            added = cast(int | None, getattr(other, name))
            values[name] = current if added is None else (current or 0) + added
        return ModelUsage(**values)

    def merge_snapshot(self, other: ModelUsage) -> ModelUsage:
        """Replace fields reported by a later cumulative usage snapshot."""

        expect_instance(other, ModelUsage, "model usage snapshot")
        return ModelUsage(
            **{
                name: (
                    getattr(other, name)
                    if getattr(other, name) is not None
                    else getattr(self, name)
                )
                for name in _USAGE_FIELDS
            }
        )

    def with_fallback(self, fallback: ModelUsage | None) -> ModelUsage:
        """Fill fields omitted by this usage value from a fallback value."""

        if fallback is None:
            return self
        expect_instance(fallback, ModelUsage, "fallback model usage")
        return ModelUsage(
            **{
                name: (
                    getattr(self, name)
                    if getattr(self, name) is not None
                    else getattr(fallback, name)
                )
                for name in _USAGE_FIELDS
            }
        )


_CAPABILITY_FIELDS = (
    "streaming",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "multimodal_input",
    "multimodal_output",
    "structured_output",
    "json_mode",
    "usage_reporting",
)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Immutable advertised model capabilities."""

    streaming: bool = False
    tools: bool = True
    tool_choice: bool = True
    parallel_tool_calls: bool = True
    multimodal_input: bool = True
    multimodal_output: bool = True
    structured_output: bool = True
    json_mode: bool = True
    usage_reporting: bool = True

    def __post_init__(self) -> None:
        for name in _CAPABILITY_FIELDS:
            expect_bool(getattr(self, name), f"model capability {name}")


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete provider-neutral model input."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    options: ModelOptions = field(default_factory=ModelOptions)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    response_format: ResponseFormat | None = None

    def __post_init__(self) -> None:
        from jharness.kernel.tools import ToolSpec

        messages = expect_instance_tuple(self.messages, Message, "model request messages")
        tools = expect_instance_tuple(self.tools, ToolSpec, "model request tools")
        if not messages:
            raise ValueError("model request requires messages")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("model request tool names must be unique")
        expect_instance(self.options, ModelOptions, "model request options")
        expect_instance(self.tool_choice, ToolChoice, "model request tool_choice")
        if self.response_format is not None:
            expect_instance(self.response_format, ResponseFormat, "model request response_format")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """The sole complete provider-neutral model result."""

    parts: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    model_id: str | None = None
    response_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        parts = expect_instance_tuple(self.parts, ContentPart, "model response parts")
        calls = expect_instance_tuple(self.tool_calls, ToolCall, "model response tool_calls")
        if not parts and not calls:
            raise ValueError("model response requires parts or tool calls")
        ids = [call.id for call in calls]
        if len(ids) != len(set(ids)):
            raise ValueError("model response tool call ids must be unique")
        if self.finish_reason is not None:
            expect_str(self.finish_reason, "finish_reason")
        if self.usage is not None:
            expect_instance(self.usage, ModelUsage, "model response usage")
        if self.model_id is not None:
            expect_str(self.model_id, "model_id")
        if self.response_id is not None:
            expect_str(self.response_id, "response_id")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "model metadata"))

    def to_assistant_message(self) -> Message:
        """Project the complete response into durable conversation history."""

        return Message.assistant(self.parts, tool_calls=self.tool_calls)


@dataclass(frozen=True, slots=True)
class ModelContentDelta:
    """Incremental content for one zero-based response-part position."""

    index: int
    text_delta: str
    part_type: str = "text"
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.index, "content delta index")
        expect_str(self.text_delta, "content delta text")
        expect_non_empty_str(self.part_type, "content delta part_type")
        object.__setattr__(self, "data", freeze_mapping(self.data, "content delta data"))


@dataclass(frozen=True, slots=True)
class ModelToolCallDelta:
    """Incremental JSON arguments and identity for one ordered tool call."""

    index: int
    arguments_delta: str
    id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.index, "tool call delta index")
        expect_str(self.arguments_delta, "tool call arguments_delta")
        if self.id is not None:
            expect_str(self.id, "tool call delta id")
        if self.name is not None:
            expect_str(self.name, "tool call delta name")
        if self.id == "" or self.name == "":
            raise ValueError("tool call delta id and name must not be empty")


@dataclass(frozen=True, slots=True)
class ModelReasoningDelta:
    """Incremental reasoning text for one zero-based response position."""

    index: int
    text_delta: str

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.index, "reasoning delta index")
        expect_str(self.text_delta, "reasoning delta text")


@dataclass(frozen=True, slots=True)
class ModelUsageDelta:
    """A cumulative provider usage snapshot."""

    usage: ModelUsage

    def __post_init__(self) -> None:
        expect_instance(self.usage, ModelUsage, "usage delta")


ModelDelta: TypeAlias = (
    ModelContentDelta | ModelToolCallDelta | ModelReasoningDelta | ModelUsageDelta
)


class DeltaSink(Protocol):
    """Ordered async observer for live-only model deltas."""

    async def __call__(self, delta: ModelDelta, /) -> None: ...


@runtime_checkable
class Model(Protocol):
    """One provider-neutral model operation with optional live deltas."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse: ...
