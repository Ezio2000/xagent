"""Tool protocol and registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol, cast

from agent_runtime.control import PauseRequest
from agent_runtime.errors import DuplicateToolError, InvalidToolCall
from agent_runtime.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_part_without_metadata,
    content_parts_summary,
)
from agent_runtime.runtime import RuntimeContext


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


@dataclass(slots=True)
class ToolSpec:
    """Model-neutral tool contract exposed to model adapters."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any] | None = None
    annotations: Mapping[str, Any] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        self.name = _expect_str(self.name, "tool name")
        self.description = _expect_str(self.description, "tool description")
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError("tool description must not be empty")
        self.input_schema = _copy_mapping(self.input_schema)
        if self.output_schema is not None:
            self.output_schema = _copy_mapping(self.output_schema)
        self.annotations = _copy_mapping(self.annotations)
        self.metadata = _copy_mapping(self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolSpec:
        known = {
            "name",
            "description",
            "input_schema",
            "output_schema",
            "annotations",
            "metadata",
        }
        _reject_unknown_keys(value, known, "tool spec")
        output_schema = value.get("output_schema")
        raw_annotations: object = value.get("annotations", {})
        raw_metadata: object = value.get("metadata", {})
        return cls(
            name=_expect_str(value["name"], "tool name"),
            description=_expect_str(value["description"], "tool description"),
            input_schema=_expect_mapping(value["input_schema"], "tool input_schema"),
            output_schema=None
            if output_schema is None
            else _expect_mapping(output_schema, "tool output_schema"),
            annotations=_expect_mapping(raw_annotations, "tool annotations"),
            metadata=_expect_mapping(raw_metadata, "tool metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": _copy_mapping(self.input_schema),
        }
        if self.output_schema is not None:
            data["output_schema"] = _copy_mapping(self.output_schema)
        if self.annotations:
            data["annotations"] = _copy_mapping(self.annotations)
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


@dataclass(slots=True)
class ToolResult:
    """Normalized tool result."""

    parts: list[ContentPart]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    is_error: bool = False
    pause: PauseRequest | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.is_error), bool):
            raise TypeError("tool result is_error must be a boolean")
        if self.pause is not None and not isinstance(cast(object, self.pause), PauseRequest):
            raise TypeError("tool result pause must be a PauseRequest or None")
        parts: list[ContentPart] = []
        for part in _expect_sequence(self.parts, "tool result parts"):
            if not isinstance(part, ContentPart):
                _raise_type("tool result parts items must be ContentPart")
            parts.append(ContentPart.from_dict(part.to_dict()))
        self.parts = parts
        self.metadata = _copy_mapping(_expect_mapping(self.metadata, "tool result metadata"))
        if self.pause is not None and self.pause.interrupt:
            raise ValueError("tool result pause cannot interrupt model execution")

    @classmethod
    def text(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        is_error: bool = False,
    ) -> ToolResult:
        return cls(
            parts=[ContentPart.text_part(text)],
            metadata={} if metadata is None else metadata,
            is_error=is_error,
        )

    @classmethod
    def waiting(
        cls,
        text: str,
        *,
        wait_id: str,
        reason: str = "external_wait",
        metadata: Mapping[str, Any] | None = None,
        pause_metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            parts=[ContentPart.text_part(text)],
            metadata={} if metadata is None else metadata,
            pause=PauseRequest(
                reason=reason,
                source="tool",
                wait_id=wait_id,
                metadata={} if pause_metadata is None else pause_metadata,
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResult:
        known = {"parts", "metadata", "is_error", "pause"}
        _reject_unknown_keys(value, known, "tool result")
        raw_pause = value.get("pause")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "tool result part"))
                for part in _expect_sequence(value["parts"], "tool result parts")
            ],
            metadata=_expect_mapping(raw_metadata, "tool result metadata"),
            is_error=_expect_bool(value["is_error"], "tool result is_error"),
            pause=None
            if raw_pause is None
            else PauseRequest.from_dict(_expect_mapping(raw_pause, "tool result pause")),
        )

    def to_message(self, call: ToolCall) -> Message:
        metadata: dict[str, Any] = {}
        if self.is_error:
            metadata["is_error"] = True
        return Message.tool(
            [content_part_without_metadata(part) for part in self.parts],
            call.id,
            metadata=metadata,
        )

    @property
    def text_content(self) -> str:
        return "".join(part.text or "" for part in self.parts if part.type == "text")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "parts": [part.to_dict() for part in self.parts],
            "is_error": self.is_error,
        }
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        if self.pause is not None:
            data["pause"] = self.pause.to_dict()
        return data

    def summary(self) -> dict[str, Any]:
        return content_parts_summary(self.parts) | {
            "is_error": self.is_error,
            "metadata": _copy_mapping(self.metadata),
            "pause": None if self.pause is None else self.pause.to_dict(),
        }


class Tool(Protocol):
    """Protocol implemented by runtime tools."""

    spec: ToolSpec

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        """Execute the tool."""
        ...


class ToolRegistry:
    """O(1) tool lookup with cached model-neutral specs."""

    __slots__ = ("_specs", "_tools")

    _specs: tuple[ToolSpec, ...]
    _tools: dict[str, Tool]

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        specs: list[ToolSpec] = []
        if tools:
            for tool in tools:
                specs.append(self._register_without_rebuild(tool))
        self._specs = tuple(specs)

    def register(self, tool: Tool) -> None:
        spec = self._register_without_rebuild(tool)
        self._specs = (*self._specs, spec)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(ToolSpec.from_dict(spec.to_dict()) for spec in self._specs)

    def spec_for(self, name: str) -> ToolSpec | None:
        for spec in self._specs:
            if spec.name == name:
                return ToolSpec.from_dict(spec.to_dict())
        return None

    async def execute(self, call: ToolCall, context: RuntimeContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        result = cast(object, await tool.execute(deepcopy(dict(call.arguments)), context))
        if not isinstance(result, ToolResult):
            raise TypeError("tool execute must return ToolResult")
        return ToolResult.from_dict(result.to_dict())

    def _register_without_rebuild(self, tool: Tool) -> ToolSpec:
        spec = ToolSpec.from_dict(tool.spec.to_dict())
        if spec.name in self._tools:
            raise DuplicateToolError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = tool
        return spec


def _raise_type(message: str) -> NoReturn:
    raise TypeError(message)
