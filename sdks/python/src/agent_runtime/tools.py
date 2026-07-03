"""Tool protocol and registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from agent_runtime.control import PauseRequest
from agent_runtime.errors import DuplicateToolError, InvalidToolCall, ToolError
from agent_runtime.messages import (
    ContentPart,
    Message,
    ToolCall,
    content_part_without_metadata,
    content_parts_summary,
)
from agent_runtime.runtime import RuntimeContext

ToolInvocationMode = str
_RESERVED_TOOL_OUTPUT_KINDS = {"observation", "acceptance", "rejection"}


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _empty_modes() -> tuple[ToolInvocationMode, ...]:
    return ("execute",)


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


def _expect_mode(value: object, label: str) -> ToolInvocationMode:
    mode = _expect_str(value, label)
    if not mode:
        raise ValueError(f"{label} must not be empty")
    return mode


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


@dataclass(slots=True, frozen=True)
class ToolInvocation:
    """A concrete tool invocation selected by the model."""

    id: str
    name: str
    mode: ToolInvocationMode
    arguments: Mapping[str, Any] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _expect_str(self.id, "tool invocation id"))
        object.__setattr__(self, "name", _expect_str(self.name, "tool invocation name"))
        object.__setattr__(self, "mode", _expect_mode(self.mode, "tool invocation mode"))
        if not self.id:
            raise ValueError("tool invocation id must not be empty")
        if not self.name:
            raise ValueError("tool invocation name must not be empty")
        object.__setattr__(self, "arguments", _copy_mapping(self.arguments))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> ToolInvocation:
        return cls(
            id=call.id,
            name=call.name,
            mode=call.mode,
            arguments=call.arguments,
            metadata=call.metadata,
        )

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            id=self.id,
            name=self.name,
            mode=self.mode,
            arguments=self.arguments,
            metadata=self.metadata,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolInvocation:
        known = {"id", "name", "mode", "arguments", "metadata"}
        _reject_unknown_keys(value, known, "tool invocation")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            id=_expect_str(value["id"], "tool invocation id"),
            name=_expect_str(value["name"], "tool invocation name"),
            mode=_expect_mode(value["mode"], "tool invocation mode"),
            arguments=_expect_mapping(value["arguments"], "tool invocation arguments"),
            metadata=_expect_mapping(raw_metadata, "tool invocation metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "mode": self.mode,
            "arguments": _copy_mapping(self.arguments),
        }
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data


class ToolExecutionContext(RuntimeContext):
    """Tool-facing runtime context."""

    @classmethod
    def from_runtime_context(cls, context: RuntimeContext) -> ToolExecutionContext:
        data = context.to_dict()
        tool_context = cls(
            run_id=_expect_str(data["run_id"], "tool context run_id"),
            started_at=cast(float, data["started_at"]),
            deadline=cast(float | None, data["deadline"]),
            metadata=_expect_mapping(data["metadata"], "tool context metadata"),
        )
        tool_context.sequence = cast(int, data["sequence"])
        return tool_context


@dataclass(slots=True)
class ToolSpec:
    """Model-neutral tool contract exposed to model adapters."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    modes: Sequence[ToolInvocationMode] = field(default_factory=_empty_modes)
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
        _validate_json_schema(self.input_schema, "tool input_schema")
        modes: tuple[ToolInvocationMode, ...] = tuple(
            _expect_mode(mode, "tool mode") for mode in self.modes
        )
        if not modes:
            raise ValueError("tool modes must not be empty")
        if len(modes) != len(set(modes)):
            raise ValueError("tool modes must be unique")
        self.modes = modes
        if self.output_schema is not None:
            self.output_schema = _copy_mapping(self.output_schema)
            _validate_json_schema(self.output_schema, "tool output_schema")
        self.annotations = _copy_mapping(self.annotations)
        self.metadata = _copy_mapping(self.metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolSpec:
        known = {
            "name",
            "description",
            "input_schema",
            "modes",
            "output_schema",
            "annotations",
            "metadata",
        }
        _reject_unknown_keys(value, known, "tool spec")
        output_schema = value.get("output_schema")
        raw_modes: object = value["modes"]
        raw_annotations: object = value.get("annotations", {})
        raw_metadata: object = value.get("metadata", {})
        return cls(
            name=_expect_str(value["name"], "tool name"),
            description=_expect_str(value["description"], "tool description"),
            input_schema=_expect_mapping(value["input_schema"], "tool input_schema"),
            modes=tuple(
                _expect_mode(mode, "tool mode")
                for mode in _expect_sequence(raw_modes, "tool modes")
            ),
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
            "modes": list(self.modes),
        }
        if self.output_schema is not None:
            data["output_schema"] = _copy_mapping(self.output_schema)
        if self.annotations:
            data["annotations"] = _copy_mapping(self.annotations)
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        return data

    def supports(self, mode: ToolInvocationMode) -> bool:
        return mode in self.modes


@dataclass(slots=True)
class ToolOutput:
    """Model-neutral output produced by a tool invocation."""

    kind: str
    parts: list[ContentPart]
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)
    is_error: bool = False
    pause: PauseRequest | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        self.kind = _expect_str(self.kind, "tool output kind")
        if not self.kind:
            raise ValueError("tool output kind must not be empty")
        self.is_error = _expect_bool(self.is_error, "tool output is_error")
        self.correlation_id = _expect_optional_str(
            self.correlation_id, "tool output correlation_id"
        )
        if self.correlation_id is not None and not self.correlation_id:
            raise ValueError("tool output correlation_id must not be empty")
        if self.pause is not None and not isinstance(cast(object, self.pause), PauseRequest):
            raise TypeError("tool output pause must be a PauseRequest or None")
        parts: list[ContentPart] = []
        for part in _expect_sequence(self.parts, "tool output parts"):
            if not isinstance(part, ContentPart):
                _raise_type("tool output parts items must be ContentPart")
            parts.append(ContentPart.from_dict(part.to_dict()))
        self.parts = parts
        self.metadata = _copy_mapping(_expect_mapping(self.metadata, "tool output metadata"))
        if self.pause is not None and self.pause.interrupt:
            raise ValueError("tool output pause cannot interrupt model execution")
        if self.kind == "acceptance":
            if self.correlation_id is None:
                raise ValueError("tool acceptance correlation_id must not be empty")
            if self.is_error:
                raise ValueError("tool acceptance is_error must be false")
            if self.pause is not None:
                raise ValueError("tool acceptance pause must be None")
        if self.kind == "rejection":
            if not self.is_error:
                raise ValueError("tool rejection is_error must be true")
            if self.pause is not None:
                raise ValueError("tool rejection pause must be None")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolOutput:
        known = {"kind", "parts", "metadata", "is_error", "pause", "correlation_id"}
        _reject_unknown_keys(value, known, "tool output")
        raw_pause = value.get("pause")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            kind=_expect_str(value["kind"], "tool output kind"),
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "tool output part"))
                for part in _expect_sequence(value["parts"], "tool output parts")
            ],
            metadata=_expect_mapping(raw_metadata, "tool output metadata"),
            is_error=_expect_bool(value.get("is_error", False), "tool output is_error"),
            pause=None
            if raw_pause is None
            else PauseRequest.from_dict(_expect_mapping(raw_pause, "tool output pause")),
            correlation_id=_expect_optional_str(
                value.get("correlation_id"), "tool output correlation_id"
            ),
        )

    def to_message(self, invocation: ToolInvocation) -> Message:
        metadata: dict[str, Any] = {"result_kind": self.kind}
        if self.is_error:
            metadata["is_error"] = True
        if self.correlation_id is not None:
            metadata["correlation_id"] = self.correlation_id
        return Message.tool(
            [content_part_without_metadata(part) for part in self.parts],
            invocation.id,
            metadata=metadata,
        )

    @property
    def text_content(self) -> str:
        return "".join(part.text or "" for part in self.parts if part.type == "text")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "parts": [part.to_dict() for part in self.parts],
        }
        if self.kind == "observation" or self.is_error:
            data["is_error"] = self.is_error
        if self.metadata:
            data["metadata"] = _copy_mapping(self.metadata)
        if self.pause is not None:
            data["pause"] = self.pause.to_dict()
        if self.correlation_id is not None:
            data["correlation_id"] = self.correlation_id
        return data

    def summary(self) -> dict[str, Any]:
        data = content_parts_summary(self.parts) | {
            "result_kind": self.kind,
            "is_error": self.is_error,
            "metadata": _copy_mapping(self.metadata),
            "pause": None if self.pause is None else self.pause.to_dict(),
        }
        if self.correlation_id is not None:
            data["correlation_id"] = self.correlation_id
        return data


class ToolObservation(ToolOutput):
    """Tool output produced by an execute-mode invocation."""

    __slots__ = ()

    def __init__(
        self,
        parts: list[ContentPart],
        metadata: Mapping[str, Any] | None = None,
        is_error: bool = False,
        pause: PauseRequest | None = None,
    ) -> None:
        super().__init__(
            kind="observation",
            parts=parts,
            metadata={} if metadata is None else metadata,
            is_error=is_error,
            pause=pause,
        )

    @classmethod
    def text(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        is_error: bool = False,
    ) -> ToolObservation:
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
    ) -> ToolObservation:
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
    def from_dict(cls, value: Mapping[str, Any]) -> ToolObservation:
        known = {"kind", "parts", "metadata", "is_error", "pause"}
        _reject_unknown_keys(value, known, "tool observation")
        kind = _expect_str(value["kind"], "tool observation kind")
        if kind != "observation":
            raise ValueError("tool observation kind must be observation")
        raw_pause = value.get("pause")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "tool observation part"))
                for part in _expect_sequence(value["parts"], "tool observation parts")
            ],
            metadata=_expect_mapping(raw_metadata, "tool observation metadata"),
            is_error=_expect_bool(value["is_error"], "tool observation is_error"),
            pause=None
            if raw_pause is None
            else PauseRequest.from_dict(_expect_mapping(raw_pause, "tool observation pause")),
        )


class ToolAcceptance(ToolOutput):
    """Tool output produced by an accept-mode invocation."""

    __slots__ = ()

    def __init__(
        self,
        parts: list[ContentPart],
        correlation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            kind="acceptance",
            parts=parts,
            metadata={} if metadata is None else metadata,
            correlation_id=correlation_id,
        )

    @classmethod
    def text(
        cls,
        text: str,
        *,
        correlation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolAcceptance:
        return cls(
            parts=[ContentPart.text_part(text)],
            correlation_id=correlation_id,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolAcceptance:
        known = {"kind", "parts", "correlation_id", "metadata"}
        _reject_unknown_keys(value, known, "tool acceptance")
        kind = _expect_str(value["kind"], "tool acceptance kind")
        if kind != "acceptance":
            raise ValueError("tool acceptance kind must be acceptance")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "tool acceptance part"))
                for part in _expect_sequence(value["parts"], "tool acceptance parts")
            ],
            correlation_id=_expect_str(value["correlation_id"], "tool acceptance correlation_id"),
            metadata=_expect_mapping(raw_metadata, "tool acceptance metadata"),
        )


class ToolRejection(ToolOutput):
    """Accept-mode output produced when an invocation was not accepted."""

    __slots__ = ()

    def __init__(
        self,
        parts: list[ContentPart],
        metadata: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(
            kind="rejection",
            parts=parts,
            metadata={} if metadata is None else metadata,
            is_error=True,
            correlation_id=correlation_id,
        )

    @classmethod
    def text(
        cls,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> ToolRejection:
        return cls(
            parts=[ContentPart.text_part(text)],
            metadata={} if metadata is None else metadata,
            correlation_id=correlation_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolRejection:
        known = {"kind", "parts", "metadata", "is_error", "correlation_id"}
        _reject_unknown_keys(value, known, "tool rejection")
        kind = _expect_str(value["kind"], "tool rejection kind")
        if kind != "rejection":
            raise ValueError("tool rejection kind must be rejection")
        if not _expect_bool(value["is_error"], "tool rejection is_error"):
            raise ValueError("tool rejection is_error must be true")
        raw_metadata: object = value.get("metadata", {})
        return cls(
            parts=[
                ContentPart.from_dict(_expect_mapping(part, "tool rejection part"))
                for part in _expect_sequence(value["parts"], "tool rejection parts")
            ],
            metadata=_expect_mapping(raw_metadata, "tool rejection metadata"),
            correlation_id=_expect_optional_str(
                value.get("correlation_id"), "tool rejection correlation_id"
            ),
        )


class Tool(Protocol):
    """Protocol implemented by runtime tools."""

    spec: ToolSpec


class ExecutableTool(Tool, Protocol):
    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        """Execute the tool and return its final observation."""
        ...


class AcceptableTool(Tool, Protocol):
    async def accept(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolAcceptance | ToolRejection:
        """Accept the tool invocation for external completion."""
        ...


class InvocableTool(Tool, Protocol):
    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolOutput:
        """Handle an invocation mode that is not natively understood by the core."""
        ...


class ToolRegistry:
    """O(1) tool lookup with cached model-neutral specs."""

    __slots__ = ("_argument_validators", "_specs", "_specs_by_name", "_tools")

    _argument_validators: dict[str, Any]
    _specs: tuple[ToolSpec, ...]
    _specs_by_name: dict[str, ToolSpec]
    _tools: dict[str, Tool]

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._specs_by_name: dict[str, ToolSpec] = {}
        self._argument_validators: dict[str, Any] = {}
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
        spec = self._specs_by_name.get(name)
        if spec is not None:
            return ToolSpec.from_dict(spec.to_dict())
        return None

    async def invoke(self, call: ToolCall, context: RuntimeContext) -> ToolOutput:
        tool = self._tools.get(call.name)
        if tool is None:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        spec = self._specs_by_name.get(call.name)
        if spec is None:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        if not spec.supports(call.mode):
            raise InvalidToolCall(f"tool {call.name} does not support {call.mode} mode")
        validator = self._argument_validators.get(call.name)
        if validator is None:
            raise InvalidToolCall(f"unknown tool: {call.name}")
        self._validate_arguments(validator, call)

        invocation = ToolInvocation.from_tool_call(call)
        tool_context = ToolExecutionContext.from_runtime_context(context)
        if call.mode == "execute":
            if not callable(getattr(tool, "execute", None)):
                raise InvalidToolCall(f"tool {call.name} does not implement execute")
            executable = cast(ExecutableTool, tool)
            try:
                result = cast(object, await executable.execute(invocation, tool_context))
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(str(exc) or exc.__class__.__name__) from exc
            if not isinstance(result, ToolObservation):
                raise TypeError("tool execute must return ToolObservation")
            return ToolObservation.from_dict(result.to_dict())

        if call.mode == "accept":
            if not callable(getattr(tool, "accept", None)):
                raise InvalidToolCall(f"tool {call.name} does not implement accept")
            acceptable = cast(AcceptableTool, tool)
            try:
                result = cast(object, await acceptable.accept(invocation, tool_context))
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(str(exc) or exc.__class__.__name__) from exc
            if isinstance(result, ToolAcceptance):
                return ToolAcceptance.from_dict(result.to_dict())
            if isinstance(result, ToolRejection):
                return ToolRejection.from_dict(result.to_dict())
            raise TypeError("tool accept must return ToolAcceptance or ToolRejection")

        if not callable(getattr(tool, "invoke", None)):
            raise InvalidToolCall(f"tool {call.name} does not implement {call.mode} mode")
        invocable = cast(InvocableTool, tool)
        try:
            result = cast(object, await invocable.invoke(invocation, tool_context))
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc) or exc.__class__.__name__) from exc
        if not isinstance(result, ToolOutput):
            raise TypeError("tool invoke must return ToolOutput")
        output = ToolOutput.from_dict(result.to_dict())
        if output.kind in _RESERVED_TOOL_OUTPUT_KINDS:
            raise TypeError("custom tool invoke must return an extension ToolOutput kind")
        return output

    def _register_without_rebuild(self, tool: Tool) -> ToolSpec:
        spec = ToolSpec.from_dict(tool.spec.to_dict())
        if spec.name in self._tools:
            raise DuplicateToolError(f"duplicate tool name: {spec.name}")
        if "execute" in spec.modes and not callable(getattr(tool, "execute", None)):
            raise TypeError(
                f"tool {spec.name} declares execute mode but does not implement execute"
            )
        if "accept" in spec.modes and not callable(getattr(tool, "accept", None)):
            raise TypeError(f"tool {spec.name} declares accept mode but does not implement accept")
        custom_modes = {mode for mode in spec.modes if mode not in {"execute", "accept"}}
        if custom_modes and not callable(getattr(tool, "invoke", None)):
            modes = ", ".join(sorted(custom_modes))
            raise TypeError(f"tool {spec.name} declares custom mode(s) without invoke: {modes}")
        validator = _build_json_schema_validator(spec.input_schema)
        self._tools[spec.name] = tool
        self._specs_by_name[spec.name] = spec
        self._argument_validators[spec.name] = validator
        return spec

    @staticmethod
    def _validate_arguments(validator: Any, call: ToolCall) -> None:
        try:
            validator.validate(call.arguments)
        except ValidationError as exc:
            raise InvalidToolCall(
                f"tool {call.name} arguments do not match input_schema: {exc.message}"
            ) from exc


def _raise_type(message: str) -> NoReturn:
    raise TypeError(message)


def _validate_json_schema(schema: Mapping[str, Any], label: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{label} must be a valid JSON Schema: {exc.message}") from exc


def _build_json_schema_validator(schema: Mapping[str, Any]) -> Any:
    return cast(Any, Draft202012Validator(schema))
