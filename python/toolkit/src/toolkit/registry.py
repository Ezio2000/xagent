"""Default tool registry implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from kernel import (
    DuplicateToolError,
    InvalidToolCall,
    ToolAcceptance,
    ToolCall,
    ToolError,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)

from toolkit.tool import (
    AcceptableTool,
    ExecutableTool,
    InvocableTool,
    RuntimeContextSnapshot,
    Tool,
    ToolCancelChecker,
    ToolExecutionContext,
    ToolInvocation,
    ToolProgressEmitter,
)

_RESERVED_TOOL_OUTPUT_KINDS = {"observation", "acceptance", "rejection"}


class ToolRegistry:
    """O(1) tool lookup with cached model-neutral specs."""

    __slots__ = ("_argument_validators", "_specs", "_specs_by_name", "_tools")

    _argument_validators: dict[str, Any]
    _specs: tuple[ToolSpec, ...]
    _specs_by_name: dict[str, ToolSpec]
    _tools: dict[str, Tool]

    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools = {}
        self._specs_by_name = {}
        self._argument_validators = {}
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

    def validate_call(self, call: ToolCall) -> None:
        """Validate a tool call without invoking the concrete tool implementation."""

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
        if call.mode == "execute":
            if not callable(getattr(tool, "execute", None)):
                raise InvalidToolCall(f"tool {call.name} does not implement execute")
            return
        if call.mode == "accept":
            if not callable(getattr(tool, "accept", None)):
                raise InvalidToolCall(f"tool {call.name} does not implement accept")
            return
        if not callable(getattr(tool, "invoke", None)):
            raise InvalidToolCall(f"tool {call.name} does not implement {call.mode} mode")

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContextSnapshot,
        *,
        progress_emitter: ToolProgressEmitter | None = None,
        cancel_checker: ToolCancelChecker | None = None,
    ) -> ToolOutput:
        self.validate_call(call)
        tool = self._tools[call.name]
        invocation = ToolInvocation.from_tool_call(call)
        tool_context = ToolExecutionContext.from_runtime_context(
            context,
            progress_emitter=progress_emitter,
            cancel_checker=cancel_checker,
        )
        if call.mode == "execute":
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

        invocable = cast(InvocableTool, tool)
        try:
            result = cast(object, await invocable.invoke(invocation, tool_context))
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc) or exc.__class__.__name__) from exc
        if not isinstance(result, ToolOutput):
            raise TypeError("tool invoke must return ToolOutput")
        if type(result) is ToolOutput and result.kind in _RESERVED_TOOL_OUTPUT_KINDS:
            raise TypeError("custom tool invoke must return an extension ToolOutput kind")
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
        _validate_json_schema(spec.input_schema, "tool input_schema")
        if spec.output_schema is not None:
            _validate_json_schema(spec.output_schema, "tool output_schema")
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


def _validate_json_schema(schema: Mapping[str, Any], label: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{label} must be a valid JSON Schema: {exc.message}") from exc


def _build_json_schema_validator(schema: Mapping[str, Any]) -> Any:
    return cast(Any, Draft202012Validator(schema))
