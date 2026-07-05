"""Tool implementation protocols and tool-facing context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Protocol, cast

from kernel import (
    ToolAcceptance,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)

ToolProgressEmitter = Callable[[Mapping[str, Any]], None]
ToolCancelChecker = Callable[[], bool]


class RuntimeContextSnapshot(Protocol):
    """Minimal context snapshot interface needed by tool invocation."""

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable runtime context shape."""
        ...


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(_expect_mapping(value, "mapping")))


def _expect_mode(value: object, label: str) -> str:
    mode = _expect_str(value, label)
    if not mode:
        raise ValueError(f"{label} must not be empty")
    return mode


@dataclass(slots=True, frozen=True)
class ToolInvocation:
    """A tool-facing invocation selected by the model."""

    id: str
    name: str
    mode: str
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


class ToolExecutionContext:
    """Tool-facing runtime context."""

    __slots__ = (
        "_cancel_checker",
        "_progress_emitter",
        "_sequence",
        "deadline",
        "metadata",
        "parent_run_id",
        "parent_tool_call_id",
        "run_id",
        "run_kind",
        "started_at",
    )

    def __init__(
        self,
        *,
        run_id: str,
        started_at: object,
        deadline: object = None,
        metadata: Mapping[str, Any] | None = None,
        parent_run_id: str | None = None,
        parent_tool_call_id: str | None = None,
        run_kind: str | None = None,
    ) -> None:
        self.run_id = _expect_str(run_id, "tool context run_id")
        if not self.run_id:
            raise ValueError("tool context run_id must not be empty")
        if not isinstance(started_at, int | float) or isinstance(started_at, bool):
            raise TypeError("tool context started_at must be a number")
        self.started_at = float(started_at)
        if deadline is not None and (
            not isinstance(deadline, int | float) or isinstance(deadline, bool)
        ):
            raise TypeError("tool context deadline must be a number or null")
        self.deadline = None if deadline is None else float(deadline)
        if self.deadline is not None and self.deadline <= self.started_at:
            raise ValueError("tool context deadline must be after started_at")
        self.parent_run_id = _expect_optional_str(parent_run_id, "tool context parent_run_id")
        self.parent_tool_call_id = _expect_optional_str(
            parent_tool_call_id, "tool context parent_tool_call_id"
        )
        self.run_kind = _expect_optional_str(run_kind, "tool context run_kind")
        if self.parent_run_id == "":
            raise ValueError("tool context parent_run_id must not be empty")
        if self.parent_tool_call_id == "":
            raise ValueError("tool context parent_tool_call_id must not be empty")
        if self.run_kind == "":
            raise ValueError("tool context run_kind must not be empty")
        if self.parent_run_id is None and (
            self.parent_tool_call_id is not None or self.run_kind is not None
        ):
            raise ValueError("tool context parent_run_id is required for child run fields")
        self.metadata = _copy_mapping(metadata)
        self._sequence = 0
        self._progress_emitter: ToolProgressEmitter | None = None
        self._cancel_checker: ToolCancelChecker | None = None

    @classmethod
    def from_runtime_context(
        cls,
        context: RuntimeContextSnapshot,
        *,
        progress_emitter: ToolProgressEmitter | None = None,
        cancel_checker: ToolCancelChecker | None = None,
    ) -> ToolExecutionContext:
        data = context.to_dict()
        tool_context = cls(
            run_id=_expect_str(data["run_id"], "tool context run_id"),
            started_at=cast(float, data["started_at"]),
            deadline=cast(float | None, data["deadline"]),
            metadata=_expect_mapping(data["metadata"], "tool context metadata"),
            parent_run_id=_expect_optional_str(
                data.get("parent_run_id"), "tool context parent_run_id"
            ),
            parent_tool_call_id=_expect_optional_str(
                data.get("parent_tool_call_id"), "tool context parent_tool_call_id"
            ),
            run_kind=_expect_optional_str(data.get("run_kind"), "tool context run_kind"),
        )
        tool_context.sequence = cast(int, data["sequence"])
        tool_context._progress_emitter = progress_emitter
        tool_context._cancel_checker = cancel_checker
        return tool_context

    @property
    def sequence(self) -> int:
        return self._sequence

    @sequence.setter
    def sequence(self, value: object) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("tool context sequence must be an integer")
        if value < 0:
            raise ValueError("tool context sequence must be >= 0")
        self._sequence = value

    def remaining_seconds(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "deadline": self.deadline,
            "metadata": _copy_mapping(self.metadata),
            **({} if self.parent_run_id is None else {"parent_run_id": self.parent_run_id}),
            **(
                {}
                if self.parent_tool_call_id is None
                else {"parent_tool_call_id": self.parent_tool_call_id}
            ),
            **({} if self.run_kind is None else {"run_kind": self.run_kind}),
            "sequence": self._sequence,
        }

    def emit_progress(self, data: Mapping[str, Any]) -> None:
        """Emit live, non-durable tool progress for hosts that subscribed to events."""

        emitter = self._progress_emitter
        if emitter is None:
            return
        emitter(_copy_mapping(_expect_mapping(data, "tool progress data")))

    @property
    def cancel_requested(self) -> bool:
        checker = self._cancel_checker
        return False if checker is None else checker()


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


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")
