"""Tool approval protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from agent_runtime._frozen import freeze_value, thaw_value
from agent_runtime.messages import ToolCall
from agent_runtime.runtime import RuntimeContext
from agent_runtime.tools import ToolSpec

ApprovalAction = Literal["allow", "deny", "pause"]


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


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _freeze_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return cast(
        Mapping[str, Any],
        freeze_value(_expect_mapping(value, label), error_message=f"{label} is immutable"),
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], thaw_value(value))


@dataclass(slots=True, frozen=True, init=False)
class ApprovalRequest:
    """Tool execution approval request passed to host policy."""

    _tool_call: ToolCall
    _context: RuntimeContext
    _tool_spec: ToolSpec | None
    risk: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        *,
        tool_call: ToolCall,
        context: RuntimeContext,
        tool_spec: ToolSpec | None = None,
        risk: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(cast(object, tool_call), ToolCall):
            raise TypeError("approval request tool_call must be a ToolCall")
        if not isinstance(cast(object, context), RuntimeContext):
            raise TypeError("approval request context must be a RuntimeContext")
        if tool_spec is not None and not isinstance(cast(object, tool_spec), ToolSpec):
            raise TypeError("approval request tool_spec must be a ToolSpec or None")
        object.__setattr__(
            self,
            "_tool_call",
            ToolCall.from_dict(tool_call.to_dict()),
        )
        object.__setattr__(
            self,
            "_context",
            RuntimeContext.from_dict(context.to_dict()),
        )
        object.__setattr__(
            self,
            "_tool_spec",
            None if tool_spec is None else ToolSpec.from_dict(tool_spec.to_dict()),
        )
        object.__setattr__(
            self,
            "risk",
            _freeze_mapping({} if risk is None else risk, "approval request risk"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping({} if metadata is None else metadata, "approval request metadata"),
        )

    @property
    def tool_call(self) -> ToolCall:
        return ToolCall.from_dict(self._tool_call.to_dict())

    @property
    def context(self) -> RuntimeContext:
        return RuntimeContext.from_dict(self._context.to_dict())

    @property
    def tool_spec(self) -> ToolSpec | None:
        return None if self._tool_spec is None else ToolSpec.from_dict(self._tool_spec.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ApprovalRequest:
        _reject_unknown_keys(
            value,
            {"tool_call", "context", "tool_spec", "risk", "metadata"},
            "approval request",
        )
        raw_tool_spec = value["tool_spec"]
        if raw_tool_spec is not None:
            raw_tool_spec = _expect_mapping(raw_tool_spec, "approval request tool_spec")
        return cls(
            tool_call=ToolCall.from_dict(
                _expect_mapping(value["tool_call"], "approval request tool_call")
            ),
            context=RuntimeContext.from_dict(
                _expect_mapping(value["context"], "approval request context")
            ),
            tool_spec=None if raw_tool_spec is None else ToolSpec.from_dict(raw_tool_spec),
            risk=_expect_mapping(value["risk"], "approval request risk"),
            metadata=_expect_mapping(value["metadata"], "approval request metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self._tool_call.to_dict(),
            "context": self._context.to_dict(),
            "tool_spec": None if self._tool_spec is None else self._tool_spec.to_dict(),
            "risk": _copy_mapping(self.risk),
            "metadata": _copy_mapping(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class ApprovalDecision:
    """Host decision for a pending tool invocation."""

    action: ApprovalAction = "allow"
    reason: str = "approved"
    metadata: Mapping[str, Any] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        action = _expect_str(self.action, "approval action")
        if action not in {"allow", "deny", "pause"}:
            raise ValueError("approval action must be allow, deny, or pause")
        reason = _expect_str(self.reason, "approval reason")
        if not reason:
            raise ValueError("approval reason must not be empty")
        object.__setattr__(self, "action", cast(ApprovalAction, action))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "approval decision metadata"),
        )

    @classmethod
    def allow(
        cls, reason: str = "approved", *, metadata: Mapping[str, Any] | None = None
    ) -> ApprovalDecision:
        return cls("allow", reason, {} if metadata is None else metadata)

    @classmethod
    def deny(
        cls, reason: str = "denied", *, metadata: Mapping[str, Any] | None = None
    ) -> ApprovalDecision:
        return cls("deny", reason, {} if metadata is None else metadata)

    @classmethod
    def pause(
        cls, reason: str = "tool_approval", *, metadata: Mapping[str, Any] | None = None
    ) -> ApprovalDecision:
        return cls("pause", reason, {} if metadata is None else metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ApprovalDecision:
        _reject_unknown_keys(value, {"action", "reason", "metadata"}, "approval decision")
        return cls(
            action=cast(ApprovalAction, _expect_str(value["action"], "approval action")),
            reason=_expect_str(value["reason"], "approval reason"),
            metadata=_expect_mapping(value["metadata"], "approval metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "metadata": _copy_mapping(self.metadata),
        }


class ApprovalPolicy(Protocol):
    """Host-owned policy that approves, denies, or pauses tool execution."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """Return the decision for one tool invocation."""
        ...
