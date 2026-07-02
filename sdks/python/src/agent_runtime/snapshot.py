"""Serializable run checkpoint protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from agent_runtime.runtime import RuntimeContext
from agent_runtime.state import AgentState


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_agent_state(value: object) -> AgentState:
    if not isinstance(value, AgentState):
        raise TypeError("run snapshot state must be an AgentState")
    return value


def _expect_runtime_context(value: object) -> RuntimeContext:
    if not isinstance(value, RuntimeContext):
        raise TypeError("run snapshot context must be a RuntimeContext")
    return value


@dataclass(slots=True, frozen=True, init=False)
class RunSnapshot:
    """Durable run checkpoint owned by host persistence."""

    _state: AgentState
    _context: RuntimeContext

    def __init__(self, *, state: AgentState, context: RuntimeContext) -> None:
        raw_state = _expect_agent_state(cast(object, state))
        raw_context = _expect_runtime_context(cast(object, context))
        object.__setattr__(self, "_state", AgentState.from_dict(raw_state.to_dict()))
        object.__setattr__(self, "_context", RuntimeContext.from_dict(raw_context.to_dict()))

    @property
    def state(self) -> AgentState:
        return AgentState.from_dict(self._state.to_dict())

    @property
    def context(self) -> RuntimeContext:
        return RuntimeContext.from_dict(self._context.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunSnapshot:
        _reject_unknown_keys(value, {"state", "context"}, "run snapshot")
        return cls(
            state=AgentState.from_dict(
                cast(dict[str, Any], _expect_mapping(value["state"], "run snapshot state"))
            ),
            context=RuntimeContext.from_dict(
                _expect_mapping(value["context"], "run snapshot context")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.to_dict(),
            "context": self._context.to_dict(),
        }
