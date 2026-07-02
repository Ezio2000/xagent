"""Loop limit configuration."""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.state import AgentState


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def _expect_optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number or null")
    return float(value)


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


@dataclass(slots=True, frozen=True)
class LoopLimits:
    """Resource limits for a single agent run."""

    max_iterations: int = 8
    max_total_tool_calls: int = 20
    timeout_seconds: float | None = None
    stop_on_tool_error: bool = False
    max_parallel_tool_calls: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        max_iterations = _expect_int(self.max_iterations, "max_iterations")
        max_total_tool_calls = _expect_int(self.max_total_tool_calls, "max_total_tool_calls")
        timeout_seconds = _expect_optional_number(self.timeout_seconds, "timeout_seconds")
        _expect_bool(self.stop_on_tool_error, "stop_on_tool_error")
        max_parallel_tool_calls = _expect_int(
            self.max_parallel_tool_calls, "max_parallel_tool_calls"
        )
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_total_tool_calls < 0:
            raise ValueError("max_total_tool_calls must be >= 0")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_parallel_tool_calls < 1:
            raise ValueError("max_parallel_tool_calls must be >= 1")

    def iteration_reason(self, state: AgentState) -> str | None:
        if state.iterations >= self.max_iterations:
            return "max_iterations"
        return None

    def tool_call_reason(self, state: AgentState) -> str | None:
        if state.total_tool_calls >= self.max_total_tool_calls:
            return "max_total_tool_calls"
        return None
