"""Loop limit configuration."""

from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.models import ModelUsage
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


def _expect_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _expect_int(value, label)


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


class LimitReasons:
    """Known limit reason constants emitted by the runtime."""

    MAX_ITERATIONS = "max_iterations"
    MAX_TOTAL_TOOL_CALLS = "max_total_tool_calls"
    MAX_TOTAL_TOKENS = "max_total_tokens"
    TIMEOUT_SECONDS = "timeout_seconds"


@dataclass(slots=True, frozen=True)
class LoopLimits:
    """Resource limits for a single agent run."""

    max_iterations: int = 8
    max_total_tool_calls: int = 20
    timeout_seconds: float | None = None
    stop_on_tool_error: bool = False
    max_parallel_tool_calls: int = 1
    max_total_tokens: int | None = None
    max_model_retries: int = 0

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
        max_total_tokens = _expect_optional_int(self.max_total_tokens, "max_total_tokens")
        max_model_retries = _expect_int(self.max_model_retries, "max_model_retries")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_total_tool_calls < 0:
            raise ValueError("max_total_tool_calls must be >= 0")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_parallel_tool_calls < 1:
            raise ValueError("max_parallel_tool_calls must be >= 1")
        if max_total_tokens is not None and max_total_tokens < 0:
            raise ValueError("max_total_tokens must be >= 0")
        if max_model_retries < 0:
            raise ValueError("max_model_retries must be >= 0")

    def iteration_reason(self, state: AgentState) -> str | None:
        if state.iterations >= self.max_iterations:
            return LimitReasons.MAX_ITERATIONS
        return None

    def tool_call_reason(self, state: AgentState) -> str | None:
        if state.total_tool_calls >= self.max_total_tool_calls:
            return LimitReasons.MAX_TOTAL_TOOL_CALLS
        return None

    def usage_reason(self, usage: ModelUsage | None) -> str | None:
        if self.max_total_tokens is None or usage is None or usage.total_tokens is None:
            return None
        if usage.total_tokens > self.max_total_tokens:
            return LimitReasons.MAX_TOTAL_TOKENS
        return None
