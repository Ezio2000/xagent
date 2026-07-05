"""Loop limit configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kernel._validation import (
    expect_bool as _expect_bool,
)
from kernel._validation import (
    expect_int as _expect_int,
)
from kernel._validation import (
    expect_optional_int as _expect_optional_int,
)
from kernel._validation import (
    expect_optional_number as _expect_optional_number,
)
from kernel._validation import (
    reject_unknown_keys as _reject_unknown_keys,
)
from kernel.models import ModelUsage
from kernel.state import AgentState


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LoopLimits:
        known = {
            "max_iterations",
            "max_total_tool_calls",
            "timeout_seconds",
            "stop_on_tool_error",
            "max_parallel_tool_calls",
            "max_total_tokens",
            "max_model_retries",
        }
        _reject_unknown_keys(value, known, "loop limits")
        return cls(
            max_iterations=_expect_int(value.get("max_iterations", 8), "max_iterations"),
            max_total_tool_calls=_expect_int(
                value.get("max_total_tool_calls", 20), "max_total_tool_calls"
            ),
            timeout_seconds=_expect_optional_number(
                value.get("timeout_seconds"), "timeout_seconds"
            ),
            stop_on_tool_error=_expect_bool(
                value.get("stop_on_tool_error", False), "stop_on_tool_error"
            ),
            max_parallel_tool_calls=_expect_int(
                value.get("max_parallel_tool_calls", 1), "max_parallel_tool_calls"
            ),
            max_total_tokens=_expect_optional_int(
                value.get("max_total_tokens"), "max_total_tokens"
            ),
            max_model_retries=_expect_int(value.get("max_model_retries", 0), "max_model_retries"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "max_iterations": self.max_iterations,
            "max_total_tool_calls": self.max_total_tool_calls,
            "stop_on_tool_error": self.stop_on_tool_error,
            "max_parallel_tool_calls": self.max_parallel_tool_calls,
            "max_model_retries": self.max_model_retries,
        }
        if self.timeout_seconds is not None:
            data["timeout_seconds"] = self.timeout_seconds
        if self.max_total_tokens is not None:
            data["max_total_tokens"] = self.max_total_tokens
        return data

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
