"""Portable run budgets and bounded-concurrency controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from jharness.kernel._validation import (
    expect_int,
    expect_optional_int,
    expect_optional_number,
)


class LimitReason(StrEnum):
    """Closed portable limit outcomes."""

    MAX_PLANNING_STEPS = "max_planning_steps"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOTAL_TOKENS = "max_total_tokens"
    DEADLINE = "deadline"


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Run-level limits; logical retry policies live outside the kernel."""

    max_planning_steps: int = 20
    max_tool_calls: int = 20
    max_total_tokens: int | None = None
    timeout_seconds: float | None = None
    max_tool_concurrency: int = 4
    max_tool_batch_size: int = 20
    max_buffered_progress: int = 1024

    def __post_init__(self) -> None:
        max_planning_steps = expect_int(self.max_planning_steps, "max_planning_steps")
        max_tool_calls = expect_int(self.max_tool_calls, "max_tool_calls")
        max_total_tokens = expect_optional_int(self.max_total_tokens, "max_total_tokens")
        timeout_seconds = expect_optional_number(self.timeout_seconds, "timeout_seconds")
        max_tool_concurrency = expect_int(self.max_tool_concurrency, "max_tool_concurrency")
        max_tool_batch_size = expect_int(self.max_tool_batch_size, "max_tool_batch_size")
        max_buffered_progress = expect_int(self.max_buffered_progress, "max_buffered_progress")
        if max_planning_steps < 1:
            raise ValueError("max_planning_steps must be >= 1")
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")
        if max_total_tokens is not None and max_total_tokens < 1:
            raise ValueError("max_total_tokens must be >= 1")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_tool_concurrency < 1:
            raise ValueError("max_tool_concurrency must be >= 1")
        if max_tool_batch_size < 1:
            raise ValueError("max_tool_batch_size must be >= 1")
        if max_buffered_progress < 1:
            raise ValueError("max_buffered_progress must be >= 1")
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
