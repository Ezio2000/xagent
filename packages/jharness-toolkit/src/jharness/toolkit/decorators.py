"""Composable tool execution policies outside kernel state-machine semantics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Lock

from jharness.kernel import SettledResult, ToolCall, ToolContext, ToolFailure, ToolResult, ToolSpec
from jharness.toolkit.tool import Tool


@dataclass(frozen=True, slots=True)
class RetryingTool:
    """Retry implementation exceptions for an explicitly idempotent tool."""

    tool: Tool
    max_attempts: int = 2
    attempt_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_attempts > 1 and not self.spec.execution.idempotent:
            raise ValueError("retrying tool must be idempotent")
        if self.attempt_timeout_seconds is not None and self.attempt_timeout_seconds <= 0:
            raise ValueError("attempt_timeout_seconds must be > 0")

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        attempt = 1
        while True:
            try:
                async with asyncio.timeout(self.attempt_timeout_seconds):
                    return await self.tool.invoke(call, context)
            except Exception:
                if attempt == self.max_attempts:
                    raise
                attempt += 1
                await asyncio.sleep(0)


@dataclass(slots=True)
class CircuitBreakingTool:
    """Reject calls after consecutive implementation exceptions."""

    tool: Tool
    failure_threshold: int = 3
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    _failures: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        with self._lock:
            if self._failures >= self.failure_threshold:
                return SettledResult(ToolFailure.from_error("circuit_open", "tool circuit is open"))
        try:
            result = await self.tool.invoke(call, context)
        except Exception:
            with self._lock:
                self._failures += 1
            raise
        with self._lock:
            self._failures = 0
        return result
