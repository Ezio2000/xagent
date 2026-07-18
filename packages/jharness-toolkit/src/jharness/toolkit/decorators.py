"""Composable tool execution policies outside kernel state-machine semantics."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from math import isfinite
from random import uniform
from threading import Lock
from time import monotonic
from typing import cast

from jharness.kernel import SettledResult, ToolCall, ToolContext, ToolFailure, ToolResult, ToolSpec
from jharness.toolkit.tool import Tool

_CANCEL_POLL_SECONDS = 0.05


class RetryExhaustedError(Exception):
    """All configured attempts failed with retryable implementation exceptions."""

    __slots__ = ("errors",)

    def __init__(self, errors: tuple[Exception, ...]) -> None:
        if not errors:
            raise ValueError("retry exhaustion requires at least one error")
        self.errors = errors
        super().__init__(
            f"tool retry exhausted after {len(errors)} attempt(s): "
            f"{str(errors[-1]) or errors[-1].__class__.__name__}"
        )

    @property
    def attempts(self) -> int:
        return len(self.errors)

    @property
    def first_error(self) -> Exception:
        return self.errors[0]

    @property
    def last_error(self) -> Exception:
        return self.errors[-1]


@dataclass(frozen=True, slots=True)
class RetryingTool:
    """Retry selected implementation exceptions for an explicitly idempotent tool."""

    tool: Tool
    max_attempts: int = 2
    attempt_timeout_seconds: float | None = None
    retryable_exceptions: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)
    backoff_initial_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 5.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        _positive_int(self.max_attempts, "max_attempts")
        if self.max_attempts > 1 and not self.spec.execution.idempotent:
            raise ValueError("retrying tool must be idempotent")
        if self.attempt_timeout_seconds is not None:
            _positive_float(self.attempt_timeout_seconds, "attempt_timeout_seconds")
        raw_exceptions = cast(object, self.retryable_exceptions)
        if not isinstance(raw_exceptions, tuple) or any(
            not isinstance(error, type) or not issubclass(error, Exception)
            for error in cast(tuple[object, ...], raw_exceptions)
        ):
            raise TypeError("retryable_exceptions must be a tuple of Exception classes")
        _positive_float(self.backoff_initial_seconds, "backoff_initial_seconds")
        multiplier = _finite_float(self.backoff_multiplier, "backoff_multiplier")
        if multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")
        maximum = _positive_float(self.backoff_max_seconds, "backoff_max_seconds")
        if maximum < self.backoff_initial_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_initial_seconds")
        jitter = _finite_float(self.jitter_ratio, "jitter_ratio")
        if not 0 <= jitter <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        errors: list[Exception] = []
        backoff = self.backoff_initial_seconds
        for attempt in range(1, self.max_attempts + 1):
            if context.cancel_requested:
                return _cancelled()
            try:
                async with asyncio.timeout(self.attempt_timeout_seconds):
                    return await self.tool.invoke(call, context)
            except self.retryable_exceptions as exc:
                errors.append(exc)
                if attempt == self.max_attempts:
                    exhausted = RetryExhaustedError(tuple(errors))
                    raise exhausted from exc
                delay = _jittered_delay(
                    backoff,
                    self.jitter_ratio,
                    self.backoff_max_seconds,
                )
                if not await _cooperative_sleep(delay, context):
                    return _cancelled()
                backoff = min(self.backoff_max_seconds, backoff * self.backoff_multiplier)
        raise RuntimeError("retry loop ended without a result")


@dataclass(slots=True)
class CircuitBreakingTool:
    """Reject calls after failures and admit one recovery probe after a timeout."""

    tool: Tool
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    _failures: int = field(default=0, init=False, repr=False, compare=False)
    _opened_at: float | None = field(default=None, init=False, repr=False, compare=False)
    _probe_in_flight: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _positive_int(self.failure_threshold, "failure_threshold")
        _positive_float(self.recovery_timeout_seconds, "recovery_timeout_seconds")

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        probe = self._admit()
        if probe is None:
            return SettledResult(ToolFailure.from_error("circuit_open", "tool circuit is open"))
        try:
            result = await self.tool.invoke(call, context)
        except asyncio.CancelledError:
            self._release_cancelled_probe(probe)
            raise
        except Exception:
            self._record_failure(probe)
            raise
        except BaseException:
            self._release_cancelled_probe(probe)
            raise
        self._record_success()
        return result

    def _admit(self) -> bool | None:
        now = monotonic()
        with self._lock:
            if self._opened_at is None:
                return False
            if now - self._opened_at < self.recovery_timeout_seconds or self._probe_in_flight:
                return None
            self._probe_in_flight = True
            return True

    def _record_failure(self, probe: bool) -> None:
        now = monotonic()
        with self._lock:
            if probe:
                self._probe_in_flight = False
                self._failures = self.failure_threshold
                self._opened_at = now
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = now

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def _release_cancelled_probe(self, probe: bool) -> None:
        if not probe:
            return
        with self._lock:
            self._probe_in_flight = False


async def _cooperative_sleep(delay: float, context: ToolContext) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + delay
    while True:
        if context.cancel_requested:
            return False
        remaining = deadline - loop.time()
        if remaining <= 0:
            return True
        await asyncio.sleep(min(_CANCEL_POLL_SECONDS, remaining))


def _jittered_delay(delay: float, ratio: float, maximum: float) -> float:
    spread = delay * ratio
    return min(maximum, max(0.0, uniform(delay - spread, delay + spread)))


def _cancelled() -> SettledResult:
    return SettledResult(ToolFailure.from_error("cancelled", "tool retry was cancelled"))


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_float(value: object, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _finite_float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)
