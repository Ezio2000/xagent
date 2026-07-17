"""Deterministic bounded-concurrency smoke benchmark for the public runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter

from jharness.kernel import (
    Completed,
    ContentPart,
    DeltaSink,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    RunLimits,
    Runtime,
    SettledResult,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolResult,
    ToolSpec,
    ToolSuccess,
)
from jharness.toolkit import ToolRegistry

_TOOL_COUNT = 8
_CONCURRENCY = 4
_DELAY_SECONDS = 0.02
_MIN_SPEEDUP = 2.0


@dataclass(slots=True)
class _Tracker:
    active: int = 0
    maximum: int = 0


@dataclass(slots=True)
class _TimedTool:
    spec: ToolSpec
    tracker: _Tracker

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        self.tracker.active += 1
        self.tracker.maximum = max(self.tracker.maximum, self.tracker.active)
        try:
            await asyncio.sleep(_DELAY_SECONDS)
            return SettledResult(ToolSuccess((ContentPart.text_part(call.id),)))
        finally:
            self.tracker.active -= 1


class _BenchmarkModel:
    def __init__(self, calls: tuple[ToolCall, ...]) -> None:
        self._calls = calls
        self._turn = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del request, context, stream, emit_delta
        self._turn += 1
        if self._turn == 1:
            return ModelResponse(tool_calls=self._calls, finish_reason="tool_calls")
        return ModelResponse((ContentPart.text_part("done"),), finish_reason="end_turn")


async def _measure(*, parallel: bool) -> tuple[float, int]:
    tracker = _Tracker()
    concurrency = "parallel" if parallel else "serial"
    execution = ToolExecution(concurrency, read_only=parallel, idempotent=parallel)
    calls = tuple(ToolCall(f"call-{index}", f"tool-{index}") for index in range(_TOOL_COUNT))
    tools = tuple(
        _TimedTool(
            ToolSpec(call.name, "timed benchmark tool", {"type": "object"}, execution=execution),
            tracker,
        )
        for call in calls
    )
    runtime = Runtime(
        model=_BenchmarkModel(calls),
        tools=ToolRegistry(tools),
        limits=RunLimits(max_tool_concurrency=_CONCURRENCY),
    )
    started = perf_counter()
    checkpoint = await runtime.start((Message.user("benchmark"),)).result()
    elapsed = perf_counter() - started
    tool_ids = tuple(
        message.tool_call_id for message in checkpoint.snapshot.history if message.role == "tool"
    )
    if not isinstance(checkpoint.snapshot.state, Completed) or tool_ids != tuple(
        call.id for call in calls
    ):
        raise AssertionError("runtime benchmark produced invalid durable output")
    return elapsed, tracker.maximum


async def main() -> None:
    serial_seconds, serial_maximum = await _measure(parallel=False)
    parallel_seconds, parallel_maximum = await _measure(parallel=True)
    speedup = serial_seconds / parallel_seconds
    if serial_maximum != 1:
        raise AssertionError(f"serial execution reached concurrency {serial_maximum}")
    if parallel_maximum != _CONCURRENCY:
        raise AssertionError(
            f"parallel execution reached {parallel_maximum}, expected {_CONCURRENCY}"
        )
    if speedup < _MIN_SPEEDUP:
        raise AssertionError(f"parallel speedup {speedup:.2f}x is below {_MIN_SPEEDUP:.2f}x")
    print(
        f"serial={serial_seconds:.4f}s parallel={parallel_seconds:.4f}s "
        f"speedup={speedup:.2f}x max_concurrency={parallel_maximum}"
    )


if __name__ == "__main__":
    asyncio.run(main())
