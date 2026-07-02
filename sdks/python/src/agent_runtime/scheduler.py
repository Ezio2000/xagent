"""Tool-call scheduling primitives."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from agent_runtime.messages import ToolCall
from agent_runtime.tools import ToolRegistry, ToolResult, ToolSpec


@dataclass(slots=True, frozen=True)
class ToolBatch:
    """A consecutive group of tool calls with the same scheduling mode."""

    id: str
    calls: tuple[ToolCall, ...]
    parallel: bool


@dataclass(slots=True, frozen=True)
class ToolStarted:
    """A scheduled tool call has started execution."""

    batch: ToolBatch
    index: int
    call: ToolCall


@dataclass(slots=True, frozen=True)
class ToolCompleted:
    """A scheduled tool call has completed execution."""

    batch: ToolBatch
    index: int
    call: ToolCall
    result: ToolResult


ToolProgress: TypeAlias = ToolStarted | ToolCompleted
ExecuteTool: TypeAlias = Callable[[ToolCall], Awaitable[ToolResult]]


class ToolScheduler:
    """Build and execute simple ordered tool batches."""

    __slots__ = ("_batch_counter", "_max_parallel_tool_calls", "_tools")

    _batch_counter: int
    _max_parallel_tool_calls: int
    _tools: ToolRegistry

    def __init__(self, tools: ToolRegistry, *, max_parallel_tool_calls: int = 1) -> None:
        if max_parallel_tool_calls < 1:
            raise ValueError("max_parallel_tool_calls must be >= 1")
        self._tools = tools
        self._max_parallel_tool_calls = max_parallel_tool_calls
        self._batch_counter = 0

    def next_batch(self, calls: tuple[ToolCall, ...]) -> ToolBatch | None:
        if not calls:
            return None

        self._batch_counter += 1
        batch_id = f"tool-batch-{self._batch_counter}"
        first = calls[0]
        if not self._can_parallelize(first):
            return ToolBatch(batch_id, (first,), parallel=False)

        batch_calls: list[ToolCall] = []
        for call in calls:
            if not self._can_parallelize(call):
                break
            batch_calls.append(call)
        return ToolBatch(batch_id, tuple(batch_calls), parallel=len(batch_calls) > 1)

    async def run_batch(
        self,
        batch: ToolBatch,
        execute: ExecuteTool,
        *,
        stop_on_error: bool = False,
    ) -> AsyncIterator[ToolProgress]:
        if not batch.calls:
            return

        max_active = self._max_parallel_tool_calls if batch.parallel else 1
        next_index = 0
        active: dict[asyncio.Future[ToolResult], tuple[int, ToolCall]] = {}

        def start_next() -> ToolStarted | None:
            nonlocal next_index
            if next_index >= len(batch.calls):
                return None
            index = next_index
            call = batch.calls[index]
            next_index += 1
            active[asyncio.ensure_future(execute(call))] = (index, call)
            return ToolStarted(batch=batch, index=index, call=call)

        try:
            while len(active) < max_active:
                started = start_next()
                if started is None:
                    break
                yield started

            while active:
                done, _pending = await asyncio.wait(
                    set(active.keys()), return_when=asyncio.FIRST_COMPLETED
                )
                done_tasks = tuple(sorted(done, key=lambda task: active[task][0]))
                for task in done_tasks:
                    index, call = active.pop(task)
                    result = await task
                    yield ToolCompleted(batch=batch, index=index, call=call, result=result)
                    if stop_on_error and result.is_error:
                        for remaining in active:
                            remaining.cancel()
                        if active:
                            await asyncio.gather(*active.keys(), return_exceptions=True)
                            active.clear()
                        return

                    while len(active) < max_active:
                        started = start_next()
                        if started is None:
                            break
                        yield started
        except BaseException:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active.keys(), return_exceptions=True)
            raise

    def _can_parallelize(self, call: ToolCall) -> bool:
        if self._max_parallel_tool_calls <= 1:
            return False
        spec = self._tools.spec_for(call.name)
        if spec is None:
            return False
        return _parallel_safe(spec)


def _parallel_safe(spec: ToolSpec) -> bool:
    annotations = spec.annotations
    return (
        annotations.get("parallel_safe") is True
        and annotations.get("read_only") is True
        and annotations.get("idempotent") is True
    )
