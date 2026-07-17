"""Single-execution Invocation API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from time import time
from typing import Any, TypeAlias, cast
from uuid import uuid4

from jharness.kernel._validation import expect_instance, expect_mapping, expect_non_empty_str
from jharness.kernel.checkpoint import Checkpoint
from jharness.kernel.control import CancelTool, ControlSource, Insert, Pause
from jharness.kernel.events import Event, EventKind
from jharness.kernel.messages import Message
from jharness.kernel.state import Suspension

Emit: TypeAlias = Callable[[EventKind, Mapping[str, Any]], Awaitable[None]]
Execute: TypeAlias = Callable[[Emit, ControlSource], Awaitable[Checkpoint]]

_DONE = object()
_LOSSY = frozenset({EventKind.MODEL_DELTA, EventKind.TOOL_PROGRESS})


async def _ignore_event(kind: EventKind, data: Mapping[str, Any]) -> None:
    del kind, data


class Invocation:
    """One start, continue, or resume execution with optional observation."""

    __slots__ = (
        "_control",
        "_execute",
        "_invocation_id",
        "_lossy_limit",
        "_lossy_queued",
        "_mode",
        "_queue",
        "_run_id",
        "_sequence",
        "_task",
        "stream",
    )

    def __init__(
        self,
        run_id: str,
        execute: Execute,
        *,
        stream: bool,
        max_buffered_events: int = 1024,
    ) -> None:
        expect_non_empty_str(run_id, "invocation run_id")
        if not callable(execute):
            raise TypeError("invocation execute must be callable")
        if not isinstance(cast(object, stream), bool):
            raise TypeError("invocation stream must be bool")
        raw_buffer_size = cast(object, max_buffered_events)
        if isinstance(raw_buffer_size, bool) or not isinstance(raw_buffer_size, int):
            raise TypeError("max_buffered_events must be an integer")
        if max_buffered_events < 1:
            raise ValueError("max_buffered_events must be >= 1")
        self._run_id = run_id
        self._execute: Execute | None = execute
        self._invocation_id = str(uuid4())
        self._control = ControlSource()
        self._queue: asyncio.Queue[Event | object] | None = None
        self._lossy_limit = max_buffered_events
        self._lossy_queued = 0
        self._sequence = 0
        self._mode: str | None = None
        self._task: asyncio.Task[Checkpoint] | None = None
        self.stream = stream

    stream: bool

    @property
    def id(self) -> str:
        return self._invocation_id

    def events(self) -> AsyncGenerator[Event, None]:
        """Select observation mode and return the invocation's only event iterator."""

        if self._mode == "result":
            raise RuntimeError("result-only invocation cannot be observed")
        if self._mode == "events":
            raise RuntimeError("invocation events can be consumed only once")
        self._mode = "events"
        self._queue = asyncio.Queue()
        return self._iterate_events()

    async def result(self) -> Checkpoint:
        """Await the same execution and return its last committed checkpoint."""

        if self._mode is None:
            self._mode = "result"
        task = self._ensure_started()
        return await asyncio.shield(task)

    def pause(self, suspension: Suspension) -> None:
        self._control.submit(Pause(expect_instance(suspension, Suspension, "suspension")))

    def insert(self, message: Message, *, source: str = "host") -> None:
        message = expect_instance(message, Message, "insert message")
        if message.role != "external":
            raise ValueError("conversation insertion requires an external message")
        expect_non_empty_str(source, "insert source")
        self._control.submit(Insert(message, source))

    def cancel_tool(self, call_id: str) -> None:
        expect_non_empty_str(call_id, "tool call id")
        self._control.submit(CancelTool(call_id))

    async def _iterate_events(self) -> AsyncGenerator[Event, None]:
        task = self._ensure_started()
        queue = self._queue
        if queue is None:
            raise RuntimeError("invocation event queue is not initialized")
        completed = False
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    completed = True
                    break
                event = cast(Event, item)
                if event.kind in _LOSSY:
                    self._lossy_queued -= 1
                yield event
            await task
        finally:
            if not completed and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self._queue = None
            self._lossy_queued = 0

    def _ensure_started(self) -> asyncio.Task[Checkpoint]:
        if self._task is None:
            self._task = asyncio.create_task(self._drive())
        return self._task

    async def _drive(self) -> Checkpoint:
        emit = self._emit if self._mode == "events" else _ignore_event
        execute = self._execute
        if execute is None:
            raise RuntimeError("invocation execution is no longer available")
        try:
            return await execute(emit, self._control)
        finally:
            self._execute = None
            self._control.close()
            if self._mode == "events":
                queue = self._queue
                if queue is None:
                    raise RuntimeError("invocation event queue is not initialized")
                queue.put_nowait(_DONE)

    async def _emit(self, kind: EventKind, data: Mapping[str, Any]) -> None:
        kind = expect_instance(kind, EventKind, "event kind")
        data = expect_mapping(data, "event data")
        queue = self._queue
        if queue is None:
            raise RuntimeError("invocation event queue is not initialized")
        if kind in _LOSSY and self._lossy_queued >= self._lossy_limit:
            return
        self._sequence += 1
        if kind in _LOSSY:
            self._lossy_queued += 1
        queue.put_nowait(
            Event(
                run_id=self._run_id,
                invocation_id=self._invocation_id,
                sequence=self._sequence,
                kind=kind,
                created_at=time(),
                data=data,
            )
        )
