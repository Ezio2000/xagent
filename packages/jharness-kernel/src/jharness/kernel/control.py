"""Private thread-safe control channel owned by one invocation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import TypeAlias

from jharness.kernel.messages import Message
from jharness.kernel.state import Suspension


@dataclass(frozen=True, slots=True)
class Pause:
    suspension: Suspension


@dataclass(frozen=True, slots=True)
class Insert:
    message: Message
    source: str


@dataclass(frozen=True, slots=True)
class CancelTool:
    call_id: str


Control: TypeAlias = Pause | Insert | CancelTool


class ControlInbox:
    """Event-loop side of the invocation control channel."""

    __slots__ = ("_cancelled", "_closed", "_lock", "_loop", "_queue")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[Control] = asyncio.Queue()
        self._cancelled: set[str] = set()
        self._closed = False
        self._lock = Lock()

    def submit(self, control: Control) -> None:
        with self._lock:
            if self._closed:
                return
        self._loop.call_soon_threadsafe(self._enqueue_if_open, control)

    def submit_local(self, control: Control) -> None:
        with self._lock:
            if self._closed:
                return
            self._queue.put_nowait(control)

    async def next(self) -> Control:
        return await self._queue.get()

    def poll(self) -> Control | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def cancellation_requested(self, call_id: str) -> bool:
        with self._lock:
            return call_id in self._cancelled

    def request_cancellation(self, call_id: str) -> None:
        with self._lock:
            if not self._closed:
                self._cancelled.add(call_id)

    def clear_cancellation(self, call_id: str) -> None:
        with self._lock:
            self._cancelled.discard(call_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancelled.clear()
        while self.poll() is not None:
            pass

    def _enqueue_if_open(self, control: Control) -> None:
        with self._lock:
            if not self._closed:
                self._queue.put_nowait(control)


def drain_pending_controls(inbox: ControlInbox) -> tuple[Pause | None, list[Insert]]:
    """Drain controls observed at a synchronous engine boundary."""

    pause: Pause | None = None
    inserts: list[Insert] = []
    while (control := inbox.poll()) is not None:
        if isinstance(control, Pause) and pause is None:
            pause = control
        elif isinstance(control, Insert):
            inserts.append(control)
        elif isinstance(control, CancelTool):
            inbox.clear_cancellation(control.call_id)
    return pause, inserts


class ControlSource:
    """Thread-safe producer retained by the public Invocation."""

    __slots__ = ("_active", "_closed", "_lock", "_pending")

    def __init__(self) -> None:
        self._lock = Lock()
        self._active: ControlInbox | None = None
        self._closed = False
        self._pending: deque[Control] = deque()

    def submit(self, control: Control) -> None:
        with self._lock:
            if self._closed:
                return
            inbox = self._active
            if inbox is None:
                self._pending.append(control)
                return
            inbox.submit(control)

    def attach(self, loop: asyncio.AbstractEventLoop) -> ControlInbox:
        with self._lock:
            if self._closed:
                raise RuntimeError("invocation has finished; controls are closed")
            if self._active is not None:
                raise RuntimeError("invocation control is already attached")
            inbox = ControlInbox(loop)
            self._active = inbox
            pending = tuple(self._pending)
            self._pending.clear()
        for control in pending:
            inbox.submit_local(control)
        return inbox

    def detach(self, inbox: ControlInbox) -> None:
        detached = False
        with self._lock:
            if self._active is inbox:
                self._active = None
                self._closed = True
                self._pending.clear()
                detached = True
        if detached:
            inbox.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            inbox = self._active
            self._active = None
            self._pending.clear()
        if inbox is not None:
            inbox.close()
