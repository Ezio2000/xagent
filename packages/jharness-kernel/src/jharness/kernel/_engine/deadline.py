"""Monotonic invocation deadline and interruptible effect await."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import time
from typing import Any, TypeVar, cast

from jharness.kernel.control import CancelTool, Control, ControlInbox, Insert

T = TypeVar("T")


class WorkDeadlineReached(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EffectInterrupted(Exception):
    control: Control


@dataclass(frozen=True, slots=True)
class Deadline:
    """Monotonic deadline derived once from a portable wall-clock deadline."""

    at: float | None

    @classmethod
    def from_wall_time(
        cls, wall_deadline: float | None, loop: asyncio.AbstractEventLoop
    ) -> Deadline:
        if wall_deadline is None:
            return cls(None)
        return cls(loop.time() + max(0.0, wall_deadline - time()))

    def remaining(self) -> float | None:
        if self.at is None:
            return None
        return max(0.0, self.at - asyncio.get_running_loop().time())

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0


async def await_effect(
    awaitable: Awaitable[T],
    *,
    deadline: Deadline,
    inbox: ControlInbox,
    defer_insert: Callable[[Insert], None] | None = None,
) -> T:
    """Await an effect while pause/insertion and the work deadline may preempt it."""

    effect = asyncio.ensure_future(awaitable)
    control = asyncio.create_task(inbox.next())
    try:
        while True:
            remaining = deadline.remaining()
            if remaining is not None and remaining <= 0:
                raise WorkDeadlineReached
            done, _ = await asyncio.wait(
                {effect, control}, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if effect in done:
                if control in done:
                    command = control.result()
                    if not _consume_non_interrupting(command, inbox, defer_insert):
                        inbox.submit_local(command)
                return await effect
            if control in done:
                command = control.result()
                if _consume_non_interrupting(command, inbox, defer_insert):
                    control = asyncio.create_task(inbox.next())
                    continue
                raise EffectInterrupted(command)
            raise WorkDeadlineReached
    finally:
        control.cancel()
        if not effect.done():
            effect.cancel()
        await _settle(control)
        await _settle(effect)


def _consume_non_interrupting(
    control: Control,
    inbox: ControlInbox,
    defer_insert: Callable[[Insert], None] | None,
) -> bool:
    if isinstance(control, CancelTool):
        inbox.clear_cancellation(control.call_id)
        return True
    if isinstance(control, Insert) and defer_insert is not None:
        defer_insert(control)
        return True
    return False


async def _settle(task: asyncio.Future[Any]) -> None:
    try:
        async with asyncio.timeout(0.1):
            await cast(Awaitable[object], task)
    except (asyncio.CancelledError, Exception):
        pass
