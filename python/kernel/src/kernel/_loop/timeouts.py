"""Timeout, interrupt, and async iterator helpers for AgentLoop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from contextlib import aclosing, suppress
from inspect import isawaitable
from typing import Any, TypeVar, cast

from kernel._loop.types import (
    RunControlState,
    RuntimeConversationInsert,
    RuntimePauseInterrupt,
    RuntimeTimeoutError,
)
from kernel.control import ConversationInsert
from kernel.events import AgentEvent

T = TypeVar("T")


class TimeoutMixin:
    async def _await_with_timeout(self, awaitable: Awaitable[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(awaitable)
        if remaining is None:
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise
            return await task

        if remaining <= 0:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError

        try:
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        if not done:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError
        return await task

    async def _await_model_with_interrupt(
        self, awaitable: Awaitable[T], control: RunControlState
    ) -> T:
        controller = control.run_controller
        if controller is None:
            return await self._await_with_timeout(awaitable, control)

        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(awaitable)
        pending_insert = controller.pop_insert()
        if pending_insert is not None:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeConversationInsert(pending_insert)
        interrupt_task = asyncio.ensure_future(controller.wait_for_interrupt_or_insert())
        try:
            if remaining is None:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
                )
            elif remaining <= 0:
                raise RuntimeTimeoutError
            else:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise RuntimeTimeoutError

            if interrupt_task in done:
                request = await interrupt_task
                if isinstance(request, ConversationInsert):
                    task.add_done_callback(self._consume_background_task_exception)
                    task.cancel()
                    raise RuntimeConversationInsert(request)
                if task in done:
                    return await task
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                raise RuntimePauseInterrupt(request)

            if task in done:
                interrupt_task.cancel()
                return await task

            raise RuntimeError("model wait returned without a completed task")
        except RuntimeTimeoutError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        except asyncio.CancelledError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise
        finally:
            if not interrupt_task.done():
                interrupt_task.cancel()

    async def _anext_with_timeout(self, iterator: AsyncIterator[T], control: RunControlState) -> T:
        remaining = control.remaining_seconds()
        task = asyncio.ensure_future(anext(iterator))
        if remaining is None:
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                await self._close_async_iterator(iterator)
                raise
            return await task

        if remaining <= 0:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            raise RuntimeTimeoutError

        try:
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        if not done:
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise RuntimeTimeoutError
        return await task

    async def _anext_model_with_interrupt(
        self, iterator: AsyncIterator[T], control: RunControlState
    ) -> T:
        controller = control.run_controller
        if controller is None:
            return await self._anext_with_timeout(iterator, control)

        remaining = control.remaining_seconds()
        pending_insert = controller.pop_insert()
        if pending_insert is not None:
            await self._close_async_iterator(iterator)
            raise RuntimeConversationInsert(pending_insert)
        pending_request = controller.pause_request
        if pending_request is not None and pending_request.interrupt:
            await self._close_async_iterator(iterator)
            raise RuntimePauseInterrupt(pending_request)
        task = asyncio.ensure_future(anext(iterator))
        interrupt_task = asyncio.ensure_future(controller.wait_for_interrupt_or_insert())
        try:
            if remaining is None:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task}, return_when=asyncio.FIRST_COMPLETED
                )
            elif remaining <= 0:
                raise RuntimeTimeoutError
            else:
                done, _pending = await asyncio.wait(
                    {task, interrupt_task},
                    timeout=remaining,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise RuntimeTimeoutError

            if interrupt_task in done:
                request = await interrupt_task
                task.add_done_callback(self._consume_background_task_exception)
                task.cancel()
                await self._close_async_iterator(iterator)
                if isinstance(request, ConversationInsert):
                    raise RuntimeConversationInsert(request)
                raise RuntimePauseInterrupt(request)

            if task in done:
                interrupt_task.cancel()
                return await task

            raise RuntimeError("model stream wait returned without a completed task")
        except RuntimeTimeoutError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        except asyncio.CancelledError:
            interrupt_task.cancel()
            task.add_done_callback(self._consume_background_task_exception)
            task.cancel()
            await self._close_async_iterator(iterator)
            raise
        finally:
            if not interrupt_task.done():
                interrupt_task.cancel()

    async def _pump_events(
        self, iterator: AsyncIterator[AgentEvent]
    ) -> AsyncGenerator[AgentEvent, None]:
        try:
            while True:
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    break
                yield event
        finally:
            await self._close_async_iterator(iterator)

    async def _drain_events(self, iterator: AsyncIterator[AgentEvent]) -> None:
        async with aclosing(self._pump_events(iterator)) as events:
            async for _event in events:
                pass

    async def _close_async_iterator(self, iterator: AsyncIterator[object]) -> None:
        aclose = getattr(iterator, "aclose", None)
        if callable(aclose):
            with suppress(Exception):
                close_result = aclose()
                if isawaitable(close_result):
                    await cast(Awaitable[object], close_result)

    @staticmethod
    def _consume_background_task_exception(task: asyncio.Future[Any]) -> None:
        with suppress(BaseException):
            task.exception()
