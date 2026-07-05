"""Runtime hook dispatch helpers for AgentLoop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from inspect import isawaitable, iscoroutinefunction
from typing import Any, TypeVar, cast

from kernel._loop.helpers import normalize_tool_output
from kernel._loop.types import RunControlState
from kernel.context import RuntimeContext
from kernel.errors import ModelProviderError
from kernel.hooks import ModelErrorDecision, RuntimeHook
from kernel.messages import ToolCall
from kernel.models import ModelRequest, ModelResponse
from kernel.tools import ToolOutput

T = TypeVar("T")


class HookMixin:
    _hooks: tuple[RuntimeHook, ...]

    async def _await_with_timeout(self, awaitable: Awaitable[T], control: RunControlState) -> T:
        raise NotImplementedError

    async def _before_model(
        self, request: ModelRequest, context: RuntimeContext, control: RunControlState
    ) -> ModelRequest:
        current = request
        for hook in self._hooks:
            method = self._hook_method(hook, "before_model")
            if method is None:
                continue
            replacement = await self._call_hook(method, current, context, control=control)
            if replacement is not None:
                current = cast(ModelRequest, replacement)
            current = ModelRequest.from_dict(current.to_dict())
        return ModelRequest.from_dict(current.to_dict())

    async def _after_model(
        self, response: ModelResponse, context: RuntimeContext, control: RunControlState
    ) -> ModelResponse:
        current = response
        for hook in self._hooks:
            method = self._hook_method(hook, "after_model")
            if method is None:
                continue
            replacement = await self._call_hook(method, current, context, control=control)
            if replacement is not None:
                current = cast(ModelResponse, replacement)
            current = ModelResponse.from_dict(current.to_dict())
        return ModelResponse.from_dict(current.to_dict())

    async def _on_model_error(
        self,
        error: ModelProviderError,
        request: ModelRequest,
        context: RuntimeContext,
        control: RunControlState,
    ) -> ModelErrorDecision:
        current = ModelErrorDecision()
        info = error.info
        for hook in self._hooks:
            method = self._hook_method(hook, "on_model_error")
            if method is None:
                continue
            replacement = await self._call_hook(method, info, request, context, control=control)
            if replacement is not None:
                if not isinstance(cast(object, replacement), ModelErrorDecision):
                    raise TypeError("on_model_error must return ModelErrorDecision or None")
                current = ModelErrorDecision(
                    retry=replacement.retry,
                    message=replacement.message,
                )
        return current

    async def _before_tool(
        self, call: ToolCall, context: RuntimeContext, control: RunControlState
    ) -> ToolCall:
        current = call
        for hook in self._hooks:
            method = self._hook_method(hook, "before_tool")
            if method is None:
                continue
            replacement = await self._call_hook(method, current, context, control=control)
            if replacement is not None:
                current = cast(ToolCall, replacement)
            current = ToolCall.from_dict(current.to_dict())
        return ToolCall.from_dict(current.to_dict())

    async def _after_tool(
        self, result: ToolOutput, context: RuntimeContext, control: RunControlState
    ) -> ToolOutput:
        current = result
        for hook in self._hooks:
            method = self._hook_method(hook, "after_tool")
            if method is None:
                continue
            replacement = await self._call_hook(method, current, context, control=control)
            if replacement is not None:
                current = cast(ToolOutput, replacement)
            current = normalize_tool_output(current)
        return normalize_tool_output(current)

    async def _notify_hooks(self, name: str, *args: object, control: RunControlState) -> None:
        for hook in self._hooks:
            method = self._hook_method(hook, name)
            if method is None:
                continue
            await self._call_hook(method, *args, control=control)

    @staticmethod
    def _hook_method(hook: RuntimeHook, name: str) -> Callable[..., Any] | None:
        method = getattr(hook, name, None)
        if method is None:
            return None
        if not callable(method):
            raise TypeError(f"runtime hook {name} must be callable")
        return cast(Callable[..., Any], method)

    async def _call_hook(
        self,
        method: Callable[..., Any],
        *args: object,
        control: RunControlState,
    ) -> Any:
        if iscoroutinefunction(method):
            return await self._await_with_timeout(method(*args), control)
        value = await self._await_with_timeout(asyncio.to_thread(method, *args), control)
        if isawaitable(value):
            return await self._await_with_timeout(value, control)
        return value
