"""Runtime hook contract."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import TypeAlias, TypeVar

from agent_runtime.events import AgentEvent, EventEmitter
from agent_runtime.messages import ToolCall
from agent_runtime.models import ModelRequest, ModelResponse
from agent_runtime.runtime import RuntimeContext
from agent_runtime.state import AgentState, AgentStatus
from agent_runtime.tools import ToolResult

T = TypeVar("T")
MaybeAwaitable: TypeAlias = T | Awaitable[T]


class RuntimeHook:
    """Base hook class for host-owned runtime extensions.

    Subclasses may override any method. Returning `None` keeps the current value.
    Hook methods may be sync or async.
    """

    def on_event(
        self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
    ) -> MaybeAwaitable[AgentEvent | None] | None:
        _ = event, context, emitter
        return None

    def before_model(
        self, request: ModelRequest, context: RuntimeContext
    ) -> MaybeAwaitable[ModelRequest | None] | None:
        return None

    def after_model(
        self, response: ModelResponse, context: RuntimeContext
    ) -> MaybeAwaitable[ModelResponse | None] | None:
        return None

    def before_tool(
        self, call: ToolCall, context: RuntimeContext
    ) -> MaybeAwaitable[ToolCall | None] | None:
        return None

    def after_tool(
        self, result: ToolResult, context: RuntimeContext
    ) -> MaybeAwaitable[ToolResult | None] | None:
        return None

    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> MaybeAwaitable[None] | None:
        return None
