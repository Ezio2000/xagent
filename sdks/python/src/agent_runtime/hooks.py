"""Runtime hook contract."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar, cast

from agent_runtime.errors import ModelErrorInfo
from agent_runtime.events import AgentEvent, EventEmitter
from agent_runtime.messages import ToolCall
from agent_runtime.models import ModelRequest, ModelResponse
from agent_runtime.runtime import RuntimeContext
from agent_runtime.state import AgentState, AgentStatus
from agent_runtime.tools import ToolOutput

T = TypeVar("T")
MaybeAwaitable: TypeAlias = T | Awaitable[T]


@dataclass(slots=True, frozen=True)
class ModelErrorDecision:
    """Host decision for a provider-neutral model failure."""

    retry: bool = False
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.retry), bool):
            raise TypeError("model error decision retry must be a boolean")
        if self.message is not None and not isinstance(cast(object, self.message), str):
            raise TypeError("model error decision message must be a string or None")
        if self.message == "":
            raise ValueError("model error decision message must not be empty")


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

    def on_model_error(
        self,
        error: ModelErrorInfo,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> MaybeAwaitable[ModelErrorDecision | None] | None:
        _ = error, request, context
        return None

    def before_tool(
        self, call: ToolCall, context: RuntimeContext
    ) -> MaybeAwaitable[ToolCall | None] | None:
        return None

    def after_tool(
        self, result: ToolOutput, context: RuntimeContext
    ) -> MaybeAwaitable[ToolOutput | None] | None:
        return None

    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> MaybeAwaitable[None] | None:
        return None
