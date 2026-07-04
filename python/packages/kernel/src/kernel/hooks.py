"""Runtime hook contract."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, cast

from kernel.context import RuntimeContext
from kernel.errors import ModelErrorInfo
from kernel.events import AgentEvent, EventEmitter
from kernel.messages import ToolCall
from kernel.models import ModelRequest, ModelResponse
from kernel.state import AgentState
from kernel.status import AgentStatus
from kernel.tools import ToolOutput


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


class RuntimeHook(Protocol):
    """Structural hook marker for host-owned runtime extensions.

    Hook objects may implement any subset of the hook methods below. `AgentLoop`
    discovers methods structurally and skips hooks that do not implement a
    given method. Returning `None` keeps the current value.
    """


class EventHook(Protocol):
    def on_event(
        self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
    ) -> AgentEvent | Awaitable[AgentEvent | None] | None: ...


class BeforeModelHook(Protocol):
    def before_model(
        self, request: ModelRequest, context: RuntimeContext
    ) -> ModelRequest | Awaitable[ModelRequest | None] | None: ...


class AfterModelHook(Protocol):
    def after_model(
        self, response: ModelResponse, context: RuntimeContext
    ) -> ModelResponse | Awaitable[ModelResponse | None] | None: ...


class ModelErrorHook(Protocol):
    def on_model_error(
        self,
        error: ModelErrorInfo,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> ModelErrorDecision | Awaitable[ModelErrorDecision | None] | None: ...


class BeforeToolHook(Protocol):
    def before_tool(
        self, call: ToolCall, context: RuntimeContext
    ) -> ToolCall | Awaitable[ToolCall | None] | None: ...


class AfterToolHook(Protocol):
    def after_tool(
        self, result: ToolOutput, context: RuntimeContext
    ) -> ToolOutput | Awaitable[ToolOutput | None] | None: ...


class TransitionHook(Protocol):
    def on_transition(
        self,
        previous: AgentStatus,
        current: AgentStatus,
        state: AgentState,
        context: RuntimeContext,
    ) -> Awaitable[None] | None: ...
