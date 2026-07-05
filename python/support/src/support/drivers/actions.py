"""Action-aware model drivers for controlled runtime scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from kernel import (
    ConversationInsert,
    ModelCapabilities,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    PauseRequest,
    RunController,
    RuntimeContext,
)

ModelStep = ModelResponse | ModelProviderError


@dataclass(slots=True, frozen=True)
class ModelStreamSleep:
    """Sleep action inside a controlled model stream."""

    seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.seconds, bool):
            raise TypeError("model stream sleep seconds must be a number")
        seconds = float(self.seconds)
        if seconds < 0:
            raise ValueError("model stream sleep seconds must be >= 0")
        object.__setattr__(self, "seconds", seconds)


@dataclass(slots=True, frozen=True)
class ModelStreamPause:
    """Pause action inside a controlled model stream."""

    request: PauseRequest

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.request), PauseRequest):
            raise TypeError("model stream pause request must be PauseRequest")
        object.__setattr__(self, "request", PauseRequest.from_dict(self.request.to_dict()))


ModelStreamAction: TypeAlias = ModelStreamEvent | ModelStreamSleep | ModelStreamPause


def apply_pause_request(controller: RunController, request: PauseRequest) -> None:
    """Apply a pause request to a run controller."""

    if request.interrupt:
        controller.interrupt(
            reason=request.reason,
            source=request.source,
            wait_id=request.wait_id,
            metadata=request.metadata,
        )
        return
    controller.request_pause(
        reason=request.reason,
        source=request.source,
        wait_id=request.wait_id,
        metadata=request.metadata,
    )


class ControlledModelDriver:
    """Scripted model driver with optional controller actions and request validation."""

    def __init__(
        self,
        steps: Sequence[ModelStep],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
        validate_request: Callable[[ModelRequest], None] | None = None,
    ) -> None:
        self._steps = list(steps)
        self._controller = controller
        self._pause_request_on_call = pause_request_on_call
        self._conversation_insert_on_call = conversation_insert_on_call
        self._validate_request = validate_request
        self._pause_requested = False
        self._conversation_inserted = False
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self._assert_model_request_contract(request)
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        step = self._steps[self.calls]
        self.calls += 1
        self._apply_conversation_insert_once(self._conversation_insert_on_call)
        self._apply_pause_once(self._pause_request_on_call)
        if isinstance(step, ModelProviderError):
            raise step
        return step

    def _apply_pause_once(self, request: PauseRequest | None) -> None:
        if self._controller is not None and request is not None and not self._pause_requested:
            self._pause_requested = True
            apply_pause_request(self._controller, request)

    def _apply_conversation_insert_once(self, insert: ConversationInsert | None) -> None:
        if self._controller is not None and insert is not None and not self._conversation_inserted:
            self._conversation_inserted = True
            self._controller.insert(insert)

    def _assert_model_request_contract(self, request: ModelRequest) -> None:
        if self._validate_request is not None:
            self._validate_request(request)


class ControlledStreamingModelDriver(ControlledModelDriver):
    """Controlled streaming model driver with explicit stream actions."""

    capabilities = ModelCapabilities(streaming=True)

    def __init__(
        self,
        steps: Sequence[ModelStep],
        stream_steps: Sequence[Sequence[ModelStreamAction]],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
        validate_request: Callable[[ModelRequest], None] | None = None,
    ) -> None:
        super().__init__(
            steps,
            controller=controller,
            pause_request_on_call=pause_request_on_call,
            conversation_insert_on_call=conversation_insert_on_call,
            validate_request=validate_request,
        )
        self._stream_steps = [tuple(step) for step in stream_steps]
        self.stream_calls = 0

    async def stream(
        self,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> AsyncIterator[ModelStreamEvent]:
        _ = context
        self._assert_model_request_contract(request)
        if self.stream_calls >= len(self._stream_steps):
            raise AssertionError("scripted stream model exhausted")

        step = self._stream_steps[self.stream_calls]
        self.stream_calls += 1
        self._apply_conversation_insert_once(self._conversation_insert_on_call)
        self._apply_pause_once(self._pause_request_on_call)
        for action in step:
            if isinstance(action, ModelStreamSleep):
                await asyncio.sleep(action.seconds)
                continue
            if isinstance(action, ModelStreamPause):
                self._apply_pause_once(action.request)
                continue
            yield action
