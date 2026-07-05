from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from time import monotonic
from time import time as wall_time
from typing import Any, cast

import kernel._loop.agent_loop as loop_module
import pytest
from kernel import (
    AgentEvent,
    AgentLoop,
    AgentState,
    AgentStatus,
    ContentPart,
    EventEmitter,
    EventTypes,
    InvalidToolCall,
    Message,
    ModelRequest,
    ModelResponse,
    PauseRequest,
    RunController,
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolSpec,
)
from kernel._loop.types import RunControlState


def user_text(text: str) -> Message:
    return Message.user([ContentPart.text_part(text)])


def parts_text(parts: Sequence[ContentPart]) -> str:
    return "".join(part.text or "" for part in parts)


class ScriptedModel:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if not self._responses:
            raise AssertionError("scripted model exhausted")
        return self._responses.pop(0)


class EchoRegistry:
    _spec = ToolSpec(
        name="echo",
        description="Return text.",
        input_schema={"type": "object", "properties": {}},
    )

    def specs(self) -> tuple[ToolSpec, ...]:
        return (ToolSpec.from_dict(self._spec.to_dict()),)

    def spec_for(self, name: str) -> ToolSpec | None:
        if name != "echo":
            return None
        return ToolSpec.from_dict(self._spec.to_dict())

    def validate_call(self, call: ToolCall) -> None:
        if call.name != "echo":
            raise InvalidToolCall(f"unknown tool: {call.name}")

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContext,
        *,
        progress_emitter: Callable[[Mapping[str, Any]], None] | None = None,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> ToolOutput:
        _ = context, progress_emitter, cancel_checker
        return ToolObservation.text(str(call.arguments.get("text", "")))


class KeyboardInterruptOnCloseIterator:
    def __aiter__(self) -> KeyboardInterruptOnCloseIterator:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    def aclose(self) -> None:
        raise KeyboardInterrupt


class RecursiveCustomEventHook(RuntimeHook):
    def on_event(self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter) -> None:
        _ = context
        emitter.emit("recursive_custom_event", {"parent_type": event.type})


class PauseExposingLoop(AgentLoop):
    async def await_model_for_test(
        self,
        awaitable: Awaitable[ModelResponse],
        control: RunControlState,
    ) -> ModelResponse:
        return await self._await_model_with_interrupt(awaitable, control)

    async def apply_pause_for_test(
        self,
        state: AgentState,
        context: RuntimeContext,
        control: RunControlState,
        request: PauseRequest,
    ) -> tuple[AgentEvent, ...]:
        return await self._pause(
            state,
            context,
            control,
            request,
            resume_status=AgentStatus.EXECUTING_TOOLS,
            origin="control",
        )


@pytest.mark.asyncio
async def test_close_async_iterator_does_not_swallow_keyboard_interrupt() -> None:
    loop = AgentLoop(model=ScriptedModel([]))
    close_async_iterator = cast(
        Callable[[object], Awaitable[None]],
        object.__getattribute__(loop, "_close_async_iterator"),
    )

    with pytest.raises(KeyboardInterrupt):
        await close_async_iterator(KeyboardInterruptOnCloseIterator())


@pytest.mark.asyncio
async def test_completed_model_task_wins_same_turn_interrupt_race() -> None:
    controller = RunController()
    controller.interrupt(reason="race_interrupt")
    loop = PauseExposingLoop(model=ScriptedModel([]))
    future: asyncio.Future[ModelResponse] = asyncio.get_running_loop().create_future()
    future.set_result(ModelResponse.text("done"))

    response = await loop.await_model_for_test(
        future,
        RunControlState(
            run_id="race-run",
            started_at=wall_time(),
            run_controller=controller,
        ),
    )

    assert parts_text(response.parts) == "done"


@pytest.mark.asyncio
async def test_external_wait_pause_respects_expired_deadline() -> None:
    now = wall_time()
    context = RuntimeContext(run_id="expired-wait", started_at=now - 2, deadline=now - 1)
    control = RunControlState(
        run_id=context.run_id,
        started_at=context.started_at,
        deadline=context.deadline,
        monotonic_deadline=monotonic() - 1,
    )
    state = AgentState(
        status=AgentStatus.EXECUTING_TOOLS,
        messages=[user_text("wait")],
    )

    events = await PauseExposingLoop(model=ScriptedModel([])).apply_pause_for_test(
        state,
        context,
        control,
        PauseRequest(
            reason="external_callback",
            source="tool",
            wait_id="job-1",
            metadata={},
        ),
    )

    assert state.status is AgentStatus.LIMIT_EXCEEDED
    assert state.error == "timeout_seconds"
    assert state.pause is None
    assert EventTypes.PAUSE_REQUESTED not in [event.type for event in events]


@pytest.mark.asyncio
async def test_model_response_checkpoint_rechecks_timeout_after_state_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = False

    def fake_monotonic() -> float:
        return 2.0 if expired else 0.0

    class ExpireAfterModelTransitionHook(RuntimeHook):
        def on_event(
            self, event: AgentEvent, context: RuntimeContext, emitter: EventEmitter
        ) -> None:
            nonlocal expired
            _ = context, emitter
            if (
                event.type == EventTypes.STATE_CHANGED
                and event.data["from"] == AgentStatus.PLANNING.value
                and event.data["to"] == AgentStatus.EXECUTING_TOOLS.value
            ):
                expired = True

    monkeypatch.setattr(loop_module, "monotonic", fake_monotonic)
    now = wall_time()
    model = ScriptedModel(
        [ModelResponse(tool_calls=[ToolCall(id="call-1", name="echo", arguments={"text": "late"})])]
    )

    events = [
        event
        async for event in AgentLoop(
            model=model,
            tools=EchoRegistry(),
            hooks=[ExpireAfterModelTransitionHook()],
        ).run_events(
            [user_text("echo")],
            context=RuntimeContext(started_at=now, deadline=now + 1),
        )
    ]

    checkpoint_statuses = [
        RunSnapshot.from_dict(event.data).state.status
        for event in events
        if event.type == EventTypes.CHECKPOINT
    ]
    assert AgentStatus.EXECUTING_TOOLS not in checkpoint_statuses
    assert events[-1].data["state"]["status"] == AgentStatus.LIMIT_EXCEEDED.value


@pytest.mark.asyncio
async def test_custom_event_dispatch_chain_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop_module, "_MAX_DISPATCHED_EVENTS_PER_RUNTIME_EVENT", 4)

    result = await AgentLoop(
        model=ScriptedModel([ModelResponse.text("done")]),
        hooks=[RecursiveCustomEventHook()],
    ).run([user_text("hello")])

    assert result.status is AgentStatus.FAILED
    assert result.error == "custom event dispatch limit exceeded"
