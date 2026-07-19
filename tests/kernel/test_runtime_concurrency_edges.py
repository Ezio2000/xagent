from __future__ import annotations

import asyncio
from collections import deque
from typing import cast

from jharness.kernel import (
    Checkpoint,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    HistoryRewrite,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    RunHistory,
    RunSnapshot,
    Runtime,
    SettledResult,
    Suspended,
    Suspension,
    ToolBinding,
    ToolCall,
    ToolCatalog,
    ToolContext,
    ToolExecution,
    ToolSpec,
    ToolSuccess,
)


class _ScriptModel(Model):
    def __init__(self, responses: tuple[ModelResponse, ...]) -> None:
        self._responses = deque(responses)

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del request, context, stream, emit_delta
        return self._responses.popleft()


def _final() -> ModelResponse:
    return ModelResponse((ContentPart.text_part("done"),), finish_reason="stop")


class _EmptyCatalog(ToolCatalog):
    def specs(self) -> tuple[ToolSpec, ...]:
        return ()

    def spec(self, name: str) -> ToolSpec | None:
        del name
        return None

    def bind(self, call: ToolCall) -> ToolBinding:
        del call
        raise AssertionError("empty catalog cannot bind")


class _BlockingCatalogProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.tasks: list[asyncio.Task[object]] = []

    async def open_catalog(self) -> ToolCatalog:
        self.calls += 1
        task = asyncio.current_task()
        assert task is not None
        self.tasks.append(cast(asyncio.Task[object], task))
        self.started.set()
        await self.release.wait()
        return _EmptyCatalog()


async def test_catalog_insert_flood_is_iterative_and_leaves_no_task() -> None:
    provider = _BlockingCatalogProvider()
    invocation = Runtime(model=_ScriptModel((_final(),)), tools=provider).start(
        (Message.user("go"),)
    )
    result_task = asyncio.create_task(invocation.result())
    await provider.started.wait()

    for index in range(1_000):
        invocation.insert(Message.external(str(index)))
    await asyncio.sleep(0)
    provider.release.set()

    checkpoint = await asyncio.wait_for(result_task, timeout=10)
    assert checkpoint.snapshot.revision == 1_001
    assert len(checkpoint.snapshot.history) == 1_002
    assert provider.calls == 1
    assert provider.tasks and all(task.done() for task in provider.tasks)


async def test_pause_interrupts_catalog_initialization_without_leaking_work() -> None:
    provider = _BlockingCatalogProvider()
    invocation = Runtime(model=_ScriptModel((_final(),)), tools=provider).start(
        (Message.user("go"),)
    )
    result_task = asyncio.create_task(invocation.result())
    await provider.started.wait()

    invocation.pause(Suspension("user", "host"))
    checkpoint = await result_task

    assert isinstance(checkpoint.snapshot.state, Suspended)
    assert checkpoint.snapshot.revision == 1
    assert provider.tasks and all(task.done() for task in provider.tasks)


class _ImmediateBinding(ToolBinding):
    def __init__(
        self,
        call: ToolCall,
        spec: ToolSpec,
        settlements: list[str],
    ) -> None:
        self._call = call
        self._spec = spec
        self._settlements = settlements

    @property
    def call(self) -> ToolCall:
        return self._call

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, context: ToolContext) -> SettledResult:
        del context
        self._settlements.append(self._call.id)
        return SettledResult(ToolSuccess((ContentPart.text_part(self._call.id),)))


class _ParallelCatalog(ToolCatalog):
    def __init__(self, settlements: list[str]) -> None:
        self._settlements = settlements
        self._spec = ToolSpec(
            "lookup",
            "lookup",
            {"type": "object"},
            execution=ToolExecution("parallel", read_only=True, idempotent=True),
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        return (self._spec,)

    def spec(self, name: str) -> ToolSpec | None:
        return self._spec if name == self._spec.name else None

    def bind(self, call: ToolCall) -> ToolBinding:
        return _ImmediateBinding(call, self._spec, self._settlements)


class _StaticProvider:
    def __init__(self, catalog: ToolCatalog) -> None:
        self._catalog = catalog

    async def open_catalog(self) -> ToolCatalog:
        return self._catalog


class _CancellationBinding(ToolBinding):
    def __init__(self, call: ToolCall, spec: ToolSpec, observed: asyncio.Event) -> None:
        self._call = call
        self._spec = spec
        self._observed = observed

    @property
    def call(self) -> ToolCall:
        return self._call

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, context: ToolContext) -> SettledResult:
        while not context.cancel_requested:
            await asyncio.sleep(0)
        self._observed.set()
        return SettledResult(ToolSuccess((ContentPart.text_part("cancel observed"),)))


class _CancellationCatalog(ToolCatalog):
    def __init__(self, observed: asyncio.Event) -> None:
        self._observed = observed
        self._spec = ToolSpec("lookup", "lookup", {"type": "object"})

    def specs(self) -> tuple[ToolSpec, ...]:
        return (self._spec,)

    def spec(self, name: str) -> ToolSpec | None:
        return self._spec if name == self._spec.name else None

    def bind(self, call: ToolCall) -> ToolBinding:
        return _CancellationBinding(call, self._spec, self._observed)


async def _collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


async def test_parallel_tool_finished_events_follow_physical_settlement_order() -> None:
    for _ in range(20):
        settlements: list[str] = []
        calls = (ToolCall("a", "lookup"), ToolCall("b", "lookup"))
        model = _ScriptModel((ModelResponse(tool_calls=calls), _final()))
        checkpoint, events = await _collect(
            Runtime(model=model, tools=_StaticProvider(_ParallelCatalog(settlements))).start(
                (Message.user("go"),)
            )
        )
        finished = [
            cast(str, event.data["tool_call_id"])
            for event in events
            if event.kind is EventKind.TOOL_FINISHED
        ]
        assert finished == settlements
        assert tuple(message.tool_call_id for message in checkpoint.snapshot.history[2:4]) == (
            "a",
            "b",
        )


async def test_cooperative_cancellation_is_observed_only_after_its_event() -> None:
    observed = asyncio.Event()
    call = ToolCall("call-1", "lookup")
    model = _ScriptModel((ModelResponse(tool_calls=(call,)), _final()))
    invocation = Runtime(
        model=model,
        tools=_StaticProvider(_CancellationCatalog(observed)),
    ).start((Message.user("go"),))
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    kinds: list[EventKind] = []
    async for event in events:
        kinds.append(event.kind)
        if event.kind is EventKind.TOOL_STARTED:
            invocation.cancel_tool(call.id)

    await result_task
    assert observed.is_set()
    assert kinds.index(EventKind.TOOL_CANCEL_REQUESTED) < kinds.index(EventKind.TOOL_FINISHED)


class _InterruptedReducer:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.seen_roles: list[tuple[str, ...]] = []

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None:
        self.calls += 1
        self.seen_roles.append(tuple(message.role for message in snapshot.history))
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        if self.calls == 2:
            return HistoryRewrite(RunHistory((Message.user("summary"),)), "compact")
        return None


async def test_insert_interruption_retries_history_reduction_before_model() -> None:
    reducer = _InterruptedReducer()
    invocation = Runtime(
        model=_ScriptModel((_final(),)),
        history_reducer=reducer,
    ).start((Message.user("old"), Message.user("history")))

    async def insert_after_reducer_starts() -> None:
        await reducer.started.wait()
        invocation.insert(Message.external("new context"))

    insert_task = asyncio.create_task(insert_after_reducer_starts())
    checkpoint, events = await _collect(invocation)
    await insert_task
    facts = [
        event.data["fact"]["kind"]
        for event in events
        if event.kind is EventKind.CHECKPOINT_COMMITTED
    ]

    assert reducer.calls == 2
    assert reducer.seen_roles[1] == ("user", "user", "external")
    assert facts == ["started", "conversation_insert", "history_rewrite", "model_turn"]
    assert tuple(message.role for message in checkpoint.snapshot.history) == (
        "user",
        "assistant",
    )
