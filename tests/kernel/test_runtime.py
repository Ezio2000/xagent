from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping

import pytest

from jharness.kernel import (
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalSuspend,
    Checkpoint,
    CommitError,
    ContentPart,
    DeltaSink,
    DurableCommit,
    Event,
    EventKind,
    HistoryRewrite,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PendingToolCalls,
    Planning,
    RequestError,
    RunContext,
    RunHistory,
    RunLimits,
    RunRepository,
    RunSnapshot,
    Runtime,
    SettledResult,
    Suspended,
    Suspension,
    SuspensionSelector,
    ToolBinding,
    ToolCall,
    ToolCatalog,
    ToolCatalogProvider,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)


class ScriptModel(Model):
    def __init__(self, responses: list[ModelResponse], *, streaming: bool = False) -> None:
        self.responses = deque(responses)
        self.requests: list[ModelRequest] = []
        self._capabilities = ModelCapabilities(streaming=streaming)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context
        self.requests.append(request)
        if stream and emit_delta is not None:
            await emit_delta(ModelContentDelta(0, "live"))
        return self.responses.popleft()


class BlockingModel(Model):
    def __init__(self, final: ModelResponse) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.final = final

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
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return self.final


ToolEffect = Callable[[ToolContext], Awaitable[ToolResult]]


class Binding:
    __slots__ = ("_call", "_effect", "_spec")

    def __init__(self, call: ToolCall, spec: ToolSpec, effect: ToolEffect) -> None:
        self._call = call
        self._spec = spec
        self._effect = effect

    @property
    def call(self) -> ToolCall:
        return self._call

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def invoke(self, context: ToolContext) -> ToolResult:
        return await self._effect(context)


class Catalog(ToolCatalog):
    def __init__(self, effects: Mapping[str, tuple[ToolSpec, ToolEffect]]) -> None:
        self.effects = dict(effects)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec, _ in self.effects.values())

    def spec(self, name: str) -> ToolSpec | None:
        item = self.effects.get(name)
        return None if item is None else item[0]

    def bind(self, call: ToolCall) -> ToolBinding:
        spec, effect = self.effects[call.name]
        return Binding(call, spec, effect)


class CatalogProvider(ToolCatalogProvider):
    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog

    async def open_catalog(self) -> ToolCatalog:
        return self.catalog


def final(text: str = "done", *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse((ContentPart.text_part(text),), finish_reason="stop", usage=usage)


async def success(context: ToolContext) -> ToolResult:
    del context
    return SettledResult(ToolSuccess((ContentPart.text_part("tool-ok"),), {"ok": True}))


def tool_provider(
    effect: ToolEffect = success,
    *,
    execution: ToolExecution | None = None,
    name: str = "lookup",
) -> CatalogProvider:
    spec = ToolSpec(
        name,
        "Lookup",
        {"type": "object"},
        execution=ToolExecution() if execution is None else execution,
    )
    return CatalogProvider(Catalog({name: (spec, effect)}))


async def collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


def test_start_rejects_invalid_planning_history_before_creating_invocation() -> None:
    call = ToolCall("call-1", "lookup")
    model = ScriptModel([final()])

    with pytest.raises(ValueError, match="unresolved"):
        Runtime(model=model).start((Message.user("go"), Message.assistant(tool_calls=(call,))))

    assert model.requests == []


async def test_final_run_result_and_events_share_one_execution() -> None:
    model = ScriptModel([final()], streaming=True)
    invocation = Runtime(model=model).start((Message.user("hello"),), stream=True)
    checkpoint, events = await collect(invocation)
    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.revision == 1
    assert [event.kind for event in events] == [
        EventKind.INVOCATION_STARTED,
        EventKind.CHECKPOINT_COMMITTED,
        EventKind.MODEL_STARTED,
        EventKind.MODEL_DELTA,
        EventKind.MODEL_FINISHED,
        EventKind.CHECKPOINT_COMMITTED,
        EventKind.INVOCATION_STOPPED,
    ]
    assert await invocation.result() is checkpoint
    assert len(model.requests) == 1


async def test_model_request_receives_complete_history() -> None:
    messages = tuple(Message.user(str(index)) for index in range(256))
    call = ToolCall("call-complete-history", "lookup")
    model = ScriptModel([ModelResponse(tool_calls=(call,)), final()])

    checkpoint = await Runtime(model=model, tools=tool_provider()).start(messages).result()

    assert model.requests[0].messages == messages
    assert model.requests[1].messages == tuple(checkpoint.snapshot.history)[:-1]


async def test_result_only_rejects_late_event_subscription() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    assert object.__getattribute__(invocation, "_queue") is None
    first = await invocation.result()
    assert object.__getattribute__(invocation, "_queue") is None
    assert await invocation.result() is first
    with pytest.raises(RuntimeError, match="result-only"):
        invocation.events()


async def test_event_subscription_allocates_the_observation_queue_lazily() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    assert object.__getattribute__(invocation, "_queue") is None
    events = invocation.events()
    assert isinstance(object.__getattribute__(invocation, "_queue"), asyncio.Queue)
    with pytest.raises(RuntimeError, match="consumed only once"):
        invocation.events()
    assert [event async for event in events][-1].kind is EventKind.INVOCATION_STOPPED
    assert object.__getattribute__(invocation, "_queue") is None


async def test_finished_invocation_discards_all_control_operations() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    checkpoint = await invocation.result()
    assert object.__getattribute__(invocation, "_execute") is None

    invocation.pause(Suspension("late", "host"))
    invocation.insert(Message.external("late"))
    invocation.cancel_tool("late-call")

    control = object.__getattribute__(invocation, "_control")
    assert object.__getattribute__(control, "_active") is None
    assert not object.__getattribute__(control, "_pending")
    assert await invocation.result() is checkpoint


async def test_tool_result_is_committed_in_model_order() -> None:
    call = ToolCall("call-1", "lookup", {"q": "x"})
    model = ScriptModel([ModelResponse(tool_calls=(call,)), final()])
    checkpoint, events = await collect(
        Runtime(model=model, tools=tool_provider()).start((Message.user("go"),))
    )
    assert checkpoint.snapshot.revision == 3
    assert [message.role for message in checkpoint.snapshot.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolSuccess)
    assert [event.kind for event in events].count(EventKind.CHECKPOINT_COMMITTED) == 4
    selected = [event for event in events if event.kind is EventKind.TOOL_BATCH_SELECTED]
    assert len(selected) == 1
    assert selected[0].data["call_ids"] == (call.id,)
    assert selected[0].data["remaining_count"] == 0


async def test_serial_tool_batches_do_not_materialize_the_remaining_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = tuple(ToolCall(f"call-{index}", "lookup") for index in range(40))

    def fail_prefix(_: PendingToolCalls, _count: int) -> tuple[ToolCall, ...]:
        raise AssertionError("tool selection materialized the pending suffix")

    monkeypatch.setattr(PendingToolCalls, "prefix", fail_prefix)
    checkpoint = (
        await Runtime(
            model=ScriptModel([ModelResponse(tool_calls=calls), final()]),
            tools=tool_provider(),
            limits=RunLimits(max_tool_calls=len(calls), max_tool_batch_size=len(calls)),
        )
        .start((Message.user("go"),))
        .result()
    )

    assert checkpoint.snapshot.metrics.tool_calls == len(calls)
    tail = checkpoint.snapshot.history.iter_tail(len(calls) + 1)
    assert [message.tool_call_id for message in tail if message.role == "tool"] == [
        call.id for call in calls
    ]


async def test_waiting_result_suspends_and_exact_resume_completes() -> None:
    suspension = Suspension("external_work", "tool", wait_id="wait-1")

    async def waiting(context: ToolContext) -> ToolResult:
        del context
        return WaitingResult(ToolWaiting((ContentPart.text_part("waiting"),)), suspension)

    call = ToolCall("call-1", "lookup")
    model = ScriptModel([ModelResponse(tool_calls=(call,)), final("resumed")])
    runtime = Runtime(model=model, tools=tool_provider(waiting))
    paused = await runtime.start((Message.user("go"),)).result()
    assert isinstance(paused.snapshot.state, Suspended)
    assert paused.snapshot.revision == 2
    resumed = await runtime.resume(paused, selector=SuspensionSelector(wait_id="wait-1")).result()
    assert resumed.snapshot.status == "completed"
    assert resumed.snapshot.revision == 4
    with pytest.raises(RequestError, match="does not match") as mismatch:
        runtime.resume(paused, selector=SuspensionSelector(wait_id="other"))
    assert mismatch.value.code == "suspension_mismatch"


async def test_pause_and_insert_interrupt_model_without_partial_commit() -> None:
    pause_model = BlockingModel(final())
    pause_invocation = Runtime(model=pause_model).start((Message.user("go"),))
    pause_events = pause_invocation.events()
    pause_task = asyncio.create_task(pause_invocation.result())
    async for event in pause_events:
        if event.kind is EventKind.MODEL_STARTED:
            pause_invocation.pause(Suspension("user", "host"))
    paused = await pause_task
    assert paused.snapshot.status == "suspended"
    assert paused.snapshot.revision == 1

    insert_model = BlockingModel(final())
    insert_invocation = Runtime(model=insert_model).start((Message.user("go"),))
    insert_events = insert_invocation.events()
    insert_task = asyncio.create_task(insert_invocation.result())
    inserted_once = False
    async for event in insert_events:
        if event.kind is EventKind.MODEL_STARTED and not inserted_once:
            inserted_once = True
            insert_invocation.insert(Message.external("new information"))
    inserted = await insert_task
    assert inserted.snapshot.status == "completed"
    assert [message.role for message in inserted.snapshot.history] == [
        "user",
        "external",
        "assistant",
    ]


async def test_total_token_limit_commits_complete_model_observation_then_limits() -> None:
    model = ScriptModel([final(usage=ModelUsage(total_tokens=6))])
    checkpoint = (
        await Runtime(model=model, limits=RunLimits(max_total_tokens=5))
        .start((Message.user("go"),))
        .result()
    )
    assert checkpoint.snapshot.status == "limited"
    assert checkpoint.snapshot.metrics.planning_steps == 1
    assert checkpoint.snapshot.metrics.usage.total_tokens == 6


class RejectSecondCommit(RunRepository):
    def __init__(self) -> None:
        self.commits: list[DurableCommit] = []

    async def commit(self, commit: DurableCommit) -> None:
        if self.commits:
            raise RuntimeError("storage unavailable")
        self.commits.append(commit)


class BlockingStartCommit(RunRepository):
    def __init__(self) -> None:
        self.attempts = 0
        self.cancelled = False

    async def commit(self, commit: DurableCommit) -> None:
        del commit
        self.attempts += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class CancellationIgnoringModel(Model):
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

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
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
        finally:
            self.finished.set()
        return final()


async def test_start_commit_is_bounded_by_the_invocation_work_deadline() -> None:
    repository = BlockingStartCommit()
    model = ScriptModel([final()])
    invocation = Runtime(
        model=model,
        limits=RunLimits(timeout_seconds=0.02),
        repository=repository,
        repository_timeout=1.0,
    ).start((Message.user("go"),))

    with pytest.raises(
        CommitError,
        match="start checkpoint commit exceeded work deadline",
    ) as caught:
        await invocation.result()

    assert caught.value.last_checkpoint is None
    assert repository.attempts == 1
    assert repository.cancelled
    assert model.requests == []


async def test_noncompliant_port_is_reported_after_bounded_cleanup() -> None:
    model = CancellationIgnoringModel()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    reports: list[Mapping[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    try:
        checkpoint = (
            await Runtime(
                model=model,
                limits=RunLimits(timeout_seconds=0.01),
            )
            .start((Message.user("go"),))
            .result()
        )
        assert checkpoint.snapshot.status == "limited"
        assert model.cancellations >= 1
        assert any("abandoned a port task" in str(item.get("message")) for item in reports)
    finally:
        model.release.set()
        await asyncio.wait_for(model.finished.wait(), timeout=1)
        loop.set_exception_handler(previous_handler)


async def test_repository_failure_preserves_last_checkpoint_and_stops_observation() -> None:
    repository = RejectSecondCommit()
    invocation = Runtime(model=ScriptModel([final()]), repository=repository).start(
        (Message.user("go"),)
    )
    events = invocation.events()
    with pytest.raises(CommitError, match="storage unavailable") as caught:
        async for _ in events:
            pass
    assert caught.value.last_checkpoint is repository.commits[0].checkpoint


class DenyPolicy(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalDeny(request.call.id, "denied") for request in requests)


class SuspendPolicy(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(
            ApprovalSuspend(request.call.id, Suspension("approval", "policy"))
            for request in requests
        )


async def test_approval_deny_is_model_visible_and_suspend_invokes_nothing() -> None:
    invoked = 0

    async def effect(context: ToolContext) -> ToolResult:
        nonlocal invoked
        del context
        invoked += 1
        return SettledResult(ToolSuccess((ContentPart.text_part("unexpected"),)))

    call = ToolCall("call-1", "lookup")
    denied_model = ScriptModel([ModelResponse(tool_calls=(call,)), final()])
    denied = (
        await Runtime(
            model=denied_model,
            tools=tool_provider(effect),
            approval_policy=DenyPolicy(),
        )
        .start((Message.user("go"),))
        .result()
    )
    assert isinstance(denied.snapshot.history[2].outcome, ToolFailure)
    assert invoked == 0

    suspended_model = ScriptModel([ModelResponse(tool_calls=(call,))])
    suspended = (
        await Runtime(
            model=suspended_model,
            tools=tool_provider(effect),
            approval_policy=SuspendPolicy(),
        )
        .start((Message.user("go"),))
        .result()
    )
    assert isinstance(suspended.snapshot.state, Suspended)
    assert invoked == 0


class OneRewrite:
    def __init__(self) -> None:
        self.used = False

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None:
        del snapshot
        if self.used:
            return None
        self.used = True
        return HistoryRewrite(RunHistory((Message.user("summary"),)), "compact")


async def test_history_rewrite_is_a_separate_checkpoint() -> None:
    checkpoint, events = await collect(
        Runtime(model=ScriptModel([final()]), history_reducer=OneRewrite()).start(
            (Message.user("old"), Message.user("history"))
        )
    )
    facts = [
        event.data["fact"]["kind"]
        for event in events
        if event.kind is EventKind.CHECKPOINT_COMMITTED
    ]
    assert facts == ["started", "history_rewrite", "model_turn"]
    assert checkpoint.snapshot.history[0].parts[0].text == "summary"


async def test_consumer_close_returns_last_committed_checkpoint() -> None:
    invocation = Runtime(model=BlockingModel(final())).start((Message.user("go"),))
    events = invocation.events()
    assert (await anext(events)).kind is EventKind.INVOCATION_STARTED
    assert (await anext(events)).kind is EventKind.CHECKPOINT_COMMITTED
    assert (await anext(events)).kind is EventKind.MODEL_STARTED
    await events.aclose()
    checkpoint = await invocation.result()
    assert checkpoint.snapshot.state == Planning()
