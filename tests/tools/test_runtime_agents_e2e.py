from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from jharness.kernel import (
    Checkpoint,
    Completed,
    ContentPart,
    DeltaSink,
    ErrorInfo,
    Failed,
    Invocation,
    Limited,
    Message,
    Model,
    ModelCapabilities,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    RunContext,
    RunLimits,
    Runtime,
    Suspended,
    Suspension,
    ToolAccepted,
    ToolCall,
    ToolChoice,
    ToolSuccess,
    ToolWaiting,
    thaw_json_value,
)
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint
from jharness.toolkit import Tool, ToolRegistry
from jharness.tools import ReadTool
from jharness.tools.agent import (
    AgentBackendError,
    AgentCancelTool,
    AgentGetTool,
    AgentRequest,
    AgentSnapshot,
    AgentTool,
    AgentWaitTool,
    extract_agent_wait,
    resume_agent,
)

_Mode = Literal["foreground", "background_wait", "cancel"]
_TERMINAL = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class _InheritedRuntimeConfig:
    model: Model
    delegable_tools: tuple[Tool, ...]
    limits: RunLimits
    model_options: ModelOptions
    tool_choice: ToolChoice
    system_prompt: str


@dataclass(slots=True)
class _AgentRecord:
    request: AgentRequest
    parent: RunContext
    parent_tool_call_id: str
    snapshot: AgentSnapshot
    child_context: RunContext
    started: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    waiters: set[tuple[str, str]] = field(default_factory=lambda: set[tuple[str, str]]())
    invocation: Invocation | None = None
    checkpoint: Checkpoint | None = None
    task: asyncio.Task[None] | None = None


class _InMemoryAgentHost:
    """Test Host that actually supervises Child Runtime invocations."""

    def __init__(self, config: _InheritedRuntimeConfig) -> None:
        self.config = config
        self.records: dict[str, _AgentRecord] = {}
        self._by_start_key: dict[tuple[str, str], str] = {}

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        key = (parent.run_id, parent_tool_call_id)
        existing_id = self._by_start_key.get(key)
        if existing_id is not None:
            record = self.records[existing_id]
            if record.request != request:
                raise AgentBackendError(
                    "agent_conflict",
                    "The parent tool call already owns a different Agent request.",
                )
            return record.snapshot

        agent_id = _agent_id(parent.run_id, parent_tool_call_id)
        child_context = RunContext(
            run_id=f"{agent_id}:run",
            started_at=time.time(),
            deadline=parent.deadline,
            parent_run_id=parent.run_id,
            parent_tool_call_id=parent_tool_call_id,
            run_kind="agent",
            metadata={"agent_id": agent_id, "agent_depth": 1},
        )
        record = _AgentRecord(
            request=request,
            parent=parent,
            parent_tool_call_id=parent_tool_call_id,
            snapshot=AgentSnapshot(
                agent_id,
                request.description,
                "running",
                request.background,
            ),
            child_context=child_context,
        )
        self.records[agent_id] = record
        self._by_start_key[key] = agent_id
        record.task = asyncio.create_task(self._run_child(record))
        await record.started.wait()
        return record.snapshot

    async def get(self, agent_id: str, *, requester: RunContext) -> AgentSnapshot:
        return self._authorized_record(agent_id, requester).snapshot

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        record = self._authorized_record(agent_id, requester)
        if record.snapshot.status not in _TERMINAL:
            record.waiters.add((requester.run_id, requester_tool_call_id))
        return record.snapshot

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        del requester_tool_call_id
        record = self._authorized_record(agent_id, requester)
        if record.snapshot.status in _TERMINAL:
            return record.snapshot
        record.snapshot = replace(record.snapshot, cancellation_requested=True)
        await record.started.wait()
        invocation = record.invocation
        if invocation is None:
            raise RuntimeError("Child invocation was not registered")
        invocation.pause(
            Suspension(
                reason="agent_cancelled",
                source="AgentCancel",
                wait_id=agent_id,
            )
        )
        await record.done.wait()
        return record.snapshot

    async def wait_terminal(self, agent_id: str) -> AgentSnapshot:
        record = self.records[agent_id]
        await record.done.wait()
        return record.snapshot

    async def resume_parent(
        self,
        runtime: Runtime,
        checkpoint: Checkpoint,
        agent_id: str,
    ) -> Checkpoint:
        snapshot = await self.wait_terminal(agent_id)
        return await resume_agent(runtime, checkpoint, snapshot).result()

    def only_record(self) -> _AgentRecord:
        assert len(self.records) == 1
        return next(iter(self.records.values()))

    def _authorized_record(self, agent_id: str, requester: RunContext) -> _AgentRecord:
        record = self.records.get(agent_id)
        if record is None or record.parent.run_id != requester.run_id:
            raise AgentBackendError("agent_not_found", "Agent not found.")
        return record

    async def _run_child(self, record: _AgentRecord) -> None:
        runtime = Runtime(
            model=self.config.model,
            tools=ToolRegistry(self.config.delegable_tools),
            limits=self.config.limits,
            model_options=self.config.model_options,
            tool_choice=self.config.tool_choice,
        )
        invocation = runtime.start(
            (
                Message.system(self.config.system_prompt),
                Message.user(record.request.prompt),
            ),
            context=record.child_context,
        )
        record.invocation = invocation
        record.started.set()
        try:
            checkpoint = await invocation.result()
            record.checkpoint = checkpoint
            record.snapshot = _terminal_snapshot(record, checkpoint)
        except Exception as exc:
            record.snapshot = AgentSnapshot(
                record.snapshot.agent_id,
                record.request.description,
                "failed",
                record.request.background,
                error=ErrorInfo("child_host_error", str(exc) or exc.__class__.__name__),
            )
        finally:
            record.done.set()


class _EndToEndModel(Model):
    def __init__(
        self,
        mode: _Mode,
        *,
        expected_options: ModelOptions,
        expected_tool_choice: ToolChoice,
    ) -> None:
        self.mode = mode
        self.expected_options = expected_options
        self.expected_tool_choice = expected_tool_choice
        self.release_child = asyncio.Event()
        self.child_entered = asyncio.Event()
        self.child_model_cancelled = False
        self.parent_requests: list[ModelRequest] = []
        self.child_requests: list[tuple[ModelRequest, RunContext]] = []
        self.observed_completion: dict[str, object] | None = None

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
        del stream, emit_delta
        if context.run_kind == "agent":
            return await self._invoke_child(request, context)
        return self._invoke_parent(request)

    async def _invoke_child(
        self,
        request: ModelRequest,
        context: RunContext,
    ) -> ModelResponse:
        self.child_requests.append((request, context))
        assert request.options == self.expected_options
        assert request.tool_choice == self.expected_tool_choice
        assert [spec.name for spec in request.tools] == ["Read"]
        assert context.parent_run_id is not None
        assert context.parent_tool_call_id is not None

        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            self.child_entered.set()
            try:
                await self.release_child.wait()
            except asyncio.CancelledError:
                self.child_model_cancelled = True
                raise
            return ModelResponse(
                tool_calls=(ToolCall("child-read", "Read", {"file_path": "evidence.txt"}),)
            )

        outcome = tool_messages[-1].outcome
        assert isinstance(outcome, ToolSuccess)
        text = outcome.parts[0].text
        assert text is not None and "E2E evidence from disk" in text
        return ModelResponse(
            (
                ContentPart.text_part(
                    "Child completed after inherited Read tool: E2E evidence from disk"
                ),
            ),
            finish_reason="stop",
        )

    def _invoke_parent(self, request: ModelRequest) -> ModelResponse:
        self.parent_requests.append(request)
        completion = _external_completion(request)
        if completion is not None:
            self.observed_completion = completion
            return ModelResponse(
                (
                    ContentPart.text_part(
                        f"Parent observed {completion['status']}: {completion.get('result', '')}"
                    ),
                ),
                finish_reason="stop",
            )

        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "delegate-child",
                        "Agent",
                        {
                            "description": "Read delegated evidence",
                            "prompt": "Read evidence.txt and report its evidence.",
                            "background": self.mode != "foreground",
                        },
                    ),
                )
            )

        agent_id = _accepted_agent_id(tool_messages)
        if self.mode == "background_wait":
            return self._background_wait_step(tool_messages, agent_id)
        if self.mode == "cancel":
            return self._cancel_step(tool_messages, agent_id)
        raise AssertionError("foreground Parent should resume through an external completion")

    def _background_wait_step(
        self,
        tool_messages: list[Message],
        agent_id: str,
    ) -> ModelResponse:
        if len(tool_messages) == 1:
            return ModelResponse(
                tool_calls=(ToolCall("get-child", "AgentGet", {"agent_id": agent_id}),)
            )
        assert len(tool_messages) == 2
        get_outcome = tool_messages[-1].outcome
        assert isinstance(get_outcome, ToolSuccess)
        payload = thaw_json_value(get_outcome.structured_content)
        assert isinstance(payload, dict) and payload["status"] == "running"
        return ModelResponse(
            tool_calls=(ToolCall("wait-child", "AgentWait", {"agent_id": agent_id}),)
        )

    def _cancel_step(
        self,
        tool_messages: list[Message],
        agent_id: str,
    ) -> ModelResponse:
        if len(tool_messages) == 1:
            return ModelResponse(
                tool_calls=(ToolCall("cancel-child", "AgentCancel", {"agent_id": agent_id}),)
            )
        cancel_outcome = tool_messages[-1].outcome
        assert isinstance(cancel_outcome, ToolSuccess)
        payload = thaw_json_value(cancel_outcome.structured_content)
        assert isinstance(payload, dict) and payload["status"] == "cancelled"
        return ModelResponse(
            (ContentPart.text_part("Parent observed cancelled Child."),),
            finish_reason="stop",
        )


def _terminal_snapshot(record: _AgentRecord, checkpoint: Checkpoint) -> AgentSnapshot:
    state = checkpoint.snapshot.state
    current = record.snapshot
    if isinstance(state, Completed):
        result = "\n".join(part.text for part in state.parts if part.text is not None)
        return AgentSnapshot(
            current.agent_id,
            record.request.description,
            "completed",
            record.request.background,
            result=result,
        )
    if isinstance(state, Suspended) and state.suspension.reason == "agent_cancelled":
        return AgentSnapshot(
            current.agent_id,
            record.request.description,
            "cancelled",
            record.request.background,
            cancellation_requested=True,
        )
    if isinstance(state, Failed):
        error = state.error
    elif isinstance(state, Limited):
        error = ErrorInfo("child_limited", f"Child reached limit: {state.reason.value}")
    else:
        error = ErrorInfo("child_suspended", "Child requires an unsupported external resume.")
    return AgentSnapshot(
        current.agent_id,
        record.request.description,
        "failed",
        record.request.background,
        error=error,
    )


def _agent_id(parent_run_id: str, parent_tool_call_id: str) -> str:
    return (
        f"agent:{len(parent_run_id)}:{parent_run_id}:"
        f"{len(parent_tool_call_id)}:{parent_tool_call_id}"
    )


def _accepted_agent_id(tool_messages: list[Message]) -> str:
    accepted = tool_messages[0].outcome
    assert isinstance(accepted, ToolAccepted)
    return accepted.correlation_id


def _external_completion(request: ModelRequest) -> dict[str, object] | None:
    messages = [
        message
        for message in request.messages
        if message.role == "external" and message.metadata.get("kind") == "agent_completion"
    ]
    if not messages:
        return None
    assert len(messages) == 1
    text = messages[0].parts[0].text
    assert text is not None
    value = json.loads(text.removeprefix("Agent completion:\n"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _harness(tmp_path: Path, mode: _Mode) -> tuple[_EndToEndModel, _InMemoryAgentHost]:
    (tmp_path / "evidence.txt").write_text("E2E evidence from disk\n", encoding="utf-8")
    options = ModelOptions(
        model="inherited-test-model",
        temperature=0,
        seed=7,
        metadata={"profile": "parent-runtime"},
    )
    tool_choice = ToolChoice(allow_parallel_tool_calls=False)
    model = _EndToEndModel(
        mode,
        expected_options=options,
        expected_tool_choice=tool_choice,
    )
    config = _InheritedRuntimeConfig(
        model=model,
        delegable_tools=(ReadTool(tmp_path),),
        limits=RunLimits(
            max_planning_steps=8,
            max_tool_calls=8,
            timeout_seconds=10,
            max_tool_concurrency=1,
            max_tool_batch_size=1,
        ),
        model_options=options,
        tool_choice=tool_choice,
        system_prompt="Shared parent policy inherited by the Child.",
    )
    return model, _InMemoryAgentHost(config)


def _parent_runtime(host: _InMemoryAgentHost) -> Runtime:
    config = host.config
    tools: tuple[Tool, ...] = (
        *config.delegable_tools,
        AgentTool(host),
        AgentGetTool(host),
        AgentWaitTool(host),
        AgentCancelTool(host),
    )
    return Runtime(
        model=config.model,
        tools=ToolRegistry(tools),
        limits=config.limits,
        model_options=config.model_options,
        tool_choice=config.tool_choice,
    )


async def _start_parent(host: _InMemoryAgentHost) -> Checkpoint:
    return (
        await _parent_runtime(host)
        .start(
            (
                Message.system(host.config.system_prompt),
                Message.user("Delegate the evidence inspection."),
            )
        )
        .result()
    )


def test_real_foreground_agent_runs_child_tool_and_resumes_parent(tmp_path: Path) -> None:
    async def scenario() -> None:
        model, host = _harness(tmp_path, "foreground")
        parent_paused = await _start_parent(host)
        assert isinstance(parent_paused.snapshot.state, Suspended)
        wait = extract_agent_wait(parent_paused)
        assert wait.source == "Agent"

        record = host.only_record()
        assert record.snapshot.status == "running"
        assert model.child_entered.is_set()
        assert record.child_context.parent_run_id == parent_paused.snapshot.context.run_id
        assert record.child_context.parent_tool_call_id == "delegate-child"
        assert record.child_context.deadline == parent_paused.snapshot.context.deadline

        model.release_child.set()
        terminal = await host.wait_terminal(wait.agent_id)
        assert terminal.status == "completed"
        assert terminal.result is not None and "E2E evidence from disk" in terminal.result
        child_checkpoint = record.checkpoint
        assert child_checkpoint is not None
        assert isinstance(child_checkpoint.snapshot.state, Completed)
        assert child_checkpoint.snapshot.metrics.planning_steps == 2
        assert child_checkpoint.snapshot.metrics.tool_calls == 1
        assert [message.role for message in child_checkpoint.snapshot.history] == [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert child_checkpoint.snapshot.history[0].parts[0].text == host.config.system_prompt
        assert decode_checkpoint(encode_checkpoint(child_checkpoint)) == child_checkpoint

        restored_parent = decode_checkpoint(encode_checkpoint(parent_paused))
        completed_parent = await host.resume_parent(
            _parent_runtime(host),
            restored_parent,
            wait.agent_id,
        )
        assert isinstance(completed_parent.snapshot.state, Completed)
        parent_text = completed_parent.snapshot.state.parts[0].text
        assert parent_text is not None and "E2E evidence from disk" in parent_text
        assert model.observed_completion is not None
        assert model.observed_completion["status"] == "completed"

    asyncio.run(scenario())


def test_real_background_agent_get_wait_and_resume_parent(tmp_path: Path) -> None:
    async def scenario() -> None:
        model, host = _harness(tmp_path, "background_wait")
        parent_paused = await _start_parent(host)
        assert isinstance(parent_paused.snapshot.state, Suspended)
        wait = extract_agent_wait(parent_paused)
        assert wait.source == "AgentWait"

        record = host.only_record()
        assert record.snapshot.background is True
        assert record.snapshot.status == "running"
        assert (parent_paused.snapshot.context.run_id, "wait-child") in record.waiters
        outcomes = [
            message.outcome for message in parent_paused.snapshot.history if message.role == "tool"
        ]
        assert isinstance(outcomes[0], ToolAccepted)
        assert isinstance(outcomes[1], ToolSuccess)
        assert isinstance(outcomes[2], ToolWaiting)
        get_payload = thaw_json_value(outcomes[1].structured_content)
        assert isinstance(get_payload, dict) and get_payload["status"] == "running"

        model.release_child.set()
        completed_parent = await host.resume_parent(
            _parent_runtime(host),
            parent_paused,
            wait.agent_id,
        )
        assert isinstance(completed_parent.snapshot.state, Completed)
        assert host.records[wait.agent_id].snapshot.status == "completed"
        assert model.observed_completion is not None
        assert model.observed_completion["status"] == "completed"

    asyncio.run(scenario())


def test_real_background_agent_cancel_pauses_active_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        model, host = _harness(tmp_path, "cancel")
        completed_parent = await _start_parent(host)
        assert isinstance(completed_parent.snapshot.state, Completed)

        record = host.only_record()
        assert record.snapshot.status == "cancelled"
        assert record.snapshot.cancellation_requested is True
        assert record.done.is_set()
        assert record.task is not None and record.task.done()
        assert record.checkpoint is not None
        child_state = record.checkpoint.snapshot.state
        assert isinstance(child_state, Suspended)
        assert child_state.suspension.reason == "agent_cancelled"
        assert child_state.suspension.source == "AgentCancel"
        assert model.child_model_cancelled is True
        assert record.checkpoint.snapshot.metrics.tool_calls == 0

        tool_outcomes = [
            message.outcome
            for message in completed_parent.snapshot.history
            if message.role == "tool"
        ]
        assert isinstance(tool_outcomes[0], ToolAccepted)
        assert isinstance(tool_outcomes[1], ToolSuccess)
        cancel_payload = thaw_json_value(tool_outcomes[1].structured_content)
        assert isinstance(cancel_payload, dict)
        assert cancel_payload["status"] == "cancelled"

    asyncio.run(scenario())
