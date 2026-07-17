from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, cast

import pytest

from jharness.kernel import (
    Checkpoint,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    Failed,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    Suspended,
    ToolAccepted,
    ToolCall,
    ToolChoice,
    ToolSuccess,
)
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint
from jharness.toolkit import ToolRegistry
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


class _Backend:
    def __init__(self, snapshot: AgentSnapshot) -> None:
        self.snapshot = snapshot
        self.start_calls: list[tuple[AgentRequest, RunContext, str]] = []
        self.get_calls: list[tuple[str, RunContext]] = []
        self.wait_calls: list[tuple[str, RunContext, str]] = []
        self.cancel_calls: list[tuple[str, RunContext, str]] = []

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        self.start_calls.append((request, parent, parent_tool_call_id))
        return self.snapshot

    async def get(self, agent_id: str, *, requester: RunContext) -> AgentSnapshot:
        self._check_id(agent_id)
        self.get_calls.append((agent_id, requester))
        return self.snapshot

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        self._check_id(agent_id)
        self.wait_calls.append((agent_id, requester, requester_tool_call_id))
        return self.snapshot

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        self._check_id(agent_id)
        self.cancel_calls.append((agent_id, requester, requester_tool_call_id))
        if self.snapshot.status in {"queued", "running"}:
            self.snapshot = replace(self.snapshot, cancellation_requested=True)
        return self.snapshot

    def _check_id(self, agent_id: str) -> None:
        if agent_id != self.snapshot.agent_id:
            raise AgentBackendError("agent_not_found", "Agent was not found.")


def _completion_payload(request: ModelRequest) -> dict[str, Any] | None:
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
    prefix = "Agent completion:\n"
    assert text.startswith(prefix)
    value = json.loads(text.removeprefix(prefix))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


class _WaitingAgentModel(Model):
    def __init__(self, tool_name: str, arguments: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.requests: list[ModelRequest] = []
        self.observed_completion: dict[str, Any] | None = None

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
        del context, stream, emit_delta
        self.requests.append(request)
        completion = _completion_payload(request)
        if completion is None:
            return ModelResponse(
                tool_calls=(ToolCall("agent-call", self.tool_name, self.arguments),)
            )
        self.observed_completion = completion
        return ModelResponse(
            (ContentPart.text_part(f"Observed {completion['status']}"),),
            finish_reason="stop",
        )


class _BackgroundAgentModel(Model):
    def __init__(self) -> None:
        self.accepted: ToolAccepted | None = None

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
        del context, stream, emit_delta
        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        "agent-background",
                        "Agent",
                        {
                            "description": "Inspect authentication",
                            "prompt": "Inspect authentication and report findings.",
                            "background": True,
                        },
                    ),
                )
            )
        outcome = tool_messages[-1].outcome
        assert isinstance(outcome, ToolAccepted)
        self.accepted = outcome
        return ModelResponse(
            (ContentPart.text_part(f"Started {outcome.correlation_id}"),),
            finish_reason="stop",
        )


class _ManagementModel(Model):
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.observed: list[dict[str, Any]] = []

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
        del context, stream, emit_delta
        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            return ModelResponse(
                tool_calls=(ToolCall("get-agent", "AgentGet", {"agent_id": self.agent_id}),)
            )
        latest = tool_messages[-1].outcome
        assert isinstance(latest, ToolSuccess)
        payload = cast(dict[str, Any], latest.structured_content)
        if len(tool_messages) == 1:
            self.observed.append(payload)
            return ModelResponse(
                tool_calls=(ToolCall("cancel-agent", "AgentCancel", {"agent_id": self.agent_id}),)
            )
        self.observed.append(payload)
        return ModelResponse(
            (ContentPart.text_part("Cancellation requested"),),
            finish_reason="stop",
        )


class _MultipleAgentCallsModel(Model):
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
        arguments = {"description": "One", "prompt": "Do one task."}
        return ModelResponse(
            tool_calls=(
                ToolCall("agent-one", "Agent", arguments),
                ToolCall("agent-two", "Agent", arguments),
            )
        )


async def _collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


def _runtime(model: Model, *tools: object) -> Runtime:
    return Runtime(
        model=model,
        tools=ToolRegistry(cast(Any, tools)),
        tool_choice=ToolChoice(allow_parallel_tool_calls=False),
    )


@pytest.mark.parametrize(
    ("tool_name", "background"),
    [("Agent", False), ("AgentWait", True)],
)
def test_runtime_agent_wait_wire_roundtrip_and_fresh_runtime_resume(
    tool_name: str,
    background: bool,
) -> None:
    running = AgentSnapshot("agent-1", "Inspect authentication", "running", background)
    backend = _Backend(running)
    arguments: dict[str, Any]
    tool: object
    if tool_name == "Agent":
        arguments = {
            "description": running.description,
            "prompt": "Inspect authentication and report findings.",
        }
        tool = AgentTool(backend)
    else:
        arguments = {"agent_id": running.agent_id}
        tool = AgentWaitTool(backend)

    initial_model = _WaitingAgentModel(tool_name, arguments)
    paused, events = asyncio.run(
        _collect(_runtime(initial_model, tool).start((Message.user("Delegate work"),)))
    )
    assert isinstance(paused.snapshot.state, Suspended)
    assert paused.snapshot.state.resume_to.kind == "planning"
    assert [event.kind for event in events].count(EventKind.TOOL_STARTED) == 1
    assert [event.kind for event in events].count(EventKind.TOOL_FINISHED) == 1

    restored = decode_checkpoint(json.loads(json.dumps(encode_checkpoint(paused))))
    wait_request = extract_agent_wait(restored)
    assert wait_request.agent_id == running.agent_id
    assert wait_request.source == tool_name

    completion = AgentSnapshot(
        running.agent_id,
        running.description,
        "completed",
        background,
        result="Authentication review complete.",
    )
    resumed_model = _WaitingAgentModel(tool_name, arguments)
    completed, resume_events = asyncio.run(
        _collect(resume_agent(_runtime(resumed_model, tool), restored, completion))
    )
    assert completed.snapshot.status == "completed"
    assert resumed_model.observed_completion is not None
    assert resumed_model.observed_completion["result"] == "Authentication review complete."
    assert EventKind.TOOL_STARTED not in [event.kind for event in resume_events]
    assert [message.role for message in completed.snapshot.history] == [
        "user",
        "assistant",
        "tool",
        "external",
        "assistant",
    ]


def test_runtime_background_agent_is_accepted_and_parent_continues() -> None:
    backend = _Backend(AgentSnapshot("agent-bg", "Inspect authentication", "queued", True))
    model = _BackgroundAgentModel()
    checkpoint = asyncio.run(
        _runtime(model, AgentTool(backend)).start((Message.user("Start a review"),)).result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert model.accepted is not None
    assert model.accepted.correlation_id == "agent-bg"
    assert model.accepted.task is not None
    assert model.accepted.task.id == "agent-bg"
    assert model.accepted.task.status == "queued"
    assert len(backend.start_calls) == 1
    request, parent, parent_tool_call_id = backend.start_calls[0]
    assert request.background is True
    assert parent.run_id == checkpoint.snapshot.context.run_id
    assert parent_tool_call_id == "agent-background"


def test_runtime_agent_get_then_cancel_observes_acknowledged_cancellation() -> None:
    backend = _Backend(AgentSnapshot("agent-manage", "Manage me", "running", True))
    model = _ManagementModel("agent-manage")
    checkpoint = asyncio.run(
        _runtime(model, AgentGetTool(backend), AgentCancelTool(backend))
        .start((Message.user("Inspect and cancel the agent"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert [payload["cancellation_requested"] for payload in model.observed] == [False, True]
    assert len(backend.get_calls) == 1
    assert len(backend.cancel_calls) == 1
    assert backend.cancel_calls[0][2] == "cancel-agent"


def test_runtime_disallowed_multiple_foreground_agents_fail_before_start() -> None:
    backend = _Backend(AgentSnapshot("unused", "One", "running", False))
    checkpoint, events = asyncio.run(
        _collect(
            _runtime(_MultipleAgentCallsModel(), AgentTool(backend)).start(
                (Message.user("Delegate one task"),)
            )
        )
    )

    state = checkpoint.snapshot.state
    assert isinstance(state, Failed)
    assert state.error.code == "model_protocol_error"
    assert "disallowed parallel tool calls" in state.error.message
    assert not backend.start_calls
    kinds = [event.kind for event in events]
    assert EventKind.TOOL_STARTED not in kinds
    assert EventKind.TOOL_FINISHED not in kinds


def test_resume_agent_rejects_wrong_or_nonterminal_completion() -> None:
    running = AgentSnapshot("agent-1", "Inspect authentication", "running", False)
    backend = _Backend(running)
    tool = AgentTool(backend)
    paused = asyncio.run(
        _runtime(
            _WaitingAgentModel(
                "Agent",
                {
                    "description": running.description,
                    "prompt": "Inspect authentication.",
                },
            ),
            tool,
        )
        .start((Message.user("Delegate"),))
        .result()
    )
    runtime = _runtime(_WaitingAgentModel("Agent", {}), tool)

    with pytest.raises(ValueError, match="terminal"):
        resume_agent(runtime, paused, running)
    wrong = AgentSnapshot(
        "agent-other",
        running.description,
        "completed",
        False,
        result="Done",
    )
    with pytest.raises(ValueError, match="agent_id"):
        resume_agent(runtime, paused, wrong)
