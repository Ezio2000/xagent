# pyright: reportPrivateUsage=false
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from jharness.kernel import (
    ContentPart,
    DeltaSink,
    ErrorInfo,
    Message,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    Planning,
    RunContext,
    Runtime,
    Suspended,
    Suspension,
    SuspensionView,
    TaskRef,
    ToolBatchFact,
    ToolCall,
    ToolOutcomeKind,
    ToolsPending,
    ToolSuccess,
    ToolWaiting,
)
from jharness.tools.agent import AgentSnapshot, AgentWaitRequest, agent_completion_message
from jharness.tools.agent import response as agent_response
from jharness.tools.agent._schema import AgentContractError, build_wait_id


class _Model(Model):
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
        return ModelResponse((ContentPart.text_part("done"),), finish_reason="stop")


def _running(*, background: bool = False) -> AgentSnapshot:
    return AgentSnapshot("agent-1", "Inspect", "running", background)


def _request(
    *,
    source: str = "Agent",
    snapshot: AgentSnapshot | None = None,
    wait_id: str = "wait-1",
    tool_call_id: str = "call-1",
    limit: object = 100,
) -> AgentWaitRequest:
    return AgentWaitRequest(
        wait_id,
        cast(Any, source),
        tool_call_id,
        _running() if snapshot is None else snapshot,
        cast(Any, limit),
        100,
        100,
        100,
        100,
    )


def _waiting(
    snapshot: AgentSnapshot | None = None,
    *,
    task: TaskRef | None | object = ...,
) -> ToolWaiting:
    selected = _running() if snapshot is None else snapshot
    selected_task = (
        TaskRef(selected.agent_id, selected.status) if task is ... else cast(TaskRef | None, task)
    )
    return ToolWaiting(
        (ContentPart.text_part("waiting"),),
        task=selected_task,
        structured_content={
            "agent_id": selected.agent_id,
            "status": selected.status,
            "description": selected.description,
            "background": selected.background,
            "cancellation_requested": selected.cancellation_requested,
        },
    )


def _waiting_fact(call_id: str = "call-1") -> ToolBatchFact:
    return ToolBatchFact(
        time.time(),
        "batch-1",
        (call_id,),
        False,
        (ToolOutcomeKind.WAITING,),
        SuspensionView("agent_completion", "Agent", "wait-1", ()),
    )


def _checkpoint_like(*, fact: object, history: tuple[Message, ...]) -> object:
    return SimpleNamespace(fact=fact, snapshot=SimpleNamespace(history=history))


def _metadata(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "agent_id": "agent-1",
        "schema_version": 1,
        "tool_call_id": "call-1",
        "max_agent_id_chars": 100,
        "max_description_chars": 100,
        "max_result_chars": 100,
        "max_error_code_chars": 100,
        "max_error_message_chars": 100,
    }
    value.update(changes)
    return value


def _state_like(
    *,
    metadata: dict[str, object] | None = None,
    wait_id: object = "agent-wait:5:run-1:6:call-1",
) -> object:
    suspension = SimpleNamespace(
        metadata=_metadata() if metadata is None else metadata,
        wait_id=wait_id,
    )
    return SimpleNamespace(suspension=suspension)


def test_agent_wait_request_constructor_defenses() -> None:
    assert _request().agent_id == "agent-1"
    with pytest.raises(ValueError, match="wait_id"):
        _request(wait_id="")
    with pytest.raises(ValueError, match="source"):
        _request(source="Other")
    with pytest.raises(ValueError, match="tool_call_id"):
        _request(tool_call_id="")
    with pytest.raises(TypeError, match="AgentSnapshot"):
        AgentWaitRequest("wait", "Agent", "call", cast(Any, object()), 1, 1, 1, 1, 1)
    terminal = AgentSnapshot("agent-1", "Inspect", "completed", False, result="done")
    with pytest.raises(ValueError, match="non-terminal"):
        _request(snapshot=terminal)
    with pytest.raises(ValueError, match="positive integer"):
        _request(limit=0)


def test_agent_completion_message_defenses_and_terminal_variants() -> None:
    request = _request()
    with pytest.raises(TypeError, match="AgentWaitRequest"):
        agent_completion_message(cast(Any, object()), _running())
    with pytest.raises(TypeError, match="AgentSnapshot"):
        agent_completion_message(request, cast(Any, object()))
    with pytest.raises(ValueError, match="terminal"):
        agent_completion_message(request, _running())
    with pytest.raises(ValueError, match="agent_id"):
        agent_completion_message(
            request,
            AgentSnapshot("other", "Inspect", "completed", False, result="done"),
        )
    with pytest.raises(ValueError, match="description"):
        agent_completion_message(
            request,
            AgentSnapshot("agent-1", "Other", "completed", False, result="done"),
        )
    with pytest.raises(ValueError, match="background"):
        agent_completion_message(
            request,
            AgentSnapshot("agent-1", "Inspect", "completed", True, result="done"),
        )

    failed = AgentSnapshot(
        "agent-1",
        "Inspect",
        "failed",
        False,
        error=ErrorInfo("child_failed", "Child failed."),
    )
    message = agent_completion_message(request, failed)
    assert message.metadata == {
        "agent_id": "agent-1",
        "kind": "agent_completion",
        "status": "failed",
        "wait_id": "wait-1",
    }
    with pytest.raises(AgentContractError, match="character limit"):
        agent_completion_message(
            _request(limit=1),
            AgentSnapshot("agent-1", "Inspect", "completed", False, result="too long"),
        )


def test_resume_agent_rejects_invalid_runtime_and_stream() -> None:
    completed = AgentSnapshot("agent-1", "Inspect", "completed", False, result="done")
    with pytest.raises(TypeError, match="runtime"):
        agent_response.resume_agent(cast(Any, object()), cast(Any, object()), completed)
    with pytest.raises(TypeError, match="stream"):
        agent_response.resume_agent(
            Runtime(model=_Model()),
            cast(Any, object()),
            completed,
            stream=cast(Any, "yes"),
        )


def test_extract_agent_wait_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="Checkpoint"):
        agent_response.extract_agent_wait(cast(Any, object()))


def test_current_waiting_outcome_defenses() -> None:
    call = ToolCall(
        "call-1",
        "Agent",
        {"description": "Inspect", "prompt": "Do it"},
    )
    assistant = Message.assistant(tool_calls=(call,))
    waiting_message = Message.tool("call-1", _waiting())

    with pytest.raises(ValueError, match="waiting tool batch"):
        agent_response._current_waiting_outcome(
            cast(Any, _checkpoint_like(fact=object(), history=())), "call-1", "Agent"
        )
    wrong_fact = ToolBatchFact(
        time.time(),
        "batch",
        ("call-1",),
        False,
        (ToolOutcomeKind.SUCCESS,),
        None,
    )
    with pytest.raises(ValueError, match="current waiting tool call"):
        agent_response._current_waiting_outcome(
            cast(Any, _checkpoint_like(fact=wrong_fact, history=())), "call-1", "Agent"
        )
    with pytest.raises(ValueError, match="missing"):
        agent_response._current_waiting_outcome(
            cast(Any, _checkpoint_like(fact=_waiting_fact(), history=())), "call-1", "Agent"
        )
    with pytest.raises(ValueError, match="trailing tool"):
        agent_response._current_waiting_outcome(
            cast(
                Any,
                _checkpoint_like(
                    fact=_waiting_fact(),
                    history=(assistant, Message.user("not a tool")),
                ),
            ),
            "call-1",
            "Agent",
        )
    with pytest.raises(ValueError, match="only call"):
        agent_response._current_waiting_outcome(
            cast(
                Any,
                _checkpoint_like(
                    fact=_waiting_fact(),
                    history=(Message.user("not assistant"), waiting_message),
                ),
            ),
            "call-1",
            "Agent",
        )
    wrong_call = Message.assistant(
        tool_calls=(ToolCall("call-1", "AgentWait", {"agent_id": "agent-1"}),)
    )
    with pytest.raises(ValueError, match="wrong identity"):
        agent_response._current_waiting_outcome(
            cast(
                Any,
                _checkpoint_like(
                    fact=_waiting_fact(),
                    history=(wrong_call, waiting_message),
                ),
            ),
            "call-1",
            "Agent",
        )
    success_message = Message.tool(
        "call-1",
        ToolSuccess((ContentPart.text_part("done"),)),
    )
    with pytest.raises(ValueError, match="ToolWaiting"):
        agent_response._current_waiting_outcome(
            cast(
                Any,
                _checkpoint_like(
                    fact=_waiting_fact(),
                    history=(assistant, success_message),
                ),
            ),
            "call-1",
            "Agent",
        )
    outcome, observed_call = agent_response._current_waiting_outcome(
        cast(
            Any,
            _checkpoint_like(
                fact=_waiting_fact(),
                history=(assistant, waiting_message),
            ),
        ),
        "call-1",
        "Agent",
    )
    assert isinstance(outcome, ToolWaiting)
    assert observed_call == call


def test_agent_suspension_defenses() -> None:
    with pytest.raises(ValueError, match="must be suspended"):
        agent_response._agent_suspension(
            cast(Any, SimpleNamespace(snapshot=SimpleNamespace(state=Planning())))
        )
    wrong = Suspended(Planning(), Suspension("other", "host"))
    with pytest.raises(ValueError, match="not an Agent"):
        agent_response._agent_suspension(
            cast(Any, SimpleNamespace(snapshot=SimpleNamespace(state=wrong)))
        )
    pending = Suspended(
        ToolsPending((ToolCall("next", "Read", {}),)),
        Suspension("agent_completion", "Agent", "wait"),
    )
    with pytest.raises(ValueError, match="resume to Planning"):
        agent_response._agent_suspension(
            cast(Any, SimpleNamespace(snapshot=SimpleNamespace(state=pending)))
        )
    state = Suspended(Planning(), Suspension("agent_completion", "AgentWait", "wait"))
    observed, source = agent_response._agent_suspension(
        cast(Any, SimpleNamespace(snapshot=SimpleNamespace(state=state)))
    )
    assert observed == state
    assert source == "AgentWait"


def test_suspension_contract_defenses() -> None:
    checkpoint = cast(
        Any,
        SimpleNamespace(snapshot=SimpleNamespace(context=RunContext("run-1", time.time()))),
    )
    with pytest.raises(ValueError, match="unexpected fields"):
        agent_response._suspension_contract(
            checkpoint,
            cast(Any, _state_like(metadata={"agent_id": "agent-1"})),
        )
    for schema_version in ("1", 2):
        with pytest.raises(ValueError, match="schema_version"):
            agent_response._suspension_contract(
                checkpoint,
                cast(Any, _state_like(metadata=_metadata(schema_version=schema_version))),
            )
    for field in ("tool_call_id", "agent_id"):
        with pytest.raises(ValueError, match=field):
            agent_response._suspension_contract(
                checkpoint,
                cast(Any, _state_like(metadata=_metadata(**{field: ""}))),
            )
    with pytest.raises(ValueError, match="wait_id"):
        agent_response._suspension_contract(
            checkpoint,
            cast(Any, _state_like(wait_id="")),
        )
    with pytest.raises(ValueError, match="does not match"):
        agent_response._suspension_contract(
            checkpoint,
            cast(Any, _state_like(wait_id="wrong")),
        )
    for bad_limit in (True, 0):
        with pytest.raises(ValueError, match="max_result_chars"):
            agent_response._suspension_contract(
                checkpoint,
                cast(
                    Any,
                    _state_like(metadata=_metadata(max_result_chars=bad_limit)),
                ),
            )

    wait_id, tool_call_id, agent_id, limits = agent_response._suspension_contract(
        checkpoint,
        cast(Any, _state_like()),
    )
    assert wait_id == build_wait_id("run-1", "call-1")
    assert tool_call_id == "call-1"
    assert agent_id == "agent-1"
    assert limits["max_result_chars"] == 100


def test_validate_waiting_identity_and_call_defenses() -> None:
    snapshot = _running()
    agent_call = ToolCall(
        "call-1",
        "Agent",
        {"description": "Inspect", "prompt": "Do it"},
    )
    waiting = _waiting(snapshot)
    with pytest.raises(ValueError, match="suspension agent_id"):
        agent_response._validate_waiting_identity(waiting, agent_call, snapshot, "other")
    with pytest.raises(ValueError, match="task reference"):
        agent_response._validate_waiting_identity(
            _waiting(snapshot, task=None), agent_call, snapshot, "agent-1"
        )
    for task in (TaskRef("other", "running"), TaskRef("agent-1", "queued")):
        with pytest.raises(ValueError, match="does not match"):
            agent_response._validate_waiting_identity(
                _waiting(snapshot, task=task), agent_call, snapshot, "agent-1"
            )

    invalid_agent_arguments = (
        {"description": "Inspect", "prompt": "Do it", "extra": True},
        {"description": "Other", "prompt": "Do it"},
        {"description": "Inspect", "prompt": 1},
        {"description": "Inspect", "prompt": ""},
        {"description": "Inspect", "prompt": "Do it", "background": True},
    )
    patterns = ("unexpected", "description", "prompt", "prompt", "background")
    for arguments, pattern in zip(invalid_agent_arguments, patterns, strict=True):
        with pytest.raises(ValueError, match=pattern):
            agent_response._validate_waiting_identity(
                waiting,
                ToolCall("call-1", "Agent", arguments),
                snapshot,
                "agent-1",
            )
    agent_response._validate_waiting_identity(waiting, agent_call, snapshot, "agent-1")

    for arguments, pattern in (
        ({"agent_id": "agent-1", "extra": True}, "only agent_id"),
        ({"agent_id": "other"}, "do not match"),
    ):
        with pytest.raises(ValueError, match=pattern):
            agent_response._validate_waiting_identity(
                waiting,
                ToolCall("call-1", "AgentWait", arguments),
                snapshot,
                "agent-1",
            )
    agent_response._validate_waiting_identity(
        waiting,
        ToolCall("call-1", "AgentWait", {"agent_id": "agent-1"}),
        snapshot,
        "agent-1",
    )


def test_waiting_snapshot_and_scalar_helper_defenses() -> None:
    limits = {
        "max_agent_id_chars": 100,
        "max_description_chars": 100,
        "max_result_chars": 100,
        "max_error_code_chars": 100,
        "max_error_message_chars": 100,
    }
    payload: dict[str, Any] = {
        "agent_id": "agent-1",
        "status": "running",
        "description": "Inspect",
        "background": False,
        "cancellation_requested": False,
    }
    assert agent_response._waiting_snapshot(payload, limits) == _running()
    with pytest.raises(ValueError, match="non-terminal"):
        agent_response._waiting_snapshot({**payload, "status": "completed"}, limits)
    with pytest.raises(ValueError, match="invalid"):
        agent_response._waiting_snapshot({**payload, "agent_id": 1}, limits)
    with pytest.raises(ValueError, match="canonical"):
        agent_response._waiting_snapshot({**payload, "extra": True}, limits)

    with pytest.raises(ValueError, match="object"):
        agent_response._mapping([], "value")
    with pytest.raises(ValueError, match="keys"):
        agent_response._mapping(cast(Any, {1: "value"}), "value")
    assert agent_response._mapping({"key": "value"}, "value") == {"key": "value"}

    for value in (1, ""):
        with pytest.raises(ValueError, match="non-empty string"):
            agent_response._non_empty_string(value, "value")
    assert agent_response._non_empty_string("value", "value") == "value"
    for value in (True, "1"):
        with pytest.raises(ValueError, match="integer"):
            agent_response._exact_int(value, "value")
    assert agent_response._exact_int(1, "value") == 1
    with pytest.raises(ValueError, match="positive"):
        agent_response._positive_int(0, "value")
    assert agent_response._positive_int(1, "value") == 1
