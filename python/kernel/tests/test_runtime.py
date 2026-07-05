from __future__ import annotations

from typing import Any, cast

import pytest
from kernel import AgentState, AgentStatus, ContentPart, Message, RunSnapshot, RuntimeContext


def user_text(text: str) -> Message:
    return Message.user([ContentPart.text_part(text)])


def test_runtime_context_from_dict_rejects_missing_required_fields() -> None:
    payload = RuntimeContext(run_id="run-1").to_dict()
    del payload["deadline"]

    with pytest.raises(KeyError):
        RuntimeContext.from_dict(payload)


def test_runtime_context_from_dict_rejects_invalid_required_types() -> None:
    payload = RuntimeContext(run_id="run-1").to_dict()
    payload["sequence"] = True

    with pytest.raises(TypeError, match="sequence"):
        RuntimeContext.from_dict(payload)

    payload = RuntimeContext(run_id="run-1").to_dict()
    payload["metadata"] = None

    with pytest.raises(TypeError, match="metadata"):
        RuntimeContext.from_dict(payload)

    with pytest.raises(TypeError, match="run_id"):
        RuntimeContext(run_id=cast(Any, 123))

    with pytest.raises(TypeError, match="sequence"):
        RuntimeContext(run_id="run-1").sequence = cast(Any, True)


def test_runtime_context_round_trips_child_run_relation() -> None:
    context = RuntimeContext(
        run_id="child-run",
        parent_run_id="parent-run",
        parent_tool_call_id="call-1",
        run_kind="subagent",
    )

    restored = RuntimeContext.from_dict(context.to_dict())

    assert restored.parent_run_id == "parent-run"
    assert restored.parent_tool_call_id == "call-1"
    assert restored.run_kind == "subagent"


def test_runtime_context_rejects_orphan_child_run_fields() -> None:
    with pytest.raises(ValueError, match="parent_run_id"):
        RuntimeContext(run_id="child-run", parent_tool_call_id="call-1")

    with pytest.raises(ValueError, match="parent_run_id"):
        RuntimeContext(run_id="child-run", run_kind="subagent")


def test_run_snapshot_from_dict_rejects_unknown_fields() -> None:
    payload = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")]),
        context=RuntimeContext(run_id="run-1"),
    ).to_dict()
    payload["legacy"] = True

    with pytest.raises(ValueError, match="unknown"):
        RunSnapshot.from_dict(payload)

    payload = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")]),
        context=RuntimeContext(run_id="run-1"),
    ).to_dict()
    payload["state"]["legacy_resume_token"] = "old"

    with pytest.raises(ValueError, match="agent state"):
        RunSnapshot.from_dict(payload)

    payload = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")]),
        context=RuntimeContext(run_id="run-1"),
    ).to_dict()
    payload["context"]["old_checkpoint_id"] = "old"

    with pytest.raises(ValueError, match="runtime context"):
        RunSnapshot.from_dict(payload)

    payload = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")]),
        context=RuntimeContext(run_id="run-1"),
    ).to_dict()
    payload["state"] = []

    with pytest.raises(TypeError, match="run snapshot state"):
        RunSnapshot.from_dict(payload)

    with pytest.raises(TypeError, match="run snapshot state"):
        RunSnapshot(state=cast(Any, []), context=RuntimeContext(run_id="run-1"))


def test_run_snapshot_constructor_copies_state_and_context() -> None:
    state = AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")])
    context = RuntimeContext(run_id="run-1", metadata={"tenant": "acme"})
    snapshot = RunSnapshot(state=state, context=context)

    state.messages.append(user_text("mutated"))
    context.metadata = {"tenant": "other"}

    assert [message.text for message in snapshot.state.messages] == ["hello"]
    assert snapshot.context.metadata == {"tenant": "acme"}


def test_run_snapshot_accessors_return_copies() -> None:
    snapshot = RunSnapshot(
        state=AgentState(status=AgentStatus.PLANNING, messages=[user_text("hello")]),
        context=RuntimeContext(run_id="run-1", metadata={"tenant": "acme"}),
    )

    snapshot.state.messages.append(user_text("mutated"))
    snapshot.context.metadata = {"tenant": "other"}

    assert [message.text for message in snapshot.state.messages] == ["hello"]
    assert snapshot.context.metadata == {"tenant": "acme"}


def test_agent_state_constructor_and_from_dict_reject_invalid_shape() -> None:
    with pytest.raises(TypeError, match="status"):
        AgentState(status=cast(Any, "planning"), messages=[])

    with pytest.raises(TypeError, match="messages"):
        AgentState(status=AgentStatus.PLANNING, messages=[cast(Any, {"role": "user"})])

    with pytest.raises(TypeError, match="iterations"):
        AgentState(
            status=AgentStatus.PLANNING,
            messages=[user_text("hello")],
            iterations=cast(Any, True),
        )

    payload = AgentState(
        status=AgentStatus.PLANNING,
        messages=[user_text("hello")],
    ).to_dict()
    payload["messages"] = [None]

    with pytest.raises(TypeError, match="agent state message"):
        AgentState.from_dict(payload)
