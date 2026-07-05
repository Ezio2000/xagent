from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
from diagnostics import ReplayError, RunTrace, TraceStep, TraceStepKinds, replay_trace
from kernel import AgentLoop, AgentStatus, EventTypes, ModelRequest, ModelResponse, RuntimeContext
from prompting import user_text


class OneShotModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        return ModelResponse.text("done")


async def test_agent_result_trace_is_wire_payload_not_trace_object() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])

    assert isinstance(result.trace, Mapping)
    assert not isinstance(result.trace, RunTrace)
    assert RunTrace.from_dict(result.trace).steps[0].kind == TraceStepKinds.RUN_STARTED
    with pytest.raises(TypeError, match="immutable"):
        cast(dict[str, Any], result.trace)["run_id"] = "changed"


async def test_agent_result_trace_payload_replays() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])
    assert result.trace is not None
    trace = RunTrace.from_dict(result.trace)
    replay = replay_trace(trace)

    assert replay.valid is True
    assert replay.final_status == result.status


async def test_agent_result_trace_payload_omits_raw_context_metadata_values() -> None:
    result = await AgentLoop(model=OneShotModel()).run(
        [user_text("x")],
        context=RuntimeContext(metadata={"secret": "tenant"}),
    )
    assert result.trace is not None
    payload: dict[str, Any] = RunTrace.from_dict(result.trace).to_dict()

    assert "tenant" not in str(payload)


async def test_run_trace_from_events_replays() -> None:
    events = [event async for event in AgentLoop(model=OneShotModel()).run_events([user_text("x")])]
    trace = RunTrace.from_events(events[0].run_id, events)

    result = replay_trace(trace)

    assert result.valid is True
    assert result.final_status is AgentStatus.COMPLETED


async def test_run_trace_from_dict_and_replay_reject_raw_payload_metadata() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])
    assert result.trace is not None
    payload = RunTrace.from_dict(result.trace).to_dict()
    final_step = next(step for step in payload["steps"] if step["kind"] == TraceStepKinds.FINAL)
    final_step["payload"]["metadata"] = {"secret": "value"}

    with pytest.raises(ValueError, match="final payload has unknown field"):
        RunTrace.from_dict(payload)

    corrupted = RunTrace(
        run_id=str(payload["run_id"]),
        steps=[TraceStep.from_dict(step) for step in cast(list[dict[str, Any]], payload["steps"])],
        metadata=cast(dict[str, Any], payload["metadata"]),
        schema_version=str(payload["schema_version"]),
    )

    with pytest.raises(ReplayError, match="final payload has unknown field"):
        replay_trace(corrupted)


async def test_replay_rejects_corrupted_terminal_tail() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])
    assert result.trace is not None
    payload = RunTrace.from_dict(result.trace).to_dict()
    steps = list(payload["steps"])
    steps[-2] = dict(steps[-2])
    steps[-2]["kind"] = TraceStepKinds.TOOL_CALL
    steps[-2]["payload"] = {
        "id": "call-1",
        "name": "tool",
        "mode": "execute",
        "batch_id": "batch-1",
        "parallel": False,
        "index": 0,
        "implementation_invoked": True,
    }
    payload["steps"] = steps
    corrupted = RunTrace.from_dict(payload)

    with pytest.raises(ReplayError, match="tool_call is only valid"):
        replay_trace(corrupted)


async def test_replay_diagnostic_mode_reports_invalid_trace() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])
    assert result.trace is not None
    payload: dict[str, Any] = RunTrace.from_dict(result.trace).to_dict()
    payload["steps"][0]["after_status"] = "executing_tools"

    diagnostic = replay_trace(RunTrace.from_dict(payload), strict=False)

    assert diagnostic.valid is False
    assert diagnostic.message is not None


async def test_model_delta_trace_is_not_durable_replay_input() -> None:
    result = await AgentLoop(model=OneShotModel()).run([user_text("x")])
    assert result.trace is not None
    payload = RunTrace.from_dict(result.trace).to_dict()
    checkpoint_index = next(
        index
        for index, step in enumerate(payload["steps"])
        if step["kind"] == TraceStepKinds.CHECKPOINT
    )
    checkpoint_step_id = payload["steps"][checkpoint_index]["step_id"]
    delta_step = {
        "step_id": checkpoint_step_id,
        "kind": TraceStepKinds.MODEL_DELTA,
        "before_status": None,
        "after_status": None,
        "references": {"event_sequence": 999, "event_type": EventTypes.MODEL_DELTA},
        "payload": {
            "kind": "text_delta",
            "index": 0,
            "text_delta_length": 7,
            "part_type": "text",
        },
        "schema_version": "v0",
    }
    for step in payload["steps"][checkpoint_index:]:
        if step["step_id"] >= checkpoint_step_id:
            step["step_id"] += 1
    payload["steps"].insert(checkpoint_index, delta_step)
    corrupted_checkpoint = payload["steps"][checkpoint_index + 1]
    corrupted_checkpoint["payload"]["message_roles"].append("assistant")
    corrupted_checkpoint["payload"]["message_count"] += 1

    with pytest.raises(ReplayError, match="model_delta requires an open model_call"):
        replay_trace(RunTrace.from_dict(payload))
