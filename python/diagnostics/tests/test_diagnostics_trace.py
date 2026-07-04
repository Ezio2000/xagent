from __future__ import annotations

from typing import Any, cast

import pytest
from diagnostics import ReplayError, RunTrace, TraceStep, TraceStepKinds, replay_trace
from kernel import AgentStatus


def trace_state_payload(
    status: str,
    *,
    message_roles: tuple[str, ...] = ("user",),
    iterations: int = 0,
    total_tool_calls: int = 0,
    final_part_count: int = 0,
    context_sequence: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "message_roles": list(message_roles),
        "message_count": len(message_roles),
        "pending_tool_call_ids": [],
        "iterations": iterations,
        "total_tool_calls": total_tool_calls,
        "total_usage": None,
        "final_part_count": final_part_count,
        "error": None,
        "pause": None,
    }
    if context_sequence is not None:
        payload["context_sequence"] = context_sequence
    return payload


def transition_payload(from_status: str, to_status: str, *, iterations: int) -> dict[str, object]:
    return {
        "from": from_status,
        "to": to_status,
        "iterations": iterations,
        "total_tool_calls": 0,
        "total_usage": None,
        "error": None,
        "pause": None,
    }


def minimal_completed_trace() -> RunTrace:
    return RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload("planning"),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.MODEL_CALL,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={"iteration": 1},
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.MODEL_RESULT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={
                    "part_count": 1,
                    "part_types": ["text"],
                    "text_length": 4,
                    "tool_call_count": 0,
                    "has_tool_calls": False,
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=transition_payload("planning", "completed", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant"),
                    iterations=1,
                    final_part_count=1,
                    context_sequence=5,
                ),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "part_count": 1,
                    "part_types": ["text"],
                    "text_length": 4,
                    "metadata_keys": [],
                },
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "state": trace_state_payload(
                        "completed",
                        message_roles=("user", "assistant"),
                        iterations=1,
                        final_part_count=1,
                    )
                },
            ),
        ],
    )


def test_replay_accepts_minimal_completed_trace() -> None:
    result = replay_trace(minimal_completed_trace())

    assert result.valid is True
    assert result.final_status is AgentStatus.COMPLETED


def test_replay_diagnostic_mode_reports_invalid_trace_without_raising() -> None:
    trace = RunTrace(run_id="run-1", steps=minimal_completed_trace().steps[:-1])

    result = replay_trace(trace, strict=False)

    assert result.valid is False
    assert result.message is not None
    assert "run_completed" in result.message


def test_trace_payload_is_immutable_but_round_trips_to_plain_dict() -> None:
    trace = minimal_completed_trace()

    with pytest.raises(TypeError):
        cast(Any, trace.steps[0].payload)["status"] = "failed"

    restored = RunTrace.from_dict(trace.to_dict())
    assert restored.to_dict() == trace.to_dict()


def test_replay_rejects_out_of_order_step_ids() -> None:
    steps = list(minimal_completed_trace().steps)
    steps[1] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.MODEL_CALL,
        before_status=AgentStatus.PLANNING,
        after_status=AgentStatus.PLANNING,
        payload={"iteration": 1},
    )

    with pytest.raises(ReplayError, match="increasing"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))
