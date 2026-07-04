from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

import pytest

from agent_runtime import (
    AgentEvent,
    AgentLoop,
    AgentStatus,
    EventTypes,
    Message,
    ModelRequest,
    ModelResponse,
    ReplayError,
    RuntimeContext,
    RunTrace,
    TraceStep,
    TraceStepKinds,
    replay_trace,
)


class OneShotModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        return ModelResponse.text("done")


def trace_state_payload(
    status: str,
    *,
    message_roles: Sequence[str] = ("user",),
    pending_tool_call_ids: Sequence[str] = (),
    iterations: int = 0,
    total_tool_calls: int = 0,
    total_usage: dict[str, Any] | None = None,
    final_part_count: int = 0,
    error: str | None = None,
    pause: dict[str, Any] | None = None,
    context_sequence: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "message_roles": list(message_roles),
        "message_count": len(message_roles),
        "pending_tool_call_ids": list(pending_tool_call_ids),
        "iterations": iterations,
        "total_tool_calls": total_tool_calls,
        "total_usage": total_usage,
        "final_part_count": final_part_count,
        "error": error,
        "pause": pause,
    }
    if context_sequence is not None:
        payload["context_sequence"] = context_sequence
    return payload


def trace_transition_payload(
    from_status: str,
    to_status: str,
    *,
    iterations: int = 0,
    total_tool_calls: int = 0,
    total_usage: dict[str, Any] | None = None,
    error: str | None = None,
    pause: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "from": from_status,
        "to": to_status,
        "iterations": iterations,
        "total_tool_calls": total_tool_calls,
        "total_usage": total_usage,
        "error": error,
        "pause": pause,
    }


def trace_model_result_payload(*, tool_call_count: int = 0) -> dict[str, Any]:
    return {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 4,
        "tool_call_count": tool_call_count,
        "has_tool_calls": tool_call_count > 0,
    }


def trace_tool_result_payload(
    *, result_kind: str = "observation", is_error: bool = False
) -> dict[str, Any]:
    return {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 2,
        "result_kind": result_kind,
        "is_error": is_error,
        "metadata_keys": [],
        "pause": None,
    }


def trace_approval_requested_payload() -> dict[str, Any]:
    return {
        "id": "call-1",
        "name": "echo",
        "mode": "execute",
        "risk_keys": [],
        "metadata_keys": [],
    }


def trace_approval_completed_payload() -> dict[str, Any]:
    return {
        "id": "call-1",
        "name": "echo",
        "mode": "execute",
        "action": "allow",
        "reason": "approved",
        "metadata_keys": [],
    }


def trace_final_payload() -> dict[str, Any]:
    return {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 4,
        "metadata_keys": [],
    }


def approval_completed_trace_steps() -> list[TraceStep]:
    return [
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
            payload=trace_model_result_payload(tool_call_count=1),
        ),
        TraceStep(
            step_id=4,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_transition_payload("planning", "executing_tools", iterations=1),
        ),
        TraceStep(
            step_id=5,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_state_payload(
                "executing_tools",
                message_roles=("user", "assistant"),
                pending_tool_call_ids=("call-1",),
                iterations=1,
                context_sequence=5,
            ),
        ),
        TraceStep(
            step_id=6,
            kind=TraceStepKinds.APPROVAL_REQUESTED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_approval_requested_payload(),
        ),
        TraceStep(
            step_id=7,
            kind=TraceStepKinds.APPROVAL_COMPLETED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_approval_completed_payload(),
        ),
        TraceStep(
            step_id=8,
            kind=TraceStepKinds.TOOL_CALL,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload={
                "id": "call-1",
                "name": "echo",
                "mode": "execute",
                "batch_id": "tool-batch-1",
                "parallel": False,
                "index": 0,
                "implementation_invoked": True,
            },
        ),
        TraceStep(
            step_id=9,
            kind=TraceStepKinds.TOOL_RESULT,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload={
                "id": "call-1",
                "name": "echo",
                "mode": "execute",
                "batch_id": "tool-batch-1",
                "parallel": False,
                "index": 0,
                "implementation_invoked": True,
                "result": trace_tool_result_payload(),
            },
        ),
        TraceStep(
            step_id=10,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.PLANNING,
            payload=trace_transition_payload(
                "executing_tools", "planning", iterations=1, total_tool_calls=1
            ),
        ),
        TraceStep(
            step_id=11,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.PLANNING,
            payload=trace_state_payload(
                "planning",
                message_roles=("user", "assistant", "tool"),
                iterations=1,
                total_tool_calls=1,
                context_sequence=11,
            ),
        ),
        TraceStep(
            step_id=12,
            kind=TraceStepKinds.MODEL_CALL,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.PLANNING,
            payload={"iteration": 2},
        ),
        TraceStep(
            step_id=13,
            kind=TraceStepKinds.MODEL_RESULT,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.PLANNING,
            payload=trace_model_result_payload(),
        ),
        TraceStep(
            step_id=14,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.COMPLETED,
            payload=trace_transition_payload(
                "planning", "completed", iterations=2, total_tool_calls=1
            ),
        ),
        TraceStep(
            step_id=15,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.COMPLETED,
            after_status=AgentStatus.COMPLETED,
            payload=trace_state_payload(
                "completed",
                message_roles=("user", "assistant", "tool", "assistant"),
                iterations=2,
                total_tool_calls=1,
                final_part_count=1,
                context_sequence=15,
            ),
        ),
        TraceStep(
            step_id=16,
            kind=TraceStepKinds.FINAL,
            before_status=AgentStatus.COMPLETED,
            after_status=AgentStatus.COMPLETED,
            payload=trace_final_payload(),
        ),
        TraceStep(
            step_id=17,
            kind=TraceStepKinds.RUN_COMPLETED,
            before_status=AgentStatus.COMPLETED,
            after_status=AgentStatus.COMPLETED,
            payload={
                "state": trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant", "tool", "assistant"),
                    iterations=2,
                    total_tool_calls=1,
                    final_part_count=1,
                )
            },
        ),
    ]


def renumber_trace_steps(steps: Sequence[TraceStep]) -> list[TraceStep]:
    return [
        TraceStep(
            step_id=index,
            kind=step.kind,
            before_status=step.before_status,
            after_status=step.after_status,
            payload=step.payload,
            references=step.references,
            schema_version=step.schema_version,
        )
        for index, step in enumerate(steps, start=1)
    ]


def test_trace_step_data_is_immutable_and_deepcopyable() -> None:
    step = TraceStep(
        step_id=1,
        kind=TraceStepKinds.MODEL_RESULT,
        payload={"nested": {"value": 1}, "items": [{"value": 1}]},
    )

    with pytest.raises(TypeError, match="trace data is immutable"):
        step.payload["nested"]["value"] = 2
    copied = deepcopy(step.payload)
    copied["items"].append({"value": 2})

    assert copied == {"nested": {"value": 1}, "items": [{"value": 1}, {"value": 2}]}
    assert step.payload == {"nested": {"value": 1}, "items": [{"value": 1}]}


def test_replay_rejects_approval_completed_without_request() -> None:
    steps = approval_completed_trace_steps()
    del steps[5]

    with pytest.raises(ReplayError, match="approval_completed without approval_requested"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_terminal_trace_with_open_approval_request() -> None:
    steps = approval_completed_trace_steps()
    del steps[6]

    with pytest.raises(ReplayError, match="approval request open"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_denied_approval_that_invokes_tool() -> None:
    steps = approval_completed_trace_steps()
    steps[6] = TraceStep(
        step_id=7,
        kind=TraceStepKinds.APPROVAL_COMPLETED,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=trace_approval_completed_payload() | {"action": "deny", "reason": "blocked"},
    )

    with pytest.raises(ReplayError, match="denied approval"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_paused_approval_that_starts_tool() -> None:
    steps = approval_completed_trace_steps()
    steps[6] = TraceStep(
        step_id=7,
        kind=TraceStepKinds.APPROVAL_COMPLETED,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=trace_approval_completed_payload() | {"action": "pause", "reason": "approval"},
    )

    with pytest.raises(ReplayError, match="paused approval"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_approval_requested_after_tool_call_started() -> None:
    steps = approval_completed_trace_steps()
    steps[8] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.APPROVAL_REQUESTED,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=trace_approval_requested_payload(),
    )

    with pytest.raises(ReplayError, match="must precede tool_call"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_approval_requested_after_tool_call_completed() -> None:
    steps = approval_completed_trace_steps()
    steps[9] = TraceStep(
        step_id=10,
        kind=TraceStepKinds.APPROVAL_REQUESTED,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=trace_approval_requested_payload(),
    )

    with pytest.raises(ReplayError, match="must precede tool_call"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_tool_call_out_of_pending_order() -> None:
    steps = [
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
            payload=trace_model_result_payload(tool_call_count=2),
        ),
        TraceStep(
            step_id=4,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_transition_payload("planning", "executing_tools", iterations=1),
        ),
        TraceStep(
            step_id=5,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_state_payload(
                "executing_tools",
                message_roles=("user", "assistant"),
                pending_tool_call_ids=("call-1", "call-2"),
                iterations=1,
                context_sequence=5,
            ),
        ),
        TraceStep(
            step_id=6,
            kind=TraceStepKinds.TOOL_CALL,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload={
                "id": "call-2",
                "name": "echo",
                "mode": "execute",
                "batch_id": "tool-batch-1",
                "parallel": True,
                "index": 0,
                "implementation_invoked": True,
            },
        ),
    ]

    with pytest.raises(ReplayError, match="pending_tool_call_ids order"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_approval_request_out_of_pending_order() -> None:
    steps = [
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
            payload=trace_model_result_payload(tool_call_count=2),
        ),
        TraceStep(
            step_id=4,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_transition_payload("planning", "executing_tools", iterations=1),
        ),
        TraceStep(
            step_id=5,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_state_payload(
                "executing_tools",
                message_roles=("user", "assistant"),
                pending_tool_call_ids=("call-1", "call-2"),
                iterations=1,
                context_sequence=5,
            ),
        ),
        TraceStep(
            step_id=6,
            kind=TraceStepKinds.APPROVAL_REQUESTED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=trace_approval_requested_payload() | {"id": "call-2"},
        ),
    ]

    with pytest.raises(ReplayError, match="pending_tool_call_ids order"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_allowed_approval_with_non_invoked_error_result() -> None:
    steps = approval_completed_trace_steps()
    call_payload = dict(steps[7].payload)
    call_payload["implementation_invoked"] = False
    steps[7] = TraceStep(
        step_id=8,
        kind=TraceStepKinds.TOOL_CALL,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=call_payload,
    )
    result_payload = dict(steps[8].payload)
    result_payload["implementation_invoked"] = False
    result_payload["result"] = dict(cast(Mapping[str, Any], result_payload["result"])) | {
        "is_error": True,
    }
    steps[8] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.TOOL_RESULT,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=result_payload,
    )

    with pytest.raises(ReplayError, match="allowed approval"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_non_invoked_tool_result_success() -> None:
    steps = approval_completed_trace_steps()
    del steps[5:7]
    call_payload = dict(steps[5].payload)
    call_payload["implementation_invoked"] = False
    steps[5] = TraceStep(
        step_id=8,
        kind=TraceStepKinds.TOOL_CALL,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=call_payload,
    )
    result_payload = dict(steps[6].payload)
    result_payload["implementation_invoked"] = False
    steps[6] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.TOOL_RESULT,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=result_payload,
    )

    with pytest.raises(ReplayError, match="non-invoked tool_result"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_non_invoked_tool_result_pause() -> None:
    steps = approval_completed_trace_steps()
    del steps[5:7]
    call_payload = dict(steps[5].payload)
    call_payload["implementation_invoked"] = False
    steps[5] = TraceStep(
        step_id=8,
        kind=TraceStepKinds.TOOL_CALL,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=call_payload,
    )
    result_payload = dict(steps[6].payload)
    result_payload["implementation_invoked"] = False
    result_payload["result"] = dict(cast(Mapping[str, Any], result_payload["result"])) | {
        "is_error": True,
        "pause": {
            "reason": "external_wait",
            "source": "tool",
            "wait_id": "job-1",
            "metadata_keys": [],
            "interrupt": False,
        },
    }
    steps[6] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.TOOL_RESULT,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=result_payload,
    )

    with pytest.raises(ReplayError, match="must not request pause"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_tool_progress_envelope_drift() -> None:
    steps = approval_completed_trace_steps()
    steps.insert(
        8,
        TraceStep(
            step_id=9,
            kind=TraceStepKinds.TOOL_PROGRESS,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload={
                "id": "call-1",
                "name": "other",
                "mode": "execute",
                "batch_id": "tool-batch-1",
                "parallel": False,
                "index": 0,
                "implementation_invoked": True,
                "progress_keys": ["step"],
            },
        ),
    )

    with pytest.raises(ReplayError, match="tool_progress envelope"):
        replay_trace(RunTrace(run_id="run-1", steps=renumber_trace_steps(steps)))


def test_replay_rejects_background_task_without_tool_result_task() -> None:
    steps = approval_completed_trace_steps()
    steps.insert(
        9,
        TraceStep(
            step_id=10,
            kind=TraceStepKinds.BACKGROUND_TASK_STARTED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload={
                "id": "task-1",
                "status": "queued",
                "kind": "research",
                "lifecycle": "started",
                "metadata_keys": [],
                "tool_call": {
                    "id": "call-1",
                    "name": "echo",
                    "mode": "execute",
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            },
        ),
    )

    with pytest.raises(ReplayError, match="tool_result background_task"):
        replay_trace(RunTrace(run_id="run-1", steps=renumber_trace_steps(steps)))


def test_replay_rejects_duplicate_background_task_event_for_tool_result() -> None:
    steps = approval_completed_trace_steps()
    result_payload = dict(steps[8].payload)
    result_payload["result"] = dict(cast(Mapping[str, Any], result_payload["result"])) | {
        "background_task": {
            "id": "task-1",
            "status": "queued",
            "kind": "research",
            "lifecycle": "started",
            "metadata_keys": [],
        }
    }
    steps[8] = TraceStep(
        step_id=9,
        kind=TraceStepKinds.TOOL_RESULT,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=result_payload,
    )
    background_step = TraceStep(
        step_id=10,
        kind=TraceStepKinds.BACKGROUND_TASK_STARTED,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload={
            "id": "task-1",
            "status": "queued",
            "kind": "research",
            "lifecycle": "started",
            "metadata_keys": [],
            "tool_call": {
                "id": "call-1",
                "name": "echo",
                "mode": "execute",
                "batch_id": "tool-batch-1",
                "parallel": False,
                "index": 0,
                "implementation_invoked": True,
            },
        },
    )
    steps.insert(9, background_step)
    steps.insert(10, background_step)

    with pytest.raises(ReplayError, match="already recorded"):
        replay_trace(RunTrace(run_id="run-1", steps=renumber_trace_steps(steps)))


def test_replay_rejects_misplaced_child_run_started() -> None:
    steps = approval_completed_trace_steps()
    steps.insert(
        3,
        TraceStep(
            step_id=4,
            kind=TraceStepKinds.CHILD_RUN_STARTED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.PLANNING,
            payload={
                "parent_run_id": "parent-run",
                "parent_tool_call_id": "call-1",
                "run_kind": "subagent",
            },
        ),
    )

    with pytest.raises(ReplayError, match="immediately follow run_started"):
        replay_trace(RunTrace(run_id="run-1", steps=renumber_trace_steps(steps)))


def test_replay_rejects_unpaired_child_run_started() -> None:
    steps = approval_completed_trace_steps()
    steps.insert(
        1,
        TraceStep(
            step_id=2,
            kind=TraceStepKinds.CHILD_RUN_STARTED,
            before_status=AgentStatus.PLANNING,
            after_status=AgentStatus.PLANNING,
            payload={
                "parent_run_id": "parent-run",
                "parent_tool_call_id": "call-1",
                "run_kind": "subagent",
            },
        ),
    )

    with pytest.raises(ReplayError, match="child_run_started"):
        replay_trace(RunTrace(run_id="run-1", steps=renumber_trace_steps(steps)))


def test_replay_rejects_allowed_approval_name_drift() -> None:
    steps = approval_completed_trace_steps()
    payload = dict(steps[7].payload)
    payload["name"] = "other"
    steps[7] = TraceStep(
        step_id=8,
        kind=TraceStepKinds.TOOL_CALL,
        before_status=AgentStatus.EXECUTING_TOOLS,
        after_status=AgentStatus.EXECUTING_TOOLS,
        payload=payload,
    )

    with pytest.raises(ReplayError, match="approval decision"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_replay_rejects_allowed_approval_left_unresolved_by_pause() -> None:
    request_pause: dict[str, Any] = {
        "reason": "manual_pause",
        "source": "control",
        "wait_id": None,
        "metadata_keys": [],
        "interrupt": False,
        "resume_status": "executing_tools",
        "origin": "control",
    }
    state_pause: dict[str, Any] = {
        "reason": "manual_pause",
        "source": "control",
        "wait_id": None,
        "metadata_keys": [],
        "resume_status": "executing_tools",
    }
    steps = approval_completed_trace_steps()[:7] + [
        TraceStep(
            step_id=8,
            kind=TraceStepKinds.PAUSE_REQUESTED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.EXECUTING_TOOLS,
            payload=request_pause,
        ),
        TraceStep(
            step_id=9,
            kind=TraceStepKinds.STATE_CHANGED,
            before_status=AgentStatus.EXECUTING_TOOLS,
            after_status=AgentStatus.PAUSED,
            payload=trace_transition_payload(
                "executing_tools",
                "paused",
                iterations=1,
                pause=state_pause,
            ),
        ),
        TraceStep(
            step_id=10,
            kind=TraceStepKinds.CHECKPOINT,
            before_status=AgentStatus.PAUSED,
            after_status=AgentStatus.PAUSED,
            payload=trace_state_payload(
                "paused",
                message_roles=("user", "assistant"),
                pending_tool_call_ids=("call-1",),
                iterations=1,
                pause=state_pause,
                context_sequence=10,
            ),
        ),
        TraceStep(
            step_id=11,
            kind=TraceStepKinds.RUN_PAUSED,
            before_status=AgentStatus.PAUSED,
            after_status=AgentStatus.PAUSED,
            payload={"pause": state_pause},
        ),
        TraceStep(
            step_id=12,
            kind=TraceStepKinds.RUN_COMPLETED,
            before_status=AgentStatus.PAUSED,
            after_status=AgentStatus.PAUSED,
            payload={
                "state": trace_state_payload(
                    "paused",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                    pause=state_pause,
                )
            },
        ),
    ]

    with pytest.raises(ReplayError, match="approval decision unresolved"):
        replay_trace(RunTrace(run_id="run-1", steps=steps))


def test_run_trace_round_trip_rejects_invalid_shape() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "status": "planning",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            )
        ],
    )
    payload = trace.to_dict()

    assert RunTrace.from_dict(payload).to_dict() == payload
    assert payload["metadata"] == {"metadata_keys": []}

    broken = dict(payload)
    broken["steps"] = None
    with pytest.raises(TypeError, match="trace steps"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["legacy"] = True
    with pytest.raises(ValueError, match="unknown"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["metadata"]["tenant"] = "raw"
    with pytest.raises(ValueError, match="unknown"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["steps"][0]["kind"] = "legacy_step"
    with pytest.raises(ValueError, match="unsupported trace step kind"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["steps"][0]["provider"] = {}
    with pytest.raises(ValueError, match="unknown"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["steps"][0]["schema_version"] = ""
    with pytest.raises(ValueError, match="schema_version"):
        RunTrace.from_dict(broken)

    broken = trace.to_dict()
    broken["steps"][0]["payload"]["message_count"] = 2
    with pytest.raises(ValueError, match="message_roles length"):
        RunTrace.from_dict(broken)

    with pytest.raises(TypeError, match="trace step kind"):
        TraceStep(step_id=1, kind=cast(Any, True))


def test_run_trace_from_dict_rejects_inconsistent_model_result_tool_call_summary() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.MODEL_RESULT,
                payload={
                    "part_count": 0,
                    "part_types": [],
                    "text_length": 0,
                    "tool_call_count": 1,
                    "has_tool_calls": False,
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="has_tool_calls"):
        RunTrace.from_dict(trace.to_dict())


def test_run_trace_from_dict_rejects_duplicate_pending_tool_call_ids() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1", "call-1"),
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="pending_tool_call_ids.*unique"):
        RunTrace.from_dict(trace.to_dict())


def test_run_trace_from_dict_rejects_resume_append_count_mismatch() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RESUME,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "snapshot_status": "paused",
                    "restored_status": "planning",
                    "append_message_roles": ["user"],
                    "append_message_count": 0,
                    "metadata_keys": [],
                    "expected_pause": None,
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="append_message_roles length"):
        RunTrace.from_dict(trace.to_dict())


def test_trace_compacts_event_metadata_values() -> None:
    trace = RunTrace.from_events(
        "run-1",
        [
            AgentEvent(
                EventTypes.MODEL_DELTA,
                {
                    "kind": "text_delta",
                    "index": 0,
                    "text_delta": "secret text",
                    "metadata": {"token": "secret"},
                },
                run_id="run-1",
                sequence=1,
            ),
            AgentEvent(
                EventTypes.TOOL_COMPLETED,
                {
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 6,
                        "result_kind": "observation",
                        "is_error": False,
                        "metadata": {"artifact": {"id": "a1"}},
                        "pause": None,
                    },
                },
                run_id="run-1",
                sequence=2,
            ),
            AgentEvent(
                EventTypes.FINAL,
                {
                    "parts": [
                        {
                            "type": "text",
                            "text": "secret final",
                            "metadata": {"secret": "value"},
                        }
                    ],
                    "summary": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 12,
                    },
                },
                run_id="run-1",
                sequence=3,
            ),
        ],
        metadata={"tenant": "acme"},
    )
    payload = trace.to_dict()

    assert payload["metadata"] == {"metadata_keys": ["tenant"]}
    assert payload["steps"][0]["payload"]["text_delta_length"] == 11
    assert payload["steps"][0]["payload"]["metadata_keys"] == ["token"]
    assert "metadata" not in payload["steps"][0]["payload"]
    assert payload["steps"][1]["payload"]["result"]["metadata_keys"] == ["artifact"]
    assert "metadata" not in payload["steps"][1]["payload"]["result"]
    assert payload["steps"][2]["payload"] == {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 12,
        "metadata_keys": ["secret"],
    }


def test_run_trace_from_events_normalizes_model_error_null_fields() -> None:
    trace = RunTrace.from_events(
        "run-1",
        [
            AgentEvent(
                EventTypes.MODEL_ERROR,
                {
                    "error": {
                        "message": "provider unavailable",
                        "provider": None,
                        "code": None,
                        "status_code": None,
                        "retryable": None,
                        "request_id": None,
                        "metadata": {"tenant": "acme"},
                    },
                    "retry": False,
                },
                run_id="run-1",
            )
        ],
    )

    step = trace.steps[0]
    assert step.payload == {
        "message": "provider unavailable",
        "retry": False,
        "retryable": False,
        "metadata_keys": ["tenant"],
    }
    RunTrace.from_dict(trace.to_dict())


def test_run_trace_from_events_rejects_schema_invalid_approval_events() -> None:
    approval_requested: dict[str, Any] = {
        "id": "call-1",
        "name": "echo",
        "mode": "execute",
        "risk": {},
        "metadata": {},
    }
    approval_completed: dict[str, Any] = {
        "id": "call-1",
        "name": "echo",
        "mode": "execute",
        "action": "allow",
        "reason": "approved",
        "metadata": {},
    }

    for event_type, event_data, required_key in (
        (EventTypes.APPROVAL_REQUESTED, approval_requested, "risk"),
        (EventTypes.APPROVAL_REQUESTED, approval_requested, "metadata"),
        (EventTypes.APPROVAL_COMPLETED, approval_completed, "metadata"),
    ):
        with pytest.raises(KeyError):
            RunTrace.from_events(
                "run-1",
                [
                    AgentEvent(
                        event_type,
                        {key: value for key, value in event_data.items() if key != required_key},
                        run_id="run-1",
                    )
                ],
            )


def event_state_summary(**overrides: Any) -> dict[str, Any]:
    payload = trace_state_payload("planning")
    payload["pending_tool_call_count"] = len(payload["pending_tool_call_ids"])
    payload["has_final"] = payload["final_part_count"] > 0
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    "required_key", ["message_roles", "pending_tool_call_ids", "final_part_count"]
)
def test_run_trace_from_events_requires_state_summary_fields(required_key: str) -> None:
    state = event_state_summary()
    del state[required_key]
    if required_key == "final_part_count":
        state["has_final"] = True

    with pytest.raises(KeyError):
        RunTrace.from_events(
            "run-1",
            [
                AgentEvent(
                    EventTypes.RUN_STARTED,
                    {"state": state},
                    run_id="run-1",
                )
            ],
        )


def test_run_trace_from_events_rejects_full_state_for_state_summary_events() -> None:
    full_state = {
        "status": "planning",
        "messages": [Message.user_text("hi").to_dict()],
        "pending_tool_calls": [],
        "iterations": 0,
        "total_tool_calls": 0,
        "total_usage": None,
        "final_parts": [],
        "error": None,
        "pause": None,
    }

    for event_type in (EventTypes.RUN_STARTED, EventTypes.RUN_COMPLETED):
        with pytest.raises(ValueError, match="unknown field"):
            RunTrace.from_events(
                "run-1",
                [
                    AgentEvent(
                        event_type,
                        {"state": full_state},
                        run_id="run-1",
                    )
                ],
            )


def test_run_trace_from_events_rejects_schema_invalid_tool_result_summary() -> None:
    result: dict[str, Any] = {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 2,
        "result_kind": "observation",
        "is_error": False,
        "metadata": {},
        "pause": None,
    }
    del result["metadata"]

    with pytest.raises(KeyError):
        RunTrace.from_events(
            "run-1",
            [
                AgentEvent(
                    EventTypes.TOOL_COMPLETED,
                    {
                        "id": "call-1",
                        "name": "echo",
                        "mode": "execute",
                        "batch_id": "batch-1",
                        "parallel": False,
                        "index": 0,
                        "implementation_invoked": True,
                        "result": result,
                    },
                    run_id="run-1",
                )
            ],
        )


def test_run_trace_from_events_rejects_schema_invalid_final_event() -> None:
    with pytest.raises(KeyError):
        RunTrace.from_events(
            "run-1",
            [
                AgentEvent(
                    EventTypes.FINAL,
                    {
                        "summary": {
                            "part_count": 0,
                            "part_types": [],
                            "text_length": 0,
                        }
                    },
                    run_id="run-1",
                )
            ],
        )


def test_run_trace_from_events_rejects_schema_invalid_checkpoint_state() -> None:
    checkpoint: dict[str, Any] = {
        "state": event_state_summary(status="planning"),
        "context": {
            "run_id": "run-1",
            "started_at": 1.0,
            "deadline": None,
            "metadata": {},
            "sequence": 1,
        },
    }

    with pytest.raises(ValueError, match="unknown field"):
        RunTrace.from_events(
            "run-1",
            [
                AgentEvent(
                    EventTypes.CHECKPOINT,
                    checkpoint,
                    run_id="run-1",
                )
            ],
        )


def test_run_trace_from_events_rejects_cross_run_events() -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunTrace.from_events(
            "run-1",
            [
                AgentEvent(
                    EventTypes.RUN_STARTED,
                    {"state": event_state_summary()},
                    run_id="other-run",
                )
            ],
        )


@pytest.mark.asyncio
async def test_run_trace_from_dict_and_replay_reject_raw_payload_metadata() -> None:
    result = await AgentLoop(model=OneShotModel()).run([Message.user_text("x")])
    assert result.trace is not None
    payload = result.trace.to_dict()
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


def test_run_trace_from_dict_rejects_interrupting_tool_result_origin_pause() -> None:
    payload: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": TraceStepKinds.PAUSE_REQUESTED,
                "before_status": AgentStatus.EXECUTING_TOOLS.value,
                "after_status": AgentStatus.EXECUTING_TOOLS.value,
                "references": {},
                "payload": {
                    "reason": "external_wait",
                    "source": "tool",
                    "wait_id": "job-1",
                    "metadata_keys": [],
                    "interrupt": True,
                    "resume_status": AgentStatus.PLANNING.value,
                    "origin": "tool_result",
                },
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    with pytest.raises(ValueError, match="interrupt must be false"):
        RunTrace.from_dict(payload)


def test_replay_rejects_invalid_state_machine_transition() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "status": "executing_tools",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": ["call-1"],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("executing_tools", "completed", iterations=1),
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "status": "completed",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": ["call-1"],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 1,
                    "error": None,
                    "pause": None,
                    "context_sequence": 3,
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=5,
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

    with pytest.raises(ReplayError, match="invalid state transition"):
        replay_trace(trace)


def test_replay_rejects_completed_transition_without_model_result() -> None:
    trace = RunTrace(
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
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed"),
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant"),
                    final_part_count=1,
                    context_sequence=3,
                ),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "state": trace_state_payload(
                        "completed",
                        message_roles=("user", "assistant"),
                        final_part_count=1,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="model_result"):
        replay_trace(trace)
    assert replay_trace(trace, strict=False).valid is False


def test_replay_rejects_planning_transition_without_tool_result() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload("executing_tools", "planning"),
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload(
                    "planning",
                    message_roles=("user", "assistant", "tool"),
                    total_tool_calls=1,
                    context_sequence=3,
                ),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.MODEL_CALL,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={"iteration": 1},
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.MODEL_RESULT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", total_tool_calls=1),
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant", "tool", "assistant"),
                    total_tool_calls=1,
                    final_part_count=1,
                    context_sequence=7,
                ),
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=9,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "state": trace_state_payload(
                        "completed",
                        message_roles=("user", "assistant", "tool", "assistant"),
                        total_tool_calls=1,
                        final_part_count=1,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="tool_result"):
        replay_trace(trace)


def test_replay_rejects_planning_transition_with_unfinished_pending_tool_call() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(tool_call_count=2),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_transition_payload("planning", "executing_tools", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1", "call-2"),
                    iterations=1,
                    context_sequence=5,
                ),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload(
                    "executing_tools",
                    "planning",
                    iterations=1,
                    total_tool_calls=1,
                ),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="all pending tool calls"):
        replay_trace(trace)
    assert replay_trace(trace, strict=False).valid is False


def test_replay_accepts_parallel_pending_tool_results_in_completion_order() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1", "call-2"),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload(
                    "executing_tools",
                    "planning",
                    iterations=1,
                    total_tool_calls=2,
                ),
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.MODEL_CALL,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={"iteration": 2},
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.MODEL_RESULT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=9,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload(
                    "planning",
                    "completed",
                    iterations=2,
                    total_tool_calls=2,
                ),
            ),
            TraceStep(
                step_id=10,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant", "tool", "tool", "assistant"),
                    iterations=2,
                    total_tool_calls=2,
                    final_part_count=1,
                    context_sequence=10,
                ),
            ),
            TraceStep(
                step_id=11,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=12,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "state": trace_state_payload(
                        "completed",
                        message_roles=("user", "assistant", "tool", "tool", "assistant"),
                        iterations=2,
                        total_tool_calls=2,
                        final_part_count=1,
                    )
                },
            ),
        ],
    )

    assert replay_trace(trace).valid is True


def test_replay_rejects_partial_stream_checkpoint_after_pause_transition() -> None:
    pause_request: dict[str, Any] = {
        "reason": "user_interrupt",
        "source": "host",
        "wait_id": None,
        "metadata_keys": [],
        "interrupt": True,
        "resume_status": "planning",
        "origin": "control",
    }
    pause_state: dict[str, Any] = {
        "reason": "user_interrupt",
        "resume_status": "planning",
        "source": "host",
        "wait_id": None,
        "metadata_keys": [],
    }
    trace = RunTrace(
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
                kind=TraceStepKinds.MODEL_DELTA,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={
                    "kind": "text_delta",
                    "index": 0,
                    "text_delta_length": 7,
                    "part_type": "text",
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=pause_request,
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PAUSED,
                payload=trace_transition_payload("planning", "paused", pause=pause_state),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload=trace_state_payload(
                    "paused",
                    message_roles=("user", "assistant"),
                    pause=pause_state,
                    context_sequence=6,
                ),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="stream delta"):
        replay_trace(trace)


def test_replay_rejects_tool_pause_without_adjacent_tool_result() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "status": "planning",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={
                    "reason": "external_wait",
                    "source": "tool",
                    "wait_id": "job-1",
                    "metadata_keys": [],
                    "interrupt": False,
                    "resume_status": "planning",
                    "origin": "tool_result",
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PAUSED,
                payload={
                    "from": "planning",
                    "to": "paused",
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "error": None,
                    "pause": {
                        "reason": "external_wait",
                        "resume_status": "planning",
                        "source": "tool",
                        "wait_id": "job-1",
                        "metadata_keys": [],
                    },
                },
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "status": "paused",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": {
                        "reason": "external_wait",
                        "resume_status": "planning",
                        "source": "tool",
                        "wait_id": "job-1",
                        "metadata_keys": [],
                    },
                    "context_sequence": 6,
                },
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.RUN_PAUSED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "pause": {
                        "reason": "external_wait",
                        "resume_status": "planning",
                        "source": "tool",
                        "wait_id": "job-1",
                        "metadata_keys": [],
                    }
                },
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "state": trace_state_payload(
                        "paused",
                        iterations=1,
                        pause={
                            "reason": "external_wait",
                            "resume_status": "planning",
                            "source": "tool",
                            "wait_id": "job-1",
                            "metadata_keys": [],
                        },
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="matching pause-bearing tool_result"):
        replay_trace(trace)


def test_replay_rejects_host_pause_checkpoint_mismatch() -> None:
    def pause_payload(wait_id: str) -> dict[str, object]:
        return {
            "reason": "operator_requested",
            "resume_status": "planning",
            "source": "host",
            "wait_id": wait_id,
            "metadata_keys": ["ticket"],
        }

    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "status": "planning",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=pause_payload("job-1") | {"interrupt": False, "origin": "control"},
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PAUSED,
                payload={
                    "from": "planning",
                    "to": "paused",
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "error": None,
                    "pause": pause_payload("job-1"),
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "status": "paused",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": pause_payload("job-2"),
                    "context_sequence": 4,
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.RUN_PAUSED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={"pause": pause_payload("job-2")},
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "state": trace_state_payload(
                        "paused",
                        pause=cast(dict[str, Any], pause_payload("job-2")),
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="checkpoint pause does not match pause request"):
        replay_trace(trace)


def test_replay_rejects_parallel_tool_pause_that_does_not_match_first_wait() -> None:
    def pause_payload(wait_id: str) -> dict[str, object]:
        return {
            "reason": "external_wait",
            "source": "tool",
            "wait_id": wait_id,
            "metadata_keys": [],
            "interrupt": False,
        }

    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "status": "executing_tools",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": ["call-1", "call-2"],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 5,
                        "result_kind": "observation",
                        "is_error": False,
                        "metadata_keys": [],
                        "pause": pause_payload("job-2"),
                    },
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 5,
                        "result_kind": "observation",
                        "is_error": False,
                        "metadata_keys": [],
                        "pause": pause_payload("job-1"),
                    },
                },
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=pause_payload("job-2")
                | {"resume_status": "planning", "origin": "tool_result"},
            ),
        ],
    )

    with pytest.raises(ReplayError, match="first pause-bearing tool_result"):
        replay_trace(trace)


def test_replay_rejects_paused_trace_with_open_tool_call() -> None:
    pause: dict[str, Any] = {
        "reason": "operator_requested",
        "resume_status": "executing_tools",
        "source": "host",
        "wait_id": None,
        "metadata_keys": [],
    }
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "reason": "operator_requested",
                    "source": "host",
                    "wait_id": None,
                    "metadata_keys": [],
                    "interrupt": False,
                    "resume_status": "executing_tools",
                    "origin": "control",
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PAUSED,
                payload=trace_transition_payload(
                    "executing_tools",
                    "paused",
                    iterations=1,
                    pause=pause,
                ),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload=trace_state_payload(
                    "paused",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                    pause=pause,
                    context_sequence=5,
                ),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.RUN_PAUSED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={"pause": pause},
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "state": trace_state_payload(
                        "paused",
                        message_roles=("user", "assistant"),
                        pending_tool_call_ids=("call-1",),
                        iterations=1,
                        pause=pause,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="paused trace cannot leave tool_call open"):
        replay_trace(trace)


def test_replay_rejects_completed_trace_missing_model_result() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "status": "planning",
                    "message_roles": ["user"],
                    "message_count": 1,
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.MODEL_CALL,
                payload={"iteration": 1},
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "status": "completed",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": [],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 1,
                    "error": None,
                    "pause": None,
                    "context_sequence": 4,
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=6,
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

    with pytest.raises(ReplayError, match="model_result"):
        replay_trace(trace)


def test_replay_rejects_run_started_payload_status_mismatch() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload("executing_tools"),
            )
        ],
    )

    with pytest.raises(ReplayError, match="run_started payload status"):
        replay_trace(trace)


def test_replay_rejects_resume_payload_status_mismatch() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RESUME,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PLANNING,
                payload={
                    "snapshot_status": "executing_tools",
                    "restored_status": "planning",
                    "append_message_roles": [],
                    "append_message_count": 0,
                    "metadata_keys": [],
                    "expected_pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload("planning"),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="snapshot_status"):
        replay_trace(trace)


def test_replay_rejects_state_changed_payload_status_mismatch() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "executing_tools", iterations=1),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="payload to"):
        replay_trace(trace)


def test_replay_rejects_run_completed_state_mismatching_last_checkpoint() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
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
                    final_part_count=2,
                    context_sequence=5,
                ),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "part_count": 2,
                    "part_types": ["text"],
                    "text_length": 8,
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

    with pytest.raises(ReplayError, match="final_part_count"):
        replay_trace(trace)


def test_replay_rejects_final_part_count_mismatching_checkpoint() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
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
                    final_part_count=2,
                    context_sequence=5,
                ),
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
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
                        final_part_count=2,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="final part_count"):
        replay_trace(trace)


def test_replay_rejects_tool_result_envelope_mismatch() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 1,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="tool_result envelope"):
        replay_trace(trace)


@pytest.mark.parametrize(
    ("result_kind", "is_error"),
    [("observation", False), ("acceptance", False), ("rejection", True)],
)
def test_trace_rejects_custom_mode_reserved_result_kind(result_kind: str, is_error: bool) -> None:
    result = trace_tool_result_payload(result_kind=result_kind, is_error=is_error)
    if result_kind == "acceptance":
        result["correlation_id"] = "job-1"
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "handoff",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "handoff",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": result,
                },
            ),
        ],
    )

    with pytest.raises(ValueError, match="result_kind"):
        RunTrace.from_dict(trace.to_dict())
    with pytest.raises(ReplayError, match="result_kind"):
        replay_trace(trace)


def test_replay_rejects_checkpoint_pending_count_mismatching_model_result() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(tool_call_count=2),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_transition_payload("planning", "executing_tools", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                    context_sequence=5,
                ),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="pending_tool_call_ids"):
        replay_trace(trace)


def test_replay_rejects_paused_checkpoint_pending_count_mismatching_model_result() -> None:
    pause: dict[str, Any] = {
        "reason": "operator_requested",
        "resume_status": "executing_tools",
        "source": "host",
        "wait_id": None,
        "metadata_keys": [],
    }
    trace = RunTrace(
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
                payload=trace_model_result_payload(tool_call_count=2),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_transition_payload("planning", "executing_tools", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.PAUSE_REQUESTED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=pause | {"interrupt": False, "origin": "control"},
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PAUSED,
                payload=trace_transition_payload(
                    "executing_tools",
                    "paused",
                    iterations=1,
                    pause=pause,
                ),
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload=trace_state_payload(
                    "paused",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                    pause=pause,
                    context_sequence=7,
                ),
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.RUN_PAUSED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={"pause": pause},
            ),
            TraceStep(
                step_id=9,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.PAUSED,
                after_status=AgentStatus.PAUSED,
                payload={
                    "state": trace_state_payload(
                        "paused",
                        message_roles=("user", "assistant"),
                        pending_tool_call_ids=("call-1",),
                        iterations=1,
                        pause=pause,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="pending_tool_call_ids"):
        replay_trace(trace)


def test_replay_rejects_tool_call_before_model_result_pending_checkpoint() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(tool_call_count=1),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_transition_payload("planning", "executing_tools", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="checkpoint after model_result"):
        replay_trace(trace)


def test_replay_rejects_total_tool_calls_mismatching_tool_results() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1",),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload(
                    "executing_tools",
                    "planning",
                    iterations=1,
                    total_tool_calls=99,
                ),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="total_tool_calls"):
        replay_trace(trace)


def test_replay_rejects_committed_total_tool_calls_undercount() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload=trace_state_payload(
                    "executing_tools",
                    message_roles=("user", "assistant"),
                    pending_tool_call_ids=("call-1", "call-2"),
                    iterations=1,
                ),
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.TOOL_CALL,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 0,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.TOOL_RESULT,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "id": "call-2",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": True,
                    "index": 1,
                    "implementation_invoked": True,
                    "result": trace_tool_result_payload(),
                },
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload(
                    "executing_tools",
                    "planning",
                    iterations=1,
                    total_tool_calls=1,
                ),
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload(
                    "planning",
                    message_roles=("user", "assistant", "tool", "tool"),
                    iterations=1,
                    total_tool_calls=1,
                    context_sequence=7,
                ),
            ),
            TraceStep(
                step_id=8,
                kind=TraceStepKinds.MODEL_CALL,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload={"iteration": 2},
            ),
            TraceStep(
                step_id=9,
                kind=TraceStepKinds.MODEL_RESULT,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=10,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload(
                    "planning",
                    "completed",
                    iterations=2,
                    total_tool_calls=1,
                ),
            ),
            TraceStep(
                step_id=11,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_state_payload(
                    "completed",
                    message_roles=("user", "assistant", "tool", "tool", "assistant"),
                    iterations=2,
                    total_tool_calls=1,
                    final_part_count=1,
                    context_sequence=11,
                ),
            ),
            TraceStep(
                step_id=12,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=13,
                kind=TraceStepKinds.RUN_COMPLETED,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "state": trace_state_payload(
                        "completed",
                        message_roles=("user", "assistant", "tool", "tool", "assistant"),
                        iterations=2,
                        total_tool_calls=1,
                        final_part_count=1,
                    )
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="committed tool_results"):
        replay_trace(trace)


def test_replay_rejects_total_tool_calls_decrease() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.PLANNING,
                payload=trace_state_payload("planning", total_tool_calls=2),
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload(
                    "planning",
                    "completed",
                    iterations=1,
                    total_tool_calls=2,
                ),
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
                    total_tool_calls=1,
                    final_part_count=1,
                    context_sequence=5,
                ),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="must not decrease"):
        replay_trace(trace)


def test_replay_rejects_final_before_completed_state() -> None:
    trace = RunTrace(
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
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.PLANNING,
                payload=trace_final_payload(),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="final is only valid"):
        replay_trace(trace)


def test_replay_rejects_run_completed_before_final_step() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
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
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
            ),
        ],
    )

    with pytest.raises(ReplayError, match="final trace step"):
        replay_trace(trace)


def test_replay_rejects_error_payload_status_mismatch() -> None:
    trace = RunTrace(
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
                payload=trace_model_result_payload(),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
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
                payload=trace_final_payload(),
            ),
            TraceStep(
                step_id=7,
                kind=TraceStepKinds.ERROR,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={"status": "failed", "message": "wrong status"},
            ),
            TraceStep(
                step_id=8,
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

    with pytest.raises(ReplayError, match="error payload status"):
        replay_trace(trace)


def test_replay_rejects_model_delta_status_mismatch() -> None:
    trace = RunTrace(
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
                kind=TraceStepKinds.MODEL_DELTA,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "kind": "text_delta",
                    "index": 0,
                    "text_delta_length": 4,
                    "part_type": "text",
                },
            ),
        ],
    )

    with pytest.raises(ReplayError, match="model_delta before_status"):
        replay_trace(trace)


def test_replay_rejects_planning_transition_missing_tool_result() -> None:
    trace = RunTrace(
        run_id="run-1",
        steps=[
            TraceStep(
                step_id=1,
                kind=TraceStepKinds.RUN_STARTED,
                after_status=AgentStatus.EXECUTING_TOOLS,
                payload={
                    "status": "executing_tools",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": ["call-1"],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
            ),
            TraceStep(
                step_id=2,
                kind=TraceStepKinds.TOOL_CALL,
                payload={
                    "id": "call-1",
                    "name": "tool",
                    "mode": "execute",
                    "batch_id": "batch-1",
                    "parallel": False,
                    "index": 0,
                    "implementation_invoked": True,
                },
            ),
            TraceStep(
                step_id=3,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.EXECUTING_TOOLS,
                after_status=AgentStatus.PLANNING,
                payload=trace_transition_payload("executing_tools", "planning", iterations=1),
            ),
            TraceStep(
                step_id=4,
                kind=TraceStepKinds.STATE_CHANGED,
                before_status=AgentStatus.PLANNING,
                after_status=AgentStatus.COMPLETED,
                payload=trace_transition_payload("planning", "completed", iterations=1),
            ),
            TraceStep(
                step_id=5,
                kind=TraceStepKinds.CHECKPOINT,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload={
                    "status": "completed",
                    "message_roles": ["user", "assistant"],
                    "message_count": 2,
                    "pending_tool_call_ids": [],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "total_usage": None,
                    "final_part_count": 1,
                    "error": None,
                    "pause": None,
                    "context_sequence": 5,
                },
            ),
            TraceStep(
                step_id=6,
                kind=TraceStepKinds.FINAL,
                before_status=AgentStatus.COMPLETED,
                after_status=AgentStatus.COMPLETED,
                payload=trace_final_payload(),
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

    with pytest.raises(ReplayError, match="tool_result"):
        replay_trace(trace)


@pytest.mark.asyncio
async def test_run_trace_from_events_replays() -> None:
    events = [
        event
        async for event in AgentLoop(model=OneShotModel()).run_events([Message.user_text("x")])
    ]
    trace = RunTrace.from_events(events[0].run_id, events)

    result = replay_trace(trace)

    assert result.valid is True
    assert result.final_status is AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_replay_rejects_corrupted_terminal_tail() -> None:
    result = await AgentLoop(model=OneShotModel()).run([Message.user_text("x")])
    assert result.trace is not None
    payload = result.trace.to_dict()
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


@pytest.mark.asyncio
async def test_replay_diagnostic_mode_reports_invalid_trace() -> None:
    result = await AgentLoop(model=OneShotModel()).run([Message.user_text("x")])
    assert result.trace is not None
    payload: dict[str, Any] = result.trace.to_dict()
    payload["steps"][0]["after_status"] = "executing_tools"

    diagnostic = replay_trace(RunTrace.from_dict(payload), strict=False)

    assert diagnostic.valid is False
    assert diagnostic.message is not None


@pytest.mark.asyncio
async def test_model_delta_trace_is_not_durable_replay_input() -> None:
    result = await AgentLoop(model=OneShotModel()).run([Message.user_text("x")])
    assert result.trace is not None
    payload = result.trace.to_dict()
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
