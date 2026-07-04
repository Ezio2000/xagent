"""Internal runtime trace payload recorder."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from kernel._validation import (
    expect_bool as _expect_bool,
)
from kernel._validation import (
    expect_mapping as _expect_mapping,
)
from kernel._validation import (
    expect_nonnegative_int as _expect_nonnegative_int,
)
from kernel._validation import (
    expect_sequence as _expect_sequence,
)
from kernel._validation import (
    expect_str as _expect_str,
)
from kernel._validation import (
    reject_unknown_keys as _reject_unknown_keys,
)
from kernel.errors import ModelErrorInfo
from kernel.events import AgentEvent, EventTypes
from kernel.resume import ResumeInput
from kernel.state import AgentState
from kernel.status import AgentStatus


def _compact_trace_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"metadata_keys": sorted(str(key) for key in value)}


def _status_value(status: AgentStatus | None) -> str | None:
    return None if status is None else status.value


class TraceStepKinds:
    RUN_STARTED = "run_started"
    RESUME = "resume"
    MODEL_CALL = "model_call"
    MODEL_DELTA = "model_delta"
    MODEL_ERROR = "model_error"
    MODEL_RESULT = "model_result"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_COMPLETED = "approval_completed"
    TOOL_CALL = "tool_call"
    TOOL_PROGRESS = "tool_progress"
    TOOL_CANCEL_REQUESTED = "tool_cancel_requested"
    TOOL_RESULT = "tool_result"
    BACKGROUND_TASK_STARTED = "background_task_started"
    BACKGROUND_TASK_UPDATED = "background_task_updated"
    BACKGROUND_TASK_COMPLETED = "background_task_completed"
    CHILD_RUN_STARTED = "child_run_started"
    CHILD_RUN_COMPLETED = "child_run_completed"
    CONVERSATION_INSERT = "conversation_insert"
    PAUSE_REQUESTED = "pause_requested"
    STATE_CHANGED = "state_changed"
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ERROR = "error"
    RUN_PAUSED = "run_paused"
    RUN_COMPLETED = "run_completed"


EVENT_TRACE_KIND: Mapping[str, str] = {
    EventTypes.RUN_STARTED: TraceStepKinds.RUN_STARTED,
    EventTypes.MODEL_STARTED: TraceStepKinds.MODEL_CALL,
    EventTypes.MODEL_DELTA: TraceStepKinds.MODEL_DELTA,
    EventTypes.MODEL_ERROR: TraceStepKinds.MODEL_ERROR,
    EventTypes.MODEL_COMPLETED: TraceStepKinds.MODEL_RESULT,
    EventTypes.APPROVAL_REQUESTED: TraceStepKinds.APPROVAL_REQUESTED,
    EventTypes.APPROVAL_COMPLETED: TraceStepKinds.APPROVAL_COMPLETED,
    EventTypes.TOOL_STARTED: TraceStepKinds.TOOL_CALL,
    EventTypes.TOOL_PROGRESS: TraceStepKinds.TOOL_PROGRESS,
    EventTypes.TOOL_CANCEL_REQUESTED: TraceStepKinds.TOOL_CANCEL_REQUESTED,
    EventTypes.TOOL_COMPLETED: TraceStepKinds.TOOL_RESULT,
    EventTypes.BACKGROUND_TASK_STARTED: TraceStepKinds.BACKGROUND_TASK_STARTED,
    EventTypes.BACKGROUND_TASK_UPDATED: TraceStepKinds.BACKGROUND_TASK_UPDATED,
    EventTypes.BACKGROUND_TASK_COMPLETED: TraceStepKinds.BACKGROUND_TASK_COMPLETED,
    EventTypes.CHILD_RUN_STARTED: TraceStepKinds.CHILD_RUN_STARTED,
    EventTypes.CHILD_RUN_COMPLETED: TraceStepKinds.CHILD_RUN_COMPLETED,
    EventTypes.CONVERSATION_INSERTED: TraceStepKinds.CONVERSATION_INSERT,
    EventTypes.PAUSE_REQUESTED: TraceStepKinds.PAUSE_REQUESTED,
    EventTypes.STATE_CHANGED: TraceStepKinds.STATE_CHANGED,
    EventTypes.CHECKPOINT: TraceStepKinds.CHECKPOINT,
    EventTypes.FINAL: TraceStepKinds.FINAL,
    EventTypes.ERROR: TraceStepKinds.ERROR,
    EventTypes.RUN_PAUSED: TraceStepKinds.RUN_PAUSED,
    EventTypes.RUN_COMPLETED: TraceStepKinds.RUN_COMPLETED,
}


class TraceRecorder:
    """Mutable builder for an immutable v0 trace payload."""

    __slots__ = ("_durable_step_count", "_next_step_id", "_run_id", "_steps")

    def __init__(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("trace run_id must not be empty")
        self._run_id = run_id
        self._durable_step_count = 0
        self._next_step_id = 1
        self._steps: list[dict[str, Any]] = []

    def record_event(self, event: AgentEvent) -> None:
        step = _step_from_event(self._next_step_id, event)
        if step is None:
            return
        self._steps.append(step)
        self._next_step_id += 1
        if event.type == EventTypes.RUN_STARTED:
            self._durable_step_count = len(self._steps)

    def record_resume(self, resume_input: ResumeInput, restored_state: AgentState) -> None:
        snapshot = resume_input.snapshot
        pause = snapshot.state.pause
        payload: dict[str, Any] = {
            "snapshot_status": snapshot.state.status.value,
            "restored_status": restored_state.status.value,
            "append_message_roles": [message.role for message in resume_input.append_messages],
            "append_message_count": len(resume_input.append_messages),
            "metadata_keys": sorted(str(key) for key in resume_input.metadata),
            "expected_pause": None
            if resume_input.expected_pause is None
            else _compact_pause_selector(resume_input.expected_pause),
        }
        if pause is not None:
            payload["pause"] = _compact_pause(pause.to_dict())
        self._steps.append(
            _trace_step(
                step_id=self._next_step_id,
                kind=TraceStepKinds.RESUME,
                before_status=snapshot.state.status,
                after_status=restored_state.status,
                references={},
                payload=payload,
                schema_version="v0",
            )
        )
        self._next_step_id += 1

    def to_payload(self, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "steps": deepcopy(self._steps),
            "metadata": _compact_trace_metadata(metadata or {}),
            "schema_version": "v0",
        }

    def rollback_to_durable(self) -> None:
        del self._steps[self._durable_step_count :]
        self._next_step_id = len(self._steps) + 1

    def mark_durable(self) -> None:
        self._durable_step_count = len(self._steps)

    def has_kind(self, kind: str) -> bool:
        return any(step["kind"] == kind for step in self._steps)

    def kinds(self) -> tuple[str, ...]:
        return tuple(_expect_str(step["kind"], "trace step kind") for step in self._steps)


def _trace_step(
    *,
    step_id: int,
    kind: str,
    before_status: AgentStatus | None,
    after_status: AgentStatus | None,
    references: Mapping[str, Any],
    payload: Mapping[str, Any],
    schema_version: str,
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "kind": kind,
        "before_status": _status_value(before_status),
        "after_status": _status_value(after_status),
        "references": deepcopy(dict(references)),
        "payload": deepcopy(dict(payload)),
        "schema_version": schema_version,
    }


def _step_from_event(step_id: int, event: AgentEvent) -> dict[str, Any] | None:
    kind = EVENT_TRACE_KIND.get(event.type)
    if kind is None:
        return None
    before_status, after_status = _statuses_from_event(kind, event.data)
    payload = _payload_from_event(kind, event.data)
    return _trace_step(
        step_id=step_id,
        kind=kind,
        before_status=before_status,
        after_status=after_status,
        references={"event_sequence": event.sequence, "event_type": event.type},
        payload=payload,
        schema_version=event.schema_version,
    )


def _statuses_from_event(
    kind: str, data: Mapping[str, Any]
) -> tuple[AgentStatus | None, AgentStatus | None]:
    if kind == TraceStepKinds.RUN_STARTED:
        return None, _snapshot_status(data)
    if kind == TraceStepKinds.STATE_CHANGED:
        return (
            AgentStatus(_expect_str(data["from"], "state_changed from")),
            AgentStatus(_expect_str(data["to"], "state_changed to")),
        )
    if kind == TraceStepKinds.CHECKPOINT:
        status = _snapshot_status(data)
        return status, status
    if kind == TraceStepKinds.ERROR:
        status = AgentStatus(_expect_str(data["status"], "error status"))
        return status, status
    if kind == TraceStepKinds.FINAL:
        return AgentStatus.COMPLETED, AgentStatus.COMPLETED
    if kind == TraceStepKinds.RUN_PAUSED:
        return AgentStatus.PAUSED, AgentStatus.PAUSED
    if kind == TraceStepKinds.RUN_COMPLETED:
        status = _snapshot_status(data)
        return status, status
    return None, None


def _payload_from_event(kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
    if kind == TraceStepKinds.RUN_STARTED:
        state = _expect_mapping(data["state"], "run_started state")
        return _compact_state_summary(state, "run_started state")
    if kind == TraceStepKinds.CHECKPOINT:
        state = _expect_mapping(data["state"], "checkpoint state")
        payload = _compact_checkpoint_state(state)
        payload["context_sequence"] = _expect_mapping(data["context"], "checkpoint context")[
            "sequence"
        ]
        return payload
    if kind == TraceStepKinds.PAUSE_REQUESTED:
        request = _expect_mapping(data["request"], "pause request")
        return {
            "reason": request["reason"],
            "source": request["source"],
            "wait_id": request["wait_id"],
            "metadata_keys": sorted(
                str(key) for key in _expect_mapping(request["metadata"], "pause request metadata")
            ),
            "interrupt": request["interrupt"],
            "resume_status": data["resume_status"],
            "origin": data["origin"],
        }
    if kind == TraceStepKinds.TOOL_CALL:
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "batch_id": data["batch_id"],
            "parallel": data["parallel"],
            "index": data["index"],
            "implementation_invoked": data["implementation_invoked"],
        }
    if kind == TraceStepKinds.TOOL_PROGRESS:
        progress = _expect_mapping(data["progress"], "tool progress")
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "batch_id": data["batch_id"],
            "parallel": data["parallel"],
            "index": data["index"],
            "implementation_invoked": True,
            "progress_keys": sorted(str(key) for key in progress),
        }
    if kind == TraceStepKinds.TOOL_CANCEL_REQUESTED:
        metadata = _expect_mapping(data["metadata"], "tool cancel metadata")
        return {
            "id": data["id"],
            "reason": data["reason"],
            "source": data["source"],
            "metadata_keys": sorted(str(key) for key in metadata),
        }
    if kind == TraceStepKinds.APPROVAL_REQUESTED:
        risk = _expect_mapping(data["risk"], "approval risk")
        metadata = _expect_mapping(data["metadata"], "approval request metadata")
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "risk_keys": sorted(str(key) for key in risk),
            "metadata_keys": sorted(str(key) for key in metadata),
        }
    if kind == TraceStepKinds.APPROVAL_COMPLETED:
        metadata = _expect_mapping(data["metadata"], "approval decision metadata")
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "action": data["action"],
            "reason": data["reason"],
            "metadata_keys": sorted(str(key) for key in metadata),
        }
    if kind == TraceStepKinds.TOOL_RESULT:
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "batch_id": data["batch_id"],
            "parallel": data["parallel"],
            "index": data["index"],
            "implementation_invoked": data["implementation_invoked"],
            "result": _compact_tool_result_summary(
                _expect_mapping(data["result"], "tool result summary")
            ),
        }
    if kind in {
        TraceStepKinds.BACKGROUND_TASK_STARTED,
        TraceStepKinds.BACKGROUND_TASK_UPDATED,
        TraceStepKinds.BACKGROUND_TASK_COMPLETED,
    }:
        return _compact_background_task_event(data)
    if kind in {TraceStepKinds.CHILD_RUN_STARTED, TraceStepKinds.CHILD_RUN_COMPLETED}:
        return deepcopy(dict(data))
    if kind == TraceStepKinds.CONVERSATION_INSERT:
        return _compact_conversation_insert(_expect_mapping(data["insert"], "conversation insert"))
    if kind == TraceStepKinds.MODEL_CALL:
        return deepcopy(dict(data))
    if kind == TraceStepKinds.MODEL_ERROR:
        return _compact_model_error(data)
    if kind == TraceStepKinds.MODEL_RESULT:
        return _compact_model_result_summary(data)
    if kind == TraceStepKinds.MODEL_DELTA:
        payload = _compact_model_delta(data)
        text_delta = payload.get("text_delta")
        if isinstance(text_delta, str):
            payload["text_delta_length"] = len(text_delta)
            del payload["text_delta"]
        arguments_delta = payload.get("arguments_delta")
        if isinstance(arguments_delta, str):
            payload["arguments_delta_length"] = len(arguments_delta)
            del payload["arguments_delta"]
        return payload
    if kind == TraceStepKinds.STATE_CHANGED:
        return _compact_state_change(data)
    if kind == TraceStepKinds.RUN_COMPLETED:
        return {
            "state": _compact_state_summary(
                _expect_mapping(data["state"], "completed state"), "completed state"
            )
        }
    if kind == TraceStepKinds.RUN_PAUSED:
        return {"pause": _compact_pause(data["pause"])}
    if kind == TraceStepKinds.FINAL:
        return _compact_final(data)
    if kind == TraceStepKinds.ERROR:
        return deepcopy(dict(data))
    return {}


def _compact_tool_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _expect_mapping(result["metadata"], "tool result metadata")
    payload = {
        "part_count": result["part_count"],
        "part_types": list(_expect_sequence(result["part_types"], "tool result part_types")),
        "text_length": result["text_length"],
        "result_kind": result["result_kind"],
        "is_error": result["is_error"],
        "metadata_keys": sorted(str(key) for key in metadata),
        "pause": _compact_pause_request(result["pause"]),
    }
    if "correlation_id" in result:
        payload["correlation_id"] = result["correlation_id"]
    if "background_task" in result:
        payload["background_task"] = _compact_background_task_summary(
            _expect_mapping(result["background_task"], "tool result background_task")
        )
    return payload


def _compact_background_task_summary(task: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _expect_mapping(task["metadata"], "background task metadata")
    payload: dict[str, Any] = {
        "id": task["id"],
        "status": task["status"],
        "kind": task["kind"],
        "lifecycle": task["lifecycle"],
        "metadata_keys": sorted(str(key) for key in metadata),
    }
    if "correlation_id" in task:
        payload["correlation_id"] = task["correlation_id"]
    return payload


def _compact_background_task_event(data: Mapping[str, Any]) -> dict[str, Any]:
    task = _expect_mapping(data["task"], "background task")
    payload = _compact_background_task_summary(task)
    payload["tool_call"] = _compact_background_task_tool_call(
        _expect_mapping(data["tool_call"], "background task tool_call")
    )
    return payload


def _compact_background_task_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": tool_call["id"],
        "name": tool_call["name"],
        "mode": tool_call["mode"],
        "batch_id": tool_call["batch_id"],
        "parallel": tool_call["parallel"],
        "index": tool_call["index"],
        "implementation_invoked": tool_call["implementation_invoked"],
    }


def _compact_conversation_insert(insert: Mapping[str, Any]) -> dict[str, Any]:
    parts = [
        _expect_mapping(part, "conversation insert part")
        for part in _expect_sequence(insert["parts"], "conversation insert parts")
    ]
    part_types: list[str] = []
    seen: set[str] = set()
    text_length = 0
    for part in parts:
        part_type = _expect_str(part["type"], "conversation insert part type")
        if part_type not in seen:
            seen.add(part_type)
            part_types.append(part_type)
        if part_type == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_length += len(text)
    metadata = _expect_mapping(insert.get("metadata", {}), "conversation insert metadata")
    payload: dict[str, Any] = {
        "id": insert["id"],
        "source": insert["source"],
        "part_count": len(parts),
        "part_types": part_types,
        "text_length": text_length,
        "metadata_keys": sorted(str(key) for key in metadata),
    }
    if "correlation_id" in insert:
        payload["correlation_id"] = insert["correlation_id"]
    return payload


def _compact_model_result_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "part_count": data["part_count"],
        "part_types": list(_expect_sequence(data["part_types"], "model result part_types")),
        "text_length": data["text_length"],
        "tool_call_count": data["tool_call_count"],
        "has_tool_calls": data["has_tool_calls"],
    }
    if "finish_reason" in data:
        payload["finish_reason"] = data["finish_reason"]
    if "model" in data:
        payload["model"] = data["model"]
    if "response_id" in data:
        payload["response_id"] = data["response_id"]
    if "usage" in data:
        payload["usage"] = _compact_usage(_expect_mapping(data["usage"], "model result usage"))
    return payload


def _compact_model_error(data: Mapping[str, Any]) -> dict[str, Any]:
    error = ModelErrorInfo.from_dict(_expect_mapping(data["error"], "model error"))
    error_payload = error.to_dict()
    payload: dict[str, Any] = {
        "message": error.message,
        "retry": data["retry"],
        "retryable": error.retryable,
        "metadata_keys": sorted(str(key) for key in error.metadata),
    }
    for key in ("provider", "code", "status_code", "request_id"):
        if key in error_payload:
            payload[key] = error_payload[key]
    return payload


def _compact_model_delta(data: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(data))
    raw_metadata = payload.pop("metadata", None)
    if raw_metadata is not None:
        payload["metadata_keys"] = sorted(
            str(key) for key in _expect_mapping(raw_metadata, "model delta metadata")
        )
    if "usage" in payload:
        payload["usage"] = _compact_usage(_expect_mapping(payload["usage"], "model delta usage"))
    return payload


def _compact_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(usage))
    raw_metadata = payload.pop("metadata", None)
    if raw_metadata is not None:
        payload["metadata_keys"] = sorted(
            str(key) for key in _expect_mapping(raw_metadata, "model usage metadata")
        )
    return payload


def _compact_usage_or_null(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    return _compact_usage(_expect_mapping(value, "model usage"))


def _compact_state_change(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "from": data["from"],
        "to": data["to"],
        "iterations": data["iterations"],
        "total_tool_calls": data["total_tool_calls"],
        "total_usage": _compact_usage_or_null(data["total_usage"]),
        "error": data["error"],
        "pause": _compact_pause(data.get("pause")),
    }


def _compact_final(data: Mapping[str, Any]) -> dict[str, Any]:
    summary = _expect_mapping(data["summary"], "final summary")
    parts = [
        _expect_mapping(part, "final part")
        for part in _expect_sequence(data["parts"], "final parts")
    ]
    metadata_keys = {
        str(key)
        for part in parts
        for key in _expect_mapping(part.get("metadata", {}), "final part metadata")
    }
    return {
        "part_count": summary["part_count"],
        "part_types": list(_expect_sequence(summary["part_types"], "final part_types")),
        "text_length": summary["text_length"],
        "metadata_keys": sorted(metadata_keys),
    }


def _compact_state_summary(state: Mapping[str, Any], label: str) -> dict[str, Any]:
    _reject_unknown_keys(
        state,
        {
            "status",
            "message_count",
            "message_roles",
            "pending_tool_call_count",
            "pending_tool_call_ids",
            "iterations",
            "total_tool_calls",
            "total_usage",
            "has_final",
            "final_part_count",
            "error",
            "pause",
        },
        label,
    )
    message_count = _expect_nonnegative_int(state["message_count"], f"{label} message_count")
    message_roles = [
        _expect_str(role, "state message role")
        for role in _expect_sequence(state["message_roles"], "state message_roles")
    ]
    if len(message_roles) != message_count:
        raise ValueError(f"{label} message_roles length must match message_count")

    pending_tool_call_count = _expect_nonnegative_int(
        state["pending_tool_call_count"], f"{label} pending_tool_call_count"
    )
    pending_tool_call_ids = [
        _expect_str(call_id, "pending tool call id")
        for call_id in _expect_sequence(state["pending_tool_call_ids"], "pending tool_call_ids")
    ]
    if len(pending_tool_call_ids) != pending_tool_call_count:
        raise ValueError(f"{label} pending_tool_call_ids length must match pending_tool_call_count")
    if len(pending_tool_call_ids) != len(set(pending_tool_call_ids)):
        raise ValueError(f"{label} pending_tool_call_ids must be unique")

    final_part_count = _expect_nonnegative_int(
        state["final_part_count"], f"{label} final_part_count"
    )
    has_final = _expect_bool(state["has_final"], f"{label} has_final")
    if has_final != (final_part_count > 0):
        raise ValueError(f"{label} has_final must match final_part_count")

    return {
        "status": state["status"],
        "message_count": message_count,
        "message_roles": message_roles,
        "pending_tool_call_ids": pending_tool_call_ids,
        "iterations": state["iterations"],
        "total_tool_calls": state["total_tool_calls"],
        "total_usage": _compact_usage_or_null(state["total_usage"]),
        "final_part_count": final_part_count,
        "error": state["error"],
        "pause": _compact_pause(state["pause"]),
    }


def _compact_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(
        state,
        {
            "status",
            "messages",
            "pending_tool_calls",
            "iterations",
            "total_tool_calls",
            "total_usage",
            "final_parts",
            "error",
            "pause",
        },
        "checkpoint state",
    )
    return _compact_state_snapshot(state)


def _compact_state_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    if "messages" in state:
        messages = [
            _expect_mapping(message, "state message")
            for message in _expect_sequence(state["messages"], "state messages")
        ]
        message_roles = [_expect_str(message["role"], "message role") for message in messages]
        message_count = len(messages)
    else:
        message_count = _expect_nonnegative_int(state["message_count"], "state message_count")
        message_roles = [
            _expect_str(role, "state message role")
            for role in _expect_sequence(state["message_roles"], "state message_roles")
        ]

    if "pending_tool_calls" in state:
        pending = [
            _expect_mapping(call, "pending tool call")
            for call in _expect_sequence(state["pending_tool_calls"], "pending tool calls")
        ]
        pending_tool_call_ids = [
            _expect_str(call["id"], "pending tool call id") for call in pending
        ]
    else:
        pending_tool_call_ids = [
            _expect_str(call_id, "pending tool call id")
            for call_id in _expect_sequence(state["pending_tool_call_ids"], "pending tool_call_ids")
        ]

    if "final_parts" in state:
        final_part_count = len(_expect_sequence(state["final_parts"], "state final_parts"))
    else:
        final_part_count = _expect_nonnegative_int(
            state["final_part_count"], "state final_part_count"
        )

    return {
        "status": state["status"],
        "message_count": message_count,
        "message_roles": message_roles,
        "pending_tool_call_ids": pending_tool_call_ids,
        "iterations": state["iterations"],
        "total_tool_calls": state["total_tool_calls"],
        "total_usage": _compact_usage_or_null(state["total_usage"]),
        "final_part_count": final_part_count,
        "error": state["error"],
        "pause": _compact_pause(state["pause"]),
    }


def _compact_pause(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    pause = _expect_mapping(value, "pause")
    return {
        "reason": pause["reason"],
        "resume_status": pause["resume_status"],
        "source": pause["source"],
        "wait_id": pause["wait_id"],
        "metadata_keys": sorted(
            str(key) for key in _expect_mapping(pause["metadata"], "pause metadata")
        ),
    }


def _compact_pause_request(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    request = _expect_mapping(value, "pause request")
    return {
        "reason": request["reason"],
        "source": request["source"],
        "wait_id": request["wait_id"],
        "metadata_keys": sorted(
            str(key) for key in _expect_mapping(request["metadata"], "pause request metadata")
        ),
        "interrupt": request["interrupt"],
    }


def _compact_pause_selector(selector: Any) -> dict[str, Any]:
    payload = selector.to_dict()
    return {
        "reason": payload["reason"],
        "source": payload["source"],
        "wait_id": payload["wait_id"],
        "metadata_keys": sorted(str(key) for key in payload["metadata"]),
    }


def _snapshot_status(data: Mapping[str, Any]) -> AgentStatus:
    state = _expect_mapping(data["state"], "snapshot state")
    return AgentStatus(_expect_str(state["status"], "snapshot status"))
