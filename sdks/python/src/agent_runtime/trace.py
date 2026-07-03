"""Portable run trace and deterministic replay validation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

from agent_runtime._frozen import freeze_value, thaw_value
from agent_runtime.events import AgentEvent, EventTypes
from agent_runtime.resume import ResumeInput
from agent_runtime.state import AgentState, AgentStatus


def _empty_mapping() -> Mapping[str, Any]:
    return {}


def _empty_trace_metadata() -> Mapping[str, Any]:
    return {"metadata_keys": []}


def _empty_steps() -> tuple[TraceStep, ...]:
    return ()


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _expect_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[object], value)


def _expect_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _expect_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string or null")
    return value


def _expect_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be > 0")
    return value


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} has unknown field(s): {names}")


def _compact_trace_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"metadata_keys": sorted(str(key) for key in value)}


def _expect_trace_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    _reject_unknown_keys(value, {"metadata_keys"}, "trace metadata")
    keys = _expect_sequence(value["metadata_keys"], "trace metadata_keys")
    return {"metadata_keys": [_expect_str(key, "trace metadata key") for key in keys]}


def _freeze_value(value: object) -> object:
    return freeze_value(value, error_message="trace data is immutable")


def _thaw_value(value: object) -> object:
    return thaw_value(value)


class TraceStepKinds:
    RUN_STARTED = "run_started"
    RESUME = "resume"
    MODEL_CALL = "model_call"
    MODEL_DELTA = "model_delta"
    MODEL_ERROR = "model_error"
    MODEL_RESULT = "model_result"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONVERSATION_INSERT = "conversation_insert"
    PAUSE_REQUESTED = "pause_requested"
    STATE_CHANGED = "state_changed"
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ERROR = "error"
    RUN_PAUSED = "run_paused"
    RUN_COMPLETED = "run_completed"


VALID_TRACE_STEP_KINDS = {
    TraceStepKinds.RUN_STARTED,
    TraceStepKinds.RESUME,
    TraceStepKinds.MODEL_CALL,
    TraceStepKinds.MODEL_DELTA,
    TraceStepKinds.MODEL_ERROR,
    TraceStepKinds.MODEL_RESULT,
    TraceStepKinds.TOOL_CALL,
    TraceStepKinds.TOOL_RESULT,
    TraceStepKinds.CONVERSATION_INSERT,
    TraceStepKinds.PAUSE_REQUESTED,
    TraceStepKinds.STATE_CHANGED,
    TraceStepKinds.CHECKPOINT,
    TraceStepKinds.FINAL,
    TraceStepKinds.ERROR,
    TraceStepKinds.RUN_PAUSED,
    TraceStepKinds.RUN_COMPLETED,
}

_RESERVED_TOOL_RESULT_KINDS = {"observation", "acceptance", "rejection"}


VALID_STATE_TRANSITIONS = {
    AgentStatus.PLANNING: {
        AgentStatus.EXECUTING_TOOLS,
        AgentStatus.PAUSED,
        AgentStatus.COMPLETED,
        AgentStatus.FAILED,
        AgentStatus.LIMIT_EXCEEDED,
    },
    AgentStatus.EXECUTING_TOOLS: {
        AgentStatus.PLANNING,
        AgentStatus.PAUSED,
        AgentStatus.FAILED,
        AgentStatus.LIMIT_EXCEEDED,
    },
}


EVENT_TRACE_KIND: Mapping[str, str] = {
    EventTypes.RUN_STARTED: TraceStepKinds.RUN_STARTED,
    EventTypes.MODEL_STARTED: TraceStepKinds.MODEL_CALL,
    EventTypes.MODEL_DELTA: TraceStepKinds.MODEL_DELTA,
    EventTypes.MODEL_ERROR: TraceStepKinds.MODEL_ERROR,
    EventTypes.MODEL_COMPLETED: TraceStepKinds.MODEL_RESULT,
    EventTypes.TOOL_STARTED: TraceStepKinds.TOOL_CALL,
    EventTypes.TOOL_COMPLETED: TraceStepKinds.TOOL_RESULT,
    EventTypes.CONVERSATION_INSERTED: TraceStepKinds.CONVERSATION_INSERT,
    EventTypes.PAUSE_REQUESTED: TraceStepKinds.PAUSE_REQUESTED,
    EventTypes.STATE_CHANGED: TraceStepKinds.STATE_CHANGED,
    EventTypes.CHECKPOINT: TraceStepKinds.CHECKPOINT,
    EventTypes.FINAL: TraceStepKinds.FINAL,
    EventTypes.ERROR: TraceStepKinds.ERROR,
    EventTypes.RUN_PAUSED: TraceStepKinds.RUN_PAUSED,
    EventTypes.RUN_COMPLETED: TraceStepKinds.RUN_COMPLETED,
}


@dataclass(slots=True, frozen=True)
class TraceStep:
    """A compact semantic runtime step."""

    step_id: int
    kind: str
    before_status: AgentStatus | None = None
    after_status: AgentStatus | None = None
    references: Mapping[str, Any] = field(default_factory=_empty_mapping)
    payload: Mapping[str, Any] = field(default_factory=_empty_mapping)
    schema_version: str = "v0"

    def __post_init__(self) -> None:
        kind = _expect_str(self.kind, "trace step kind")
        if not kind:
            raise ValueError("trace step kind must not be empty")
        if kind not in VALID_TRACE_STEP_KINDS:
            raise ValueError(f"unsupported trace step kind: {kind}")
        if not isinstance(cast(object, self.before_status), AgentStatus | None):
            raise TypeError("trace step before_status must be an AgentStatus or None")
        if not isinstance(cast(object, self.after_status), AgentStatus | None):
            raise TypeError("trace step after_status must be an AgentStatus or None")
        object.__setattr__(
            self,
            "references",
            _freeze_value(_expect_mapping(self.references, "trace step references")),
        )
        object.__setattr__(
            self,
            "payload",
            _freeze_value(_expect_mapping(self.payload, "trace step payload")),
        )
        _expect_int(self.step_id, "trace step_id")
        schema_version = _expect_str(self.schema_version, "trace step schema_version")
        if not schema_version:
            raise ValueError("trace step schema_version must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TraceStep:
        known = {
            "step_id",
            "kind",
            "before_status",
            "after_status",
            "references",
            "payload",
            "schema_version",
        }
        _reject_unknown_keys(value, known, "trace step")
        return cls(
            step_id=_expect_int(value["step_id"], "trace step_id"),
            kind=_expect_str(value["kind"], "trace kind"),
            before_status=_status_or_none(value["before_status"], "trace before_status"),
            after_status=_status_or_none(value["after_status"], "trace after_status"),
            references=_expect_mapping(value["references"], "trace references"),
            payload=_expect_mapping(value["payload"], "trace payload"),
            schema_version=_expect_str(value["schema_version"], "trace schema_version"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step_id": self.step_id,
            "kind": self.kind,
            "before_status": None if self.before_status is None else self.before_status.value,
            "after_status": None if self.after_status is None else self.after_status.value,
            "references": cast(dict[str, Any], _thaw_value(self.references)),
            "payload": cast(dict[str, Any], _thaw_value(self.payload)),
            "schema_version": self.schema_version,
        }
        return data


@dataclass(slots=True, frozen=True)
class RunTrace:
    """Portable trace for one runtime invocation."""

    run_id: str
    steps: Sequence[TraceStep] = field(default_factory=_empty_steps)
    metadata: Mapping[str, Any] = field(default_factory=_empty_trace_metadata)
    schema_version: str = "v0"

    def __post_init__(self) -> None:
        run_id = _expect_str(self.run_id, "trace run_id")
        if not run_id:
            raise ValueError("trace run_id must not be empty")
        steps = tuple(TraceStep.from_dict(step.to_dict()) for step in self.steps)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "metadata",
            _freeze_value(_expect_trace_metadata(self.metadata)),
        )
        schema_version = _expect_str(self.schema_version, "trace schema_version")
        if not schema_version:
            raise ValueError("trace schema_version must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunTrace:
        known = {"run_id", "steps", "metadata", "schema_version"}
        _reject_unknown_keys(value, known, "run trace")
        trace = cls(
            run_id=_expect_str(value["run_id"], "trace run_id"),
            steps=[
                TraceStep.from_dict(_expect_mapping(step, "trace step"))
                for step in _expect_sequence(value["steps"], "trace steps")
            ],
            metadata=_expect_mapping(value["metadata"], "trace metadata"),
            schema_version=_expect_str(value["schema_version"], "trace schema_version"),
        )
        _validate_trace_wire_shape(trace)
        return trace

    @classmethod
    def from_events(
        cls,
        run_id: str,
        events: Sequence[AgentEvent],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunTrace:
        recorder = TraceRecorder(run_id)
        for event in events:
            recorder.record_event(event)
        return recorder.to_trace(metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "steps": [step.to_dict() for step in self.steps],
            "metadata": cast(dict[str, Any], _thaw_value(self.metadata)),
            "schema_version": self.schema_version,
        }
        return data


class TraceRecorder:
    """Mutable builder for an immutable RunTrace."""

    __slots__ = ("_durable_step_count", "_next_step_id", "_run_id", "_steps")

    def __init__(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("trace run_id must not be empty")
        self._run_id = run_id
        self._durable_step_count = 0
        self._next_step_id = 1
        self._steps: list[TraceStep] = []

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
            TraceStep(
                step_id=self._next_step_id,
                kind=TraceStepKinds.RESUME,
                before_status=snapshot.state.status,
                after_status=restored_state.status,
                payload=payload,
            )
        )
        self._next_step_id += 1

    def to_trace(self, *, metadata: Mapping[str, Any] | None = None) -> RunTrace:
        return RunTrace(
            run_id=self._run_id,
            steps=tuple(self._steps),
            metadata=_compact_trace_metadata(metadata or {}),
        )

    def rollback_to_durable(self) -> None:
        del self._steps[self._durable_step_count :]
        self._next_step_id = len(self._steps) + 1

    def mark_durable(self) -> None:
        self._durable_step_count = len(self._steps)


def _validate_trace_wire_shape(trace: RunTrace) -> None:
    _validate_metadata_key_summary(trace.metadata, "trace metadata")
    for step in trace.steps:
        _validate_reference_summary(step.references, "trace references")
        _validate_trace_payload(step.kind, step.payload)


def _validate_trace_payload(kind: str, payload: Mapping[str, Any]) -> None:
    if kind == TraceStepKinds.RUN_STARTED:
        _validate_compact_state(payload, "run_started payload", checkpoint=False)
    elif kind == TraceStepKinds.RESUME:
        data = _validate_object_shape(
            payload,
            {
                "snapshot_status",
                "restored_status",
                "append_message_roles",
                "append_message_count",
                "metadata_keys",
                "expected_pause",
            },
            {"pause"},
            "resume payload",
        )
        _expect_status_value(data["snapshot_status"], "resume snapshot_status")
        _expect_status_value(
            data["restored_status"],
            "resume restored_status",
            allowed={AgentStatus.PLANNING, AgentStatus.EXECUTING_TOOLS},
        )
        append_message_roles = _expect_role_list(
            data["append_message_roles"], "resume append_message_roles"
        )
        append_message_count = _expect_nonnegative_int(
            data["append_message_count"], "resume append_message_count"
        )
        if len(append_message_roles) != append_message_count:
            raise ValueError("resume append_message_roles length must match append_message_count")
        _expect_str_list(data["metadata_keys"], "resume metadata_keys")
        _validate_compact_pause_selector_or_null(data["expected_pause"], "resume expected_pause")
        if "pause" in data:
            _validate_compact_pause(data["pause"], "resume pause")
    elif kind == TraceStepKinds.MODEL_CALL:
        data = _validate_object_shape(payload, {"iteration"}, set(), "model_call payload")
        _expect_int(data["iteration"], "model_call iteration")
    elif kind == TraceStepKinds.MODEL_DELTA:
        _validate_model_delta_payload(payload)
    elif kind == TraceStepKinds.MODEL_ERROR:
        _validate_model_error_payload(payload)
    elif kind == TraceStepKinds.MODEL_RESULT:
        data = _validate_object_shape(
            payload,
            {"part_count", "part_types", "text_length", "tool_call_count", "has_tool_calls"},
            {"finish_reason", "usage", "model", "response_id"},
            "model_result payload",
        )
        _expect_nonnegative_int(data["part_count"], "model_result part_count")
        _expect_str_list(data["part_types"], "model_result part_types")
        _expect_nonnegative_int(data["text_length"], "model_result text_length")
        tool_call_count = _expect_nonnegative_int(
            data["tool_call_count"], "model_result tool_call_count"
        )
        has_tool_calls = _expect_bool(data["has_tool_calls"], "model_result has_tool_calls")
        if has_tool_calls != (tool_call_count > 0):
            raise ValueError("model_result has_tool_calls must match tool_call_count")
        if "finish_reason" in data:
            _expect_non_empty_str(data["finish_reason"], "model_result finish_reason")
        if "usage" in data:
            _validate_usage(data["usage"], "model_result usage")
        if "model" in data:
            _expect_non_empty_str(data["model"], "model_result model")
        if "response_id" in data:
            _expect_non_empty_str(data["response_id"], "model_result response_id")
    elif kind == TraceStepKinds.TOOL_CALL:
        _validate_tool_call_payload(payload, "tool_call payload")
    elif kind == TraceStepKinds.TOOL_RESULT:
        data = _validate_tool_call_payload(
            payload, "tool_result payload", extra_required={"result"}
        )
        result = _validate_tool_result_summary(data["result"], "tool_result result")
        mode = _expect_tool_mode(data["mode"], "tool_result mode")
        result_kind = _expect_str(result["result_kind"], "tool_result result_kind")
        if not _tool_result_kind_matches_mode(mode, result_kind):
            raise ValueError("tool_result result_kind must match tool invocation mode")
    elif kind == TraceStepKinds.CONVERSATION_INSERT:
        _validate_conversation_insert_payload(payload, "conversation_insert payload")
    elif kind == TraceStepKinds.PAUSE_REQUESTED:
        _validate_compact_pause_request(payload, "pause_requested payload", require_origin=True)
    elif kind == TraceStepKinds.STATE_CHANGED:
        data = _validate_object_shape(
            payload,
            {"from", "to", "iterations", "total_tool_calls", "total_usage", "error", "pause"},
            set(),
            "state_changed payload",
        )
        _expect_status_value(data["from"], "state_changed from")
        _expect_status_value(data["to"], "state_changed to")
        _expect_nonnegative_int(data["iterations"], "state_changed iterations")
        _expect_nonnegative_int(data["total_tool_calls"], "state_changed total_tool_calls")
        _validate_usage_or_null(data["total_usage"], "state_changed total_usage")
        _expect_optional_str(data["error"], "state_changed error")
        _validate_compact_pause_or_null(data["pause"], "state_changed pause")
    elif kind == TraceStepKinds.CHECKPOINT:
        _validate_compact_state(payload, "checkpoint payload", checkpoint=True)
    elif kind == TraceStepKinds.FINAL:
        data = _validate_object_shape(
            payload,
            {"part_count", "part_types", "text_length", "metadata_keys"},
            set(),
            "final payload",
        )
        _expect_nonnegative_int(data["part_count"], "final part_count")
        _expect_str_list(data["part_types"], "final part_types")
        _expect_nonnegative_int(data["text_length"], "final text_length")
        _expect_str_list(data["metadata_keys"], "final metadata_keys")
    elif kind == TraceStepKinds.ERROR:
        data = _validate_object_shape(payload, {"status", "message"}, set(), "error payload")
        _expect_status_value(data["status"], "error status")
        _expect_str(data["message"], "error message")
    elif kind == TraceStepKinds.RUN_PAUSED:
        data = _validate_object_shape(payload, {"pause"}, set(), "run_paused payload")
        _validate_compact_pause(data["pause"], "run_paused pause")
    elif kind == TraceStepKinds.RUN_COMPLETED:
        data = _validate_object_shape(payload, {"state"}, set(), "run_completed payload")
        _validate_compact_state(
            _expect_mapping(data["state"], "run_completed state"),
            "run_completed state",
            checkpoint=False,
        )
    else:
        raise ValueError(f"unsupported trace step kind: {kind}")


def _validate_object_shape(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    label: str,
) -> Mapping[str, Any]:
    allowed = required | optional
    _reject_unknown_keys(value, allowed, label)
    missing = required - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{label} missing required field(s): {names}")
    return value


def _validate_reference_summary(value: Mapping[str, Any], label: str) -> None:
    if not value:
        return
    data = _validate_object_shape(value, {"event_sequence", "event_type"}, set(), label)
    _expect_nonnegative_int(data["event_sequence"], f"{label} event_sequence")
    _expect_non_empty_str(data["event_type"], f"{label} event_type")


def _validate_metadata_key_summary(value: Mapping[str, Any], label: str) -> None:
    data = _validate_object_shape(value, {"metadata_keys"}, set(), label)
    _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")


def _validate_compact_state(
    value: Mapping[str, Any],
    label: str,
    *,
    checkpoint: bool,
) -> None:
    required = {
        "status",
        "message_count",
        "message_roles",
        "pending_tool_call_ids",
        "iterations",
        "total_tool_calls",
        "total_usage",
        "final_part_count",
        "error",
        "pause",
    }
    if checkpoint:
        required.add("context_sequence")
    data = _validate_object_shape(value, required, set(), label)
    _expect_status_value(data["status"], f"{label} status")
    message_count = _expect_nonnegative_int(data["message_count"], f"{label} message_count")
    message_roles = _expect_role_list(data["message_roles"], f"{label} message_roles")
    if len(message_roles) != message_count:
        raise ValueError(f"{label} message_roles length must match message_count")
    pending_tool_call_ids = _expect_str_list(
        data["pending_tool_call_ids"], f"{label} pending_tool_call_ids"
    )
    if len(pending_tool_call_ids) != len(set(pending_tool_call_ids)):
        raise ValueError(f"{label} pending_tool_call_ids must be unique")
    _expect_nonnegative_int(data["iterations"], f"{label} iterations")
    _expect_nonnegative_int(data["total_tool_calls"], f"{label} total_tool_calls")
    _validate_usage_or_null(data["total_usage"], f"{label} total_usage")
    _expect_nonnegative_int(data["final_part_count"], f"{label} final_part_count")
    _expect_optional_str(data["error"], f"{label} error")
    _validate_compact_pause_or_null(data["pause"], f"{label} pause")
    if checkpoint:
        _expect_nonnegative_int(data["context_sequence"], f"{label} context_sequence")


def _validate_model_delta_payload(value: Mapping[str, Any]) -> None:
    delta_kind = _expect_str(value.get("kind"), "model_delta kind")
    if delta_kind == "text_delta":
        data = _validate_object_shape(
            value,
            {"kind", "index", "text_delta_length", "part_type"},
            {"metadata_keys"},
            "text_delta payload",
        )
        _expect_nonnegative_int(data["index"], "text_delta index")
        _expect_nonnegative_int(data["text_delta_length"], "text_delta text_delta_length")
        _expect_non_empty_str(data["part_type"], "text_delta part_type")
    elif delta_kind == "tool_call_delta":
        data = _validate_object_shape(
            value,
            {"kind", "index"},
            {"id", "name", "mode", "arguments_delta_length", "metadata_keys"},
            "tool_call_delta payload",
        )
        _expect_nonnegative_int(data["index"], "tool_call_delta index")
        if "id" in data:
            _expect_non_empty_str(data["id"], "tool_call_delta id")
        if "name" in data:
            _expect_non_empty_str(data["name"], "tool_call_delta name")
        if "mode" in data:
            _expect_tool_mode(data["mode"], "tool_call_delta mode")
        if "arguments_delta_length" in data:
            _expect_nonnegative_int(
                data["arguments_delta_length"], "tool_call_delta arguments_delta_length"
            )
    elif delta_kind == "reasoning_delta":
        data = _validate_object_shape(
            value,
            {"kind", "index", "text_delta_length"},
            {"metadata_keys"},
            "reasoning_delta payload",
        )
        _expect_nonnegative_int(data["index"], "reasoning_delta index")
        _expect_nonnegative_int(data["text_delta_length"], "reasoning_delta text_delta_length")
    elif delta_kind == "usage_delta":
        data = _validate_object_shape(
            value,
            {"kind", "usage"},
            {"metadata_keys"},
            "usage_delta payload",
        )
        _validate_usage(data["usage"], "usage_delta usage")
    else:
        raise ValueError(f"unsupported model_delta kind: {delta_kind}")
    if "metadata_keys" in data:
        _expect_str_list(data["metadata_keys"], f"{delta_kind} metadata_keys")


def _validate_usage(value: object, label: str) -> None:
    data = _expect_mapping(value, label)
    token_fields = {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    }
    data = _validate_object_shape(data, set(), token_fields | {"metadata_keys"}, label)
    for field_name in token_fields:
        if field_name in data and data[field_name] is not None:
            _expect_nonnegative_int(data[field_name], f"{label} {field_name}")
    if "metadata_keys" in data:
        _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")


def _validate_model_error_payload(value: Mapping[str, Any]) -> None:
    data = _validate_object_shape(
        value,
        {"message", "retry", "retryable", "metadata_keys"},
        {"provider", "code", "status_code", "request_id"},
        "model_error payload",
    )
    _expect_non_empty_str(data["message"], "model_error message")
    _expect_bool(data["retry"], "model_error retry")
    _expect_bool(data["retryable"], "model_error retryable")
    if "provider" in data:
        _expect_non_empty_str(data["provider"], "model_error provider")
    if "code" in data:
        _expect_non_empty_str(data["code"], "model_error code")
    if "status_code" in data:
        status_code = _expect_int(data["status_code"], "model_error status_code")
        if status_code < 100:
            raise ValueError("model_error status_code must be >= 100")
    if "request_id" in data:
        _expect_non_empty_str(data["request_id"], "model_error request_id")
    _expect_str_list(data["metadata_keys"], "model_error metadata_keys")


def _validate_usage_or_null(value: object, label: str) -> None:
    if value is None:
        return
    _validate_usage(value, label)


def _validate_tool_call_payload(
    value: Mapping[str, Any],
    label: str,
    *,
    extra_required: set[str] | None = None,
) -> Mapping[str, Any]:
    required = {"id", "name", "mode", "batch_id", "parallel", "index"} | (extra_required or set())
    data = _validate_object_shape(value, required, set(), label)
    _expect_non_empty_str(data["id"], f"{label} id")
    _expect_non_empty_str(data["name"], f"{label} name")
    _expect_tool_mode(data["mode"], f"{label} mode")
    _expect_non_empty_str(data["batch_id"], f"{label} batch_id")
    _expect_bool(data["parallel"], f"{label} parallel")
    _expect_nonnegative_int(data["index"], f"{label} index")
    return data


def _validate_tool_result_summary(value: object, label: str) -> Mapping[str, Any]:
    data = _expect_mapping(value, label)
    data = _validate_object_shape(
        data,
        {
            "part_count",
            "part_types",
            "text_length",
            "result_kind",
            "is_error",
            "metadata_keys",
            "pause",
        },
        {"correlation_id"},
        label,
    )
    _expect_nonnegative_int(data["part_count"], f"{label} part_count")
    _expect_str_list(data["part_types"], f"{label} part_types")
    _expect_nonnegative_int(data["text_length"], f"{label} text_length")
    result_kind = _expect_str(data["result_kind"], f"{label} result_kind")
    if not result_kind:
        raise ValueError(f"{label} result_kind must not be empty")
    _expect_bool(data["is_error"], f"{label} is_error")
    if "correlation_id" in data:
        _expect_non_empty_str(data["correlation_id"], f"{label} correlation_id")
    if result_kind == "acceptance":
        if "correlation_id" not in data:
            raise ValueError(f"{label} acceptance requires correlation_id")
        if data["is_error"]:
            raise ValueError(f"{label} acceptance is_error must be false")
        if data["pause"] is not None:
            raise ValueError(f"{label} acceptance pause must be null")
    if result_kind == "rejection":
        if not data["is_error"]:
            raise ValueError(f"{label} rejection is_error must be true")
        if data["pause"] is not None:
            raise ValueError(f"{label} rejection pause must be null")
    _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")
    _validate_compact_pause_request_or_null(data["pause"], f"{label} pause", tool_only=True)
    return data


def _tool_result_kind_matches_mode(mode: str, result_kind: str) -> bool:
    if mode == "execute":
        return result_kind == "observation"
    if mode == "accept":
        return result_kind in {"acceptance", "rejection"}
    return result_kind not in _RESERVED_TOOL_RESULT_KINDS


def _validate_conversation_insert_payload(value: Mapping[str, Any], label: str) -> None:
    data = _validate_object_shape(
        value,
        {"id", "source", "part_count", "part_types", "text_length", "metadata_keys"},
        {"correlation_id"},
        label,
    )
    _expect_non_empty_str(data["id"], f"{label} id")
    _expect_non_empty_str(data["source"], f"{label} source")
    if "correlation_id" in data:
        _expect_non_empty_str(data["correlation_id"], f"{label} correlation_id")
    _expect_nonnegative_int(data["part_count"], f"{label} part_count")
    _expect_str_list(data["part_types"], f"{label} part_types")
    _expect_nonnegative_int(data["text_length"], f"{label} text_length")
    _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")


def _validate_compact_pause_or_null(value: object, label: str) -> None:
    if value is None:
        return
    _validate_compact_pause(value, label)


def _validate_compact_pause(value: object, label: str) -> None:
    data = _expect_mapping(value, label)
    data = _validate_object_shape(
        data,
        {"reason", "resume_status", "source", "wait_id", "metadata_keys"},
        set(),
        label,
    )
    _expect_non_empty_str(data["reason"], f"{label} reason")
    _expect_status_value(
        data["resume_status"],
        f"{label} resume_status",
        allowed={AgentStatus.PLANNING, AgentStatus.EXECUTING_TOOLS},
    )
    _expect_non_empty_str(data["source"], f"{label} source")
    _expect_optional_str(data["wait_id"], f"{label} wait_id")
    _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")


def _validate_compact_pause_request_or_null(
    value: object,
    label: str,
    *,
    tool_only: bool,
) -> None:
    if value is None:
        return
    _validate_compact_pause_request(value, label, tool_only=tool_only, require_origin=False)


def _validate_compact_pause_request(
    value: object,
    label: str,
    *,
    tool_only: bool = False,
    require_origin: bool = False,
) -> None:
    data = _expect_mapping(value, label)
    required = {"reason", "source", "wait_id", "metadata_keys", "interrupt"}
    if require_origin:
        required |= {"resume_status", "origin"}
    data = _validate_object_shape(data, required, set(), label)
    _expect_non_empty_str(data["reason"], f"{label} reason")
    _expect_non_empty_str(data["source"], f"{label} source")
    _expect_optional_str(data["wait_id"], f"{label} wait_id")
    _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")
    interrupt = _expect_bool(data["interrupt"], f"{label} interrupt")
    if tool_only and interrupt:
        raise ValueError(f"{label} interrupt must be false")
    if require_origin:
        _expect_status_value(
            data["resume_status"],
            f"{label} resume_status",
            allowed={AgentStatus.PLANNING, AgentStatus.EXECUTING_TOOLS},
        )
        origin = _expect_str(data["origin"], f"{label} origin")
        if origin not in {"control", "tool_result"}:
            raise ValueError(f"{label} origin must be control or tool_result")
        if origin == "tool_result" and interrupt:
            raise ValueError(f"{label} interrupt must be false for tool_result origin")


def _validate_compact_pause_selector_or_null(value: object, label: str) -> None:
    if value is None:
        return
    data = _expect_mapping(value, label)
    data = _validate_object_shape(
        data,
        {"reason", "source", "wait_id", "metadata_keys"},
        set(),
        label,
    )
    reason = _expect_optional_str(data["reason"], f"{label} reason")
    source = _expect_optional_str(data["source"], f"{label} source")
    wait_id = _expect_optional_str(data["wait_id"], f"{label} wait_id")
    metadata_keys = _expect_str_list(data["metadata_keys"], f"{label} metadata_keys")
    if reason == "" or source == "":
        raise ValueError(f"{label} selector text must not be empty")
    if reason is None and source is None and wait_id is None and not metadata_keys:
        raise ValueError(f"{label} must set at least one selector field")


def _expect_status_value(
    value: object,
    label: str,
    *,
    allowed: set[AgentStatus] | None = None,
) -> AgentStatus:
    status = AgentStatus(_expect_str(value, label))
    if allowed is not None and status not in allowed:
        allowed_text = ", ".join(sorted(item.value for item in allowed))
        raise ValueError(f"{label} must be one of: {allowed_text}")
    return status


def _expect_non_empty_str(value: object, label: str) -> str:
    text = _expect_str(value, label)
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _expect_tool_mode(value: object, label: str) -> str:
    mode = _expect_str(value, label)
    if not mode:
        raise ValueError(f"{label} must not be empty")
    return mode


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _expect_str_list(value: object, label: str) -> list[str]:
    return [_expect_str(item, label) for item in _expect_sequence(value, label)]


def _expect_role_list(value: object, label: str) -> list[str]:
    roles = _expect_str_list(value, label)
    for role in roles:
        if role not in {"system", "user", "assistant", "tool", "external"}:
            raise ValueError(f"{label} contains unsupported role: {role}")
    return roles


@dataclass(slots=True, frozen=True)
class ReplayResult:
    """Replay validation result."""

    valid: bool
    steps: int
    final_status: AgentStatus | None = None
    message: str | None = None


class ReplayError(AssertionError):
    """Raised when deterministic replay detects a trace mismatch."""


def replay_trace(trace: RunTrace, *, strict: bool = True) -> ReplayResult:
    """Validate a trace without calling models or tools."""

    try:
        canonical = RunTrace.from_dict(trace.to_dict())
        result = _ReplayValidator(canonical).validate()
    except (ReplayError, TypeError, ValueError, KeyError) as exc:
        replay_error = exc if isinstance(exc, ReplayError) else ReplayError(str(exc))
        if strict:
            raise replay_error from exc
        return ReplayResult(False, len(trace.steps), message=str(replay_error))
    if strict and not result.valid:
        raise ReplayError(result.message or "trace replay failed")
    return result


class _ReplayValidator:
    def __init__(self, trace: RunTrace) -> None:
        self.trace = RunTrace.from_dict(trace.to_dict())
        self.current_status: AgentStatus | None = None
        self.durable_message_count = 0
        self.stream_baseline_message_count = 0
        self.pending_stream_delta = False
        self.model_call_open = False
        self.model_result_ready = False
        self.model_result_tool_call_count = 0
        self.open_tool_calls: dict[str, Mapping[str, Any]] = {}
        self.tool_call_ids_since_checkpoint: list[str] = []
        self.tool_result_count = 0
        self.tool_result_ready = False
        self.final_part_count: int | None = None
        self.total_tool_calls_baseline = 0
        self.last_reported_total_tool_calls = 0
        self.expected_tool_result_count: int | None = None
        self.expected_pending_checkpoint_count: int | None = None
        self.pending_tool_call_ids: tuple[str, ...] = ()
        self.tool_result_ids_since_checkpoint: list[str] = []
        self.pending_tool_pause_requests: list[tuple[int, Mapping[str, Any]]] = []
        self.pending_pause_request: Mapping[str, Any] | None = None
        self.pending_conversation_insert_baseline = 0
        self.pending_conversation_insert_count = 0
        self.run_completed_seen = False
        self.last_checkpoint_status: AgentStatus | None = None
        self.last_checkpoint_payload: Mapping[str, Any] | None = None
        self.last_checkpoint_index = -1

    def validate(self) -> ReplayResult:
        steps = tuple(self.trace.steps)
        if not steps:
            raise ReplayError("trace must contain at least one step")
        self._validate_step_ids(steps)
        start_index = 0
        if steps[0].kind == TraceStepKinds.RESUME:
            self._validate_resume(steps[0])
            start_index = 1
        if start_index >= len(steps) or steps[start_index].kind != TraceStepKinds.RUN_STARTED:
            raise ReplayError("trace must start with run_started after optional resume")
        if start_index == 1 and steps[0].after_status != steps[1].after_status:
            raise ReplayError("resume restored status does not match run_started status")

        for index, step in enumerate(steps[start_index:], start=start_index):
            self._validate_step(index, step)

        if self.current_status is None:
            raise ReplayError("trace did not establish a status")
        self._validate_terminal_tail(steps)
        return ReplayResult(True, len(steps), final_status=self.current_status)

    def _validate_step_ids(self, steps: Sequence[TraceStep]) -> None:
        step_ids = [step.step_id for step in steps]
        if step_ids != sorted(step_ids):
            raise ReplayError("trace step ids must be increasing")
        if len(step_ids) != len(set(step_ids)):
            raise ReplayError("trace step ids must be unique")

    def _validate_resume(self, step: TraceStep) -> None:
        if step.before_status not in {
            AgentStatus.PAUSED,
            AgentStatus.PLANNING,
            AgentStatus.EXECUTING_TOOLS,
        }:
            raise ReplayError("resume step must start from a resumable status")
        if step.after_status not in {AgentStatus.PLANNING, AgentStatus.EXECUTING_TOOLS}:
            raise ReplayError("resume step must restore planning or executing_tools")
        if step.before_status is not AgentStatus.PAUSED and step.after_status != step.before_status:
            raise ReplayError("non-paused resume must preserve status")
        snapshot_status = AgentStatus(
            _expect_str(step.payload.get("snapshot_status"), "resume snapshot_status")
        )
        restored_status = AgentStatus(
            _expect_str(step.payload.get("restored_status"), "resume restored_status")
        )
        if snapshot_status != step.before_status:
            raise ReplayError("resume payload snapshot_status does not match before_status")
        if restored_status != step.after_status:
            raise ReplayError("resume payload restored_status does not match after_status")

    def _validate_step(self, index: int, step: TraceStep) -> None:
        if self.run_completed_seen:
            raise ReplayError("run_completed must be the final trace step")
        if step.kind == TraceStepKinds.RUN_STARTED:
            self._validate_run_started(step)
            return
        if self.current_status is None:
            raise ReplayError(f"{step.kind} appeared before run_started")
        if self.pending_conversation_insert_count and step.kind not in {
            TraceStepKinds.CONVERSATION_INSERT,
            TraceStepKinds.CHECKPOINT,
        }:
            raise ReplayError(f"conversation_insert requires checkpoint before {step.kind}")

        if step.kind == TraceStepKinds.STATE_CHANGED:
            self._validate_state_changed(step)
        elif step.kind == TraceStepKinds.CHECKPOINT:
            self._validate_checkpoint(index, step)
        elif step.kind == TraceStepKinds.MODEL_CALL:
            self._validate_model_call(step)
        elif step.kind == TraceStepKinds.MODEL_DELTA:
            self._validate_same_status(step)
            if not self.model_call_open:
                raise ReplayError("model_delta requires an open model_call")
            if not self.pending_stream_delta:
                self.stream_baseline_message_count = self.durable_message_count
            self.pending_stream_delta = True
        elif step.kind == TraceStepKinds.MODEL_ERROR:
            self._validate_model_error(step)
        elif step.kind == TraceStepKinds.MODEL_RESULT:
            self._validate_model_result(step)
        elif step.kind == TraceStepKinds.TOOL_CALL:
            self._validate_tool_call(step)
        elif step.kind == TraceStepKinds.TOOL_RESULT:
            self._validate_tool_result(step)
        elif step.kind == TraceStepKinds.CONVERSATION_INSERT:
            self._validate_conversation_insert(step)
        elif step.kind == TraceStepKinds.PAUSE_REQUESTED:
            self._validate_pause_requested(step)
        elif step.kind == TraceStepKinds.RUN_PAUSED:
            self._validate_run_paused(step)
        elif step.kind == TraceStepKinds.FINAL:
            self._validate_final(step)
        elif step.kind == TraceStepKinds.ERROR:
            self._validate_error(step)
        elif step.kind == TraceStepKinds.RUN_COMPLETED:
            self._validate_run_completed(step)
        else:
            raise ReplayError(f"unknown trace step kind: {step.kind}")

    def _validate_run_started(self, step: TraceStep) -> None:
        if self.current_status is not None:
            raise ReplayError("run_started appeared more than once")
        if step.after_status is None:
            raise ReplayError("run_started must include after_status")
        status = _payload_status(step.payload, "run_started status")
        if status != step.after_status:
            raise ReplayError("run_started payload status does not match after_status")
        self.current_status = step.after_status
        raw_count = step.payload.get("message_count")
        if not isinstance(raw_count, int) or isinstance(raw_count, bool):
            raise ReplayError("run_started must include message_count")
        self.durable_message_count = raw_count
        self.stream_baseline_message_count = raw_count
        self.total_tool_calls_baseline = _expect_nonnegative_int(
            step.payload.get("total_tool_calls"), "run_started total_tool_calls"
        )
        self.last_reported_total_tool_calls = self.total_tool_calls_baseline
        if self.current_status is AgentStatus.EXECUTING_TOOLS:
            pending_ids = _expect_sequence(
                step.payload.get("pending_tool_call_ids", ()),
                "run_started pending_tool_call_ids",
            )
            self.pending_tool_call_ids = tuple(
                _expect_str(call_id, "run_started pending tool call id") for call_id in pending_ids
            )
            self.expected_tool_result_count = len(self.pending_tool_call_ids)

    def _validate_state_changed(self, step: TraceStep) -> None:
        before = step.before_status
        if before is None:
            raise ReplayError("state_changed must include before_status")
        if before != self.current_status:
            raise ReplayError(f"state transition expected {self.current_status} but saw {before}")
        after = step.after_status
        if after is None:
            raise ReplayError("state_changed must include after_status")
        payload_before = AgentStatus(_expect_str(step.payload.get("from"), "state_changed from"))
        payload_after = AgentStatus(_expect_str(step.payload.get("to"), "state_changed to"))
        if payload_before != before:
            raise ReplayError("state_changed payload from does not match before_status")
        if payload_after != after:
            raise ReplayError("state_changed payload to does not match after_status")
        self._validate_total_tool_calls(step.payload, "state_changed")
        if after not in VALID_STATE_TRANSITIONS.get(before, set()):
            raise ReplayError(f"invalid state transition: {before} -> {after}")
        if before is AgentStatus.PLANNING and after in {
            AgentStatus.COMPLETED,
            AgentStatus.EXECUTING_TOOLS,
        }:
            if not self.model_result_ready:
                raise ReplayError("planning transition requires a preceding model_result")
            if after is AgentStatus.COMPLETED and self.model_result_tool_call_count != 0:
                raise ReplayError("completed transition cannot ignore model tool calls")
            if after is AgentStatus.EXECUTING_TOOLS:
                if self.model_result_tool_call_count <= 0:
                    raise ReplayError("executing_tools transition requires model tool calls")
                self.expected_tool_result_count = self.model_result_tool_call_count
                self.expected_pending_checkpoint_count = self.model_result_tool_call_count
                self.pending_tool_call_ids = ()
                self.tool_call_ids_since_checkpoint = []
                self.tool_result_ids_since_checkpoint = []
            self.model_result_ready = False
        if before is AgentStatus.EXECUTING_TOOLS and after is AgentStatus.PLANNING:
            if not self.tool_result_ready:
                raise ReplayError("executing_tools to planning requires a preceding tool_result")
            if self.open_tool_calls:
                raise ReplayError("executing_tools to planning cannot leave tool_call open")
            self._validate_pending_tools_completed()
            self._validate_committed_total_tool_calls(step.payload, "state_changed")
            self.tool_result_ready = False
            self.expected_tool_result_count = None
            self.expected_pending_checkpoint_count = None
            self.pending_tool_call_ids = ()
            self.tool_call_ids_since_checkpoint = []
            self.tool_result_ids_since_checkpoint = []
        if (
            before is AgentStatus.EXECUTING_TOOLS
            and after is AgentStatus.PLANNING
            and self.pending_tool_pause_requests
        ):
            raise ReplayError("tool pause result must be applied before returning to planning")
        if after is AgentStatus.PAUSED:
            self._validate_pause_payload(step.payload.get("pause"), "state_changed pause")
        elif step.payload.get("pause") is not None:
            raise ReplayError("non-paused state_changed must not carry pause payload")
        self.current_status = after

    def _validate_checkpoint(self, index: int, step: TraceStep) -> None:
        self._validate_same_status(step)
        status = _payload_status(step.payload, "checkpoint state status")
        if status != self.current_status:
            raise ReplayError("checkpoint status does not match replay status")
        self._validate_total_tool_calls(step.payload, "checkpoint")
        checkpoint_pending_tool_call_ids = self._checkpoint_pending_tool_call_ids(step.payload)
        if self.expected_pending_checkpoint_count is not None:
            if len(checkpoint_pending_tool_call_ids) != self.expected_pending_checkpoint_count:
                raise ReplayError(
                    "checkpoint pending_tool_call_ids must match model_result tool_call_count"
                )
            self.expected_pending_checkpoint_count = None
        pause = step.payload.get("pause")
        if status is AgentStatus.PAUSED and pause is None:
            raise ReplayError("paused checkpoint requires pause payload")
        if status is not AgentStatus.PAUSED and pause is not None:
            raise ReplayError("non-paused checkpoint must not carry pause payload")
        if status is AgentStatus.PAUSED:
            self._validate_pause_payload(pause, "checkpoint pause")
        if self.pending_stream_delta:
            raw_count = step.payload.get("message_count")
            if not isinstance(raw_count, int) or isinstance(raw_count, bool):
                raise ReplayError("checkpoint must include message_count")
            if raw_count > self.stream_baseline_message_count:
                raise ReplayError("stream delta was checkpointed without a complete model result")
        raw_count = step.payload.get("message_count")
        if self.pending_conversation_insert_count:
            if not isinstance(raw_count, int) or isinstance(raw_count, bool):
                raise ReplayError("conversation_insert checkpoint must include message_count")
            expected_count = (
                self.pending_conversation_insert_baseline + self.pending_conversation_insert_count
            )
            if raw_count != expected_count:
                raise ReplayError(
                    "conversation_insert checkpoint must include every inserted message"
                )
            roles = _expect_role_list(step.payload.get("message_roles"), "checkpoint message_roles")
            inserted_roles = roles[-self.pending_conversation_insert_count :]
            if inserted_roles != ["external"] * self.pending_conversation_insert_count:
                raise ReplayError(
                    "conversation_insert checkpoint must append external message roles"
                )
            self.pending_conversation_insert_count = 0
        if isinstance(raw_count, int) and not isinstance(raw_count, bool):
            self.durable_message_count = raw_count
        if status is AgentStatus.PLANNING:
            self.model_result_ready = False
            self.expected_tool_result_count = None
            self.expected_pending_checkpoint_count = None
            self.pending_tool_call_ids = ()
            self.tool_call_ids_since_checkpoint = []
            self.tool_result_ids_since_checkpoint = []
        if status is AgentStatus.EXECUTING_TOOLS:
            self.tool_result_ready = False
            self.pending_tool_call_ids = checkpoint_pending_tool_call_ids
            self.tool_call_ids_since_checkpoint = []
            self.tool_result_ids_since_checkpoint = []
        self.last_checkpoint_status = status
        self.last_checkpoint_payload = step.payload
        self.last_checkpoint_index = index

    def _checkpoint_pending_tool_call_ids(self, payload: Mapping[str, Any]) -> tuple[str, ...]:
        raw_pending_ids = _expect_sequence(
            payload.get("pending_tool_call_ids", ()),
            "checkpoint pending_tool_call_ids",
        )
        return tuple(
            _expect_str(call_id, "checkpoint pending tool call id") for call_id in raw_pending_ids
        )

    def _validate_same_status(self, step: TraceStep) -> None:
        if step.before_status is not None and step.before_status != self.current_status:
            raise ReplayError(f"{step.kind} before_status does not match replay status")
        if step.after_status is not None and step.after_status != self.current_status:
            raise ReplayError(f"{step.kind} after_status does not match replay status")

    def _validate_model_call(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.PLANNING:
            raise ReplayError("model_call is only valid while planning")
        if self.model_call_open:
            raise ReplayError("model_call already open")
        self.model_call_open = True

    def _validate_model_result(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.PLANNING:
            raise ReplayError("model_result is only valid while planning")
        if not self.model_call_open:
            raise ReplayError("model_result requires an open model_call")
        self.model_call_open = False
        self.model_result_ready = True
        self.model_result_tool_call_count = _expect_nonnegative_int(
            step.payload.get("tool_call_count"), "model_result tool_call_count"
        )
        self.pending_stream_delta = False

    def _validate_model_error(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.PLANNING:
            raise ReplayError("model_error is only valid while planning")
        if not self.model_call_open:
            raise ReplayError("model_error requires an open model_call")
        self.model_call_open = False
        self.model_result_ready = False
        self.pending_stream_delta = False

    def _validate_tool_call(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.EXECUTING_TOOLS:
            raise ReplayError("tool_call is only valid while executing_tools")
        if self.expected_pending_checkpoint_count is not None:
            raise ReplayError("tool_call requires a checkpoint after model_result tool calls")
        call_id = _payload_str(step.payload, "id")
        if self.pending_tool_call_ids and call_id not in self.pending_tool_call_ids:
            raise ReplayError(f"tool_call is not pending: {call_id}")
        if call_id in self.open_tool_calls:
            raise ReplayError(f"tool_call already open: {call_id}")
        if call_id in self.tool_call_ids_since_checkpoint:
            raise ReplayError(f"tool_call id must be unique in execution segment: {call_id}")
        self.open_tool_calls[call_id] = {
            "name": _payload_str(step.payload, "name"),
            "mode": _payload_str(step.payload, "mode"),
            "batch_id": _payload_str(step.payload, "batch_id"),
            "parallel": _expect_bool(step.payload.get("parallel"), "tool_call parallel"),
            "index": _expect_nonnegative_int(step.payload.get("index"), "tool_call index"),
        }
        self.tool_call_ids_since_checkpoint.append(call_id)

    def _validate_tool_result(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.EXECUTING_TOOLS:
            raise ReplayError("tool_result is only valid while executing_tools")
        call_id = _payload_str(step.payload, "id")
        expected = self.open_tool_calls.get(call_id)
        if expected is None:
            raise ReplayError(f"tool_result requires an open tool_call: {call_id}")
        result_index = _expect_nonnegative_int(step.payload.get("index"), "tool_result index")
        mode = _payload_str(step.payload, "mode")
        actual = {
            "name": _payload_str(step.payload, "name"),
            "mode": mode,
            "batch_id": _payload_str(step.payload, "batch_id"),
            "parallel": _expect_bool(step.payload.get("parallel"), "tool_result parallel"),
            "index": result_index,
        }
        if actual != expected:
            raise ReplayError("tool_result envelope does not match matching tool_call")
        result = _expect_mapping(step.payload["result"], "tool result payload")
        result_kind = _expect_str(result.get("result_kind"), "tool result kind")
        if not _tool_result_kind_matches_mode(mode, result_kind):
            raise ReplayError("tool_result result_kind must match tool invocation mode")
        del self.open_tool_calls[call_id]
        self.tool_result_count += 1
        self.tool_result_ready = True
        self.tool_result_ids_since_checkpoint.append(call_id)
        pause = result.get("pause")
        if pause is not None:
            self.pending_tool_pause_requests.append(
                (
                    result_index,
                    _expect_mapping(pause, "tool result pause"),
                )
            )

    def _validate_conversation_insert(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.PLANNING:
            raise ReplayError("conversation_insert is only valid while planning")
        _validate_conversation_insert_payload(step.payload, "conversation_insert payload")
        if self.pending_conversation_insert_count == 0:
            self.pending_conversation_insert_baseline = self.durable_message_count
        self.pending_conversation_insert_count += 1
        self.model_call_open = False
        self.model_result_ready = False
        self.pending_stream_delta = False

    def _validate_final(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status is not AgentStatus.COMPLETED:
            raise ReplayError("final is only valid after completed state")
        part_count = _expect_nonnegative_int(step.payload.get("part_count"), "final part_count")
        self.final_part_count = part_count
        if self.last_checkpoint_payload is not None:
            expected = _expect_nonnegative_int(
                self.last_checkpoint_payload.get("final_part_count"),
                "checkpoint final_part_count",
            )
            if part_count != expected:
                raise ReplayError("final part_count does not match checkpoint final_part_count")

    def _validate_error(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status not in {
            AgentStatus.PAUSED,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.LIMIT_EXCEEDED,
        }:
            raise ReplayError("error is only valid after an invocation-terminal state")
        status = AgentStatus(_expect_str(step.payload.get("status"), "error status"))
        if status != self.current_status:
            raise ReplayError("error payload status does not match replay status")

    def _validate_total_tool_calls(self, payload: Mapping[str, Any], label: str) -> None:
        total_tool_calls = _expect_nonnegative_int(
            payload.get("total_tool_calls"), f"{label} total_tool_calls"
        )
        upper_bound = self.total_tool_calls_baseline + self.tool_result_count
        if total_tool_calls < self.last_reported_total_tool_calls:
            raise ReplayError(f"{label} total_tool_calls must not decrease")
        if total_tool_calls > upper_bound:
            raise ReplayError(f"{label} total_tool_calls exceeds replay tool_result count")
        self.last_reported_total_tool_calls = total_tool_calls

    def _validate_committed_total_tool_calls(self, payload: Mapping[str, Any], label: str) -> None:
        total_tool_calls = _expect_nonnegative_int(
            payload.get("total_tool_calls"), f"{label} total_tool_calls"
        )
        expected = self.total_tool_calls_baseline + self.tool_result_count
        if total_tool_calls != expected:
            raise ReplayError(f"{label} total_tool_calls must include committed tool_results")

    def _validate_pending_tools_completed(self) -> None:
        result_ids = list(self.tool_result_ids_since_checkpoint)
        if self.pending_tool_call_ids:
            expected_ids = list(self.pending_tool_call_ids)
            if Counter(result_ids) != Counter(expected_ids):
                raise ReplayError(
                    "executing_tools to planning requires tool_results for all pending tool calls"
                )
            return
        if self.expected_tool_result_count is not None and (
            len(result_ids) != self.expected_tool_result_count
        ):
            raise ReplayError(
                "executing_tools to planning requires tool_results for every model tool call"
            )

    def _validate_pause_requested(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.pending_pause_request is not None:
            raise ReplayError("pause_requested appeared before previous pause was applied")
        origin = _payload_str(step.payload, "origin")
        if origin == "tool_result":
            if not self.pending_tool_pause_requests:
                raise ReplayError("tool pause requires a matching pause-bearing tool_result")
            first_index, first_request = min(
                self.pending_tool_pause_requests,
                key=lambda item: item[0],
            )
            if not _pause_request_matches(first_request, step.payload):
                raise ReplayError("tool pause must match the first pause-bearing tool_result")
            self.pending_tool_pause_requests = [
                item for item in self.pending_tool_pause_requests if item[0] != first_index
            ]
        elif origin == "control":
            if self.pending_tool_pause_requests:
                raise ReplayError("tool pause result must be applied before control pause")
        else:
            raise ReplayError(f"unsupported pause origin: {origin}")
        self.pending_pause_request = step.payload

    def _validate_pause_payload(self, value: object, label: str) -> None:
        if self.pending_pause_request is None:
            raise ReplayError(f"{label} requires pause_requested")
        pause = _expect_mapping(value, label)
        if not _pause_state_matches_request(self.pending_pause_request, pause):
            raise ReplayError(f"{label} does not match pause request")

    def _validate_run_paused(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        self._validate_pause_payload(step.payload.get("pause"), "run_paused pause")

    def _validate_run_completed(self, step: TraceStep) -> None:
        self._validate_same_status(step)
        if self.current_status not in {
            AgentStatus.PAUSED,
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.LIMIT_EXCEEDED,
        }:
            raise ReplayError("run_completed requires an invocation-terminal state")
        raw_state = step.payload.get("state")
        if raw_state is None:
            return
        state = _expect_mapping(raw_state, "run_completed state")
        status = _payload_status(state, "run_completed state status")
        if status != self.current_status:
            raise ReplayError("run_completed state status does not match replay status")
        self._validate_total_tool_calls(state, "run_completed state")
        if self.last_checkpoint_payload is not None:
            _validate_terminal_summary_matches_checkpoint(state, self.last_checkpoint_payload)
        self.run_completed_seen = True

    def _validate_terminal_tail(self, steps: Sequence[TraceStep]) -> None:
        if steps[-1].kind != TraceStepKinds.RUN_COMPLETED:
            raise ReplayError("trace must end with run_completed")
        final_status = self.current_status
        if final_status is AgentStatus.PAUSED:
            if self.open_tool_calls:
                raise ReplayError("paused trace cannot leave tool_call open")
            self._require_one_tail(
                steps,
                (
                    [
                        TraceStepKinds.PAUSE_REQUESTED,
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.RUN_PAUSED,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                    [
                        TraceStepKinds.PAUSE_REQUESTED,
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                    [
                        TraceStepKinds.PAUSE_REQUESTED,
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.RUN_PAUSED,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                ),
            )
        elif final_status is AgentStatus.COMPLETED:
            if self.model_call_open:
                raise ReplayError("completed trace cannot leave model_call open")
            if self.open_tool_calls:
                raise ReplayError("completed trace cannot leave tool_call open")
            self._require_one_tail(
                steps,
                (
                    [
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.FINAL,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                    [
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                    [
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.FINAL,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                ),
            )
        elif final_status in {AgentStatus.FAILED, AgentStatus.LIMIT_EXCEEDED}:
            self._require_one_tail(
                steps,
                (
                    [
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                    [
                        TraceStepKinds.STATE_CHANGED,
                        TraceStepKinds.CHECKPOINT,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.ERROR,
                        TraceStepKinds.RUN_COMPLETED,
                    ],
                ),
            )
        else:
            raise ReplayError("trace ended without an invocation-terminal status")
        if self.last_checkpoint_status != final_status:
            raise ReplayError("last checkpoint does not match final status")
        if self.last_checkpoint_index < 0:
            raise ReplayError("trace must include a checkpoint")

    def _require_tail(self, steps: Sequence[TraceStep], expected: Sequence[str]) -> None:
        actual = [step.kind for step in steps[-len(expected) :]]
        if actual != list(expected):
            raise ReplayError(f"terminal trace tail mismatch: expected {expected}, got {actual}")

    def _require_one_tail(
        self, steps: Sequence[TraceStep], expected_options: Sequence[Sequence[str]]
    ) -> None:
        for expected in expected_options:
            actual = [step.kind for step in steps[-len(expected) :]]
            if actual == list(expected):
                return
        expected_text = " or ".join(str(list(expected)) for expected in expected_options)
        longest = max(len(expected) for expected in expected_options)
        actual = [step.kind for step in steps[-longest:]]
        raise ReplayError(f"terminal trace tail mismatch: expected {expected_text}, got {actual}")


def _step_from_event(step_id: int, event: AgentEvent) -> TraceStep | None:
    kind = EVENT_TRACE_KIND.get(event.type)
    if kind is None:
        return None
    before_status, after_status = _statuses_from_event(kind, event.data)
    payload = _payload_from_event(kind, event.data)
    return TraceStep(
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
        return _compact_state_snapshot(state)
    if kind == TraceStepKinds.CHECKPOINT:
        state = _expect_mapping(data["state"], "checkpoint state")
        payload = _compact_state_snapshot(state)
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
        }
    if kind == TraceStepKinds.TOOL_RESULT:
        return {
            "id": data["id"],
            "name": data["name"],
            "mode": data["mode"],
            "batch_id": data["batch_id"],
            "parallel": data["parallel"],
            "index": data["index"],
            "result": _compact_tool_result_summary(
                _expect_mapping(data["result"], "tool result summary")
            ),
        }
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
        return {"state": _compact_state_snapshot(_expect_mapping(data["state"], "completed state"))}
    if kind == TraceStepKinds.RUN_PAUSED:
        return {"pause": _compact_pause(data["pause"])}
    if kind == TraceStepKinds.FINAL:
        return _compact_final(data)
    if kind == TraceStepKinds.ERROR:
        return deepcopy(dict(data))
    return {}


def _compact_tool_result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _expect_mapping(result.get("metadata", {}), "tool result metadata")
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
    return payload


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
    error = _expect_mapping(data["error"], "model error")
    metadata = _expect_mapping(error.get("metadata", {}), "model error metadata")
    payload: dict[str, Any] = {
        "message": error["message"],
        "retry": data["retry"],
        "retryable": error.get("retryable", False),
        "metadata_keys": sorted(str(key) for key in metadata),
    }
    for key in ("provider", "code", "status_code", "request_id"):
        if key in error:
            payload[key] = error[key]
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
        for part in _expect_sequence(data.get("parts", []), "final parts")
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
        raw_roles = state.get("message_roles", [])
        message_roles = [
            _expect_str(role, "state message role")
            for role in _expect_sequence(raw_roles, "state message_roles")
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
        raw_pending_ids = state.get("pending_tool_call_ids", [])
        pending_tool_call_ids = [
            _expect_str(call_id, "pending tool call id")
            for call_id in _expect_sequence(raw_pending_ids, "pending tool_call_ids")
        ]

    if "final_parts" in state:
        final_part_count = len(_expect_sequence(state["final_parts"], "state final_parts"))
    elif "final_part_count" in state:
        final_part_count = _expect_nonnegative_int(
            state["final_part_count"], "state final_part_count"
        )
    else:
        final_part_count = 1 if state.get("has_final") is True else 0

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


def _payload_status(payload: Mapping[str, Any], label: str) -> AgentStatus:
    return AgentStatus(_expect_str(payload["status"], label))


def _payload_str(payload: Mapping[str, Any], key: str) -> str:
    return _expect_str(payload[key], f"trace payload {key}")


def _validate_terminal_summary_matches_checkpoint(
    state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    for key in (
        "status",
        "message_count",
        "message_roles",
        "pending_tool_call_ids",
        "iterations",
        "total_tool_calls",
        "final_part_count",
        "error",
        "pause",
    ):
        if state.get(key) != checkpoint.get(key):
            raise ReplayError(f"run_completed state {key} does not match last checkpoint")


def _pause_request_matches(request: Mapping[str, Any], pause_step: Mapping[str, Any]) -> bool:
    request_keys = list(_expect_sequence(request["metadata_keys"], "tool pause metadata_keys"))
    step_keys = list(_expect_sequence(pause_step["metadata_keys"], "pause metadata_keys"))
    return (
        request["reason"] == pause_step["reason"]
        and request["source"] == pause_step["source"]
        and request["wait_id"] == pause_step["wait_id"]
        and request["interrupt"] == pause_step["interrupt"]
        and request_keys == step_keys
    )


def _pause_state_matches_request(request: Mapping[str, Any], pause: Mapping[str, Any]) -> bool:
    request_keys = list(_expect_sequence(request["metadata_keys"], "pause request metadata_keys"))
    pause_keys = list(_expect_sequence(pause["metadata_keys"], "pause metadata_keys"))
    return (
        request["reason"] == pause["reason"]
        and request["source"] == pause["source"]
        and request["wait_id"] == pause["wait_id"]
        and request["resume_status"] == pause["resume_status"]
        and request_keys == pause_keys
    )


def _expect_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
    return value


def _status_or_none(value: object, label: str) -> AgentStatus | None:
    raw = _expect_optional_str(value, label)
    return None if raw is None else AgentStatus(raw)
