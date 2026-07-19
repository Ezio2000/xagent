from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from math import inf
from typing import Any, cast

import pytest

from jharness.kernel import SuspensionSelector as KernelSuspensionSelector
from jharness.kernel._digest import empty_call_id_suffix_digest
from jharness.kernel._engine.verification import run_view
from jharness.kernel.checkpoint import (
    Checkpoint,
    ControlFact,
    ConversationInsertFact,
    Fact,
    FailedControl,
    HistoryRewriteFact,
    LimitedControl,
    ModelTurnFact,
    ModelTurnResult,
    ResumedFact,
    StartedFact,
    SuspendedControl,
    SuspensionView,
    ToolBatchFact,
    ToolOutcomeKind,
)
from jharness.kernel.context import RunContext
from jharness.kernel.diagnostics import RunTrace, TraceEntry, TraceHeader
from jharness.kernel.errors import ProtocolError, RequestError
from jharness.kernel.events import Event, EventKind
from jharness.kernel.history import RunHistory
from jharness.kernel.limits import LimitReason
from jharness.kernel.messages import ArtifactRef, ContentPart, ErrorInfo, Message, TaskRef, ToolCall
from jharness.kernel.models import ModelResponse, ModelUsage
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import (
    Completed,
    Failed,
    Limited,
    PendingToolCalls,
    Planning,
    RunMetrics,
    RunState,
    Suspended,
    Suspension,
    ToolsPending,
)
from jharness.kernel.tools import (
    SettledResult,
    ToolAccepted,
    ToolExecution,
    ToolFailure,
    ToolOutcome,
    ToolRisk,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.kernel.wire import (
    ContinueRequest,
    ResumeRequest,
    StartRequest,
    SuspensionSelector,
    decode_checkpoint,
    decode_content_part,
    decode_context,
    decode_error_info,
    decode_event,
    decode_fact,
    decode_message,
    decode_metrics,
    decode_model_response,
    decode_model_usage,
    decode_run_request,
    decode_run_view,
    decode_snapshot,
    decode_state,
    decode_suspension,
    decode_tool_call,
    decode_tool_outcome,
    decode_tool_result,
    decode_tool_spec,
    decode_trace,
    encode_checkpoint,
    encode_content_part,
    encode_context,
    encode_event,
    encode_fact,
    encode_message,
    encode_model_response,
    encode_run_request,
    encode_snapshot,
    encode_state,
    encode_tool_outcome,
    encode_tool_result,
    encode_tool_spec,
    encode_trace,
)
from jharness.kernel.wire._helpers import number, optional_number


def _context() -> RunContext:
    return RunContext("run-1", 1.0, deadline=9.0, metadata={"tenant": "a"})


def _history(*messages: Message) -> RunHistory:
    return RunHistory(messages)


def _pending(*calls: ToolCall) -> PendingToolCalls:
    return PendingToolCalls(calls)


def _started() -> Checkpoint:
    history = _history(Message.user("hello", metadata={"lang": "en"}))
    snapshot = RunSnapshot(0, _context(), history, RunMetrics(), Planning())
    return Checkpoint("cp-0", snapshot, StartedFact(1.0, ("user",)))


def _suspended(*, pending: bool = False) -> Checkpoint:
    suspension = Suspension("approval_required", "approval", "wait-1", {"ticket": 7})
    history: RunHistory
    resume_to: Planning | ToolsPending
    if pending:
        call = ToolCall("call-1", "lookup", {"q": "x"})
        history = _history(Message.user("go"), Message.assistant(tool_calls=(call,)))
        resume_to = ToolsPending(_pending(call))
    else:
        history = _history(Message.user("hello"))
        resume_to = Planning()
    snapshot = RunSnapshot(
        1,
        _context(),
        history,
        RunMetrics(),
        Suspended(resume_to, suspension),
    )
    fact = ControlFact(
        2.0,
        SuspendedControl(
            suspension.reason,
            suspension.source,
            suspension.wait_id,
            ("ticket",),
        ),
    )
    return Checkpoint("cp-suspended", snapshot, fact)


def _assert_json(value: object) -> None:
    json.dumps(value, allow_nan=False)


def test_message_content_and_outcome_round_trips() -> None:
    artifact = ArtifactRef(
        "artifact:1",
        media_type="text/plain",
        size_bytes=2,
        metadata={"region": "cn"},
    )
    messages = (
        Message.system("rules"),
        Message.user("hello"),
        Message.assistant(
            (
                ContentPart.artifact_part(artifact),
                ContentPart(
                    "image",
                    uri="https://example.invalid/a.png",
                    media_type="image/png",
                    data={"width": 1},
                ),
            )
        ),
    )
    for message in messages:
        encoded = encode_message(message)
        _assert_json(encoded)
        assert decode_message(encoded) == message

    part = ContentPart.text_part("ok")
    task = TaskRef("job-1", "queued", {"queue": "a"})
    outcomes: tuple[ToolOutcome, ...] = (
        ToolSuccess((part,), {"count": 1}),
        ToolFailure((part,), ErrorInfo("bad", "failed"), {"retry": False}),
        ToolAccepted((part,), "job-1", task, {"queued": True}),
        ToolWaiting((part,), task, {"ready": False}),
    )
    for outcome in outcomes:
        encoded = encode_tool_outcome(outcome)
        _assert_json(encoded)
        assert decode_tool_outcome(encoded) == outcome


def test_model_tool_spec_and_result_round_trips() -> None:
    response = ModelResponse(
        (ContentPart.text_part("done"),),
        finish_reason="stop",
        usage=ModelUsage(2, 3, 5),
        model_id="model-1",
        response_id="response-1",
        metadata={"provider": "test"},
    )
    assert decode_model_response(encode_model_response(response)) == response

    spec = ToolSpec(
        "lookup",
        "Lookup a value",
        {"type": "object"},
        output_schema=True,
        execution=ToolExecution("parallel", read_only=True, idempotent=True),
        risk=ToolRisk(network="read", requires_approval=True, extra={"zone": "public"}),
    )
    encoded_spec = encode_tool_spec(spec)
    _assert_json(encoded_spec)
    assert decode_tool_spec(encoded_spec) == spec

    success = SettledResult(ToolSuccess((ContentPart.text_part("ok"),), {"ok": True}))
    waiting = WaitingResult(
        ToolWaiting((ContentPart.text_part("waiting"),), structured_content={"job": 1}),
        Suspension("external_wait", "tool", "job-1", {"retry_after": 1}),
    )
    for result in (success, waiting):
        encoded_result = encode_tool_result(result)
        _assert_json(encoded_result)
        assert decode_tool_result(encoded_result) == result


def test_all_flat_states_round_trip() -> None:
    call = ToolCall("call-1", "lookup")
    suspension = Suspension("pause", "host")
    states: tuple[RunState, ...] = (
        Planning(),
        ToolsPending(_pending(call)),
        Suspended(ToolsPending(_pending(call)), suspension),
        Completed((ContentPart.text_part("done"),)),
        Failed(ErrorInfo("model", "failed")),
        Limited(LimitReason.DEADLINE),
    )
    for state in states:
        encoded = encode_state(state)
        assert decode_state(encoded) == state


def test_snapshot_checkpoint_and_fact_round_trips() -> None:
    checkpoint = _started()
    encoded_snapshot = encode_snapshot(checkpoint.snapshot)
    assert "schema_version" not in encoded_snapshot
    assert decode_snapshot(encoded_snapshot) == checkpoint.snapshot
    encoded_checkpoint = encode_checkpoint(checkpoint)
    assert encoded_checkpoint["schema_version"] == "v0"
    assert decode_checkpoint(encoded_checkpoint) == checkpoint

    facts: tuple[Fact, ...] = (
        checkpoint.fact,
        ResumedFact(2.0, ("external",), ("source",)),
        ModelTurnFact(
            2.0,
            ModelTurnResult.COMPLETED,
            1,
            (),
            "stop",
            ModelUsage(total_tokens=2),
            None,
        ),
        ToolBatchFact(
            2.0,
            "batch-1",
            ("call-1",),
            False,
            (ToolOutcomeKind.SUCCESS,),
            None,
        ),
        ToolBatchFact(
            2.0,
            "batch-2",
            ("call-2",),
            False,
            (ToolOutcomeKind.WAITING,),
            SuspensionView("wait", "tool", "job-1", ()),
        ),
        ConversationInsertFact(2.0, "host"),
        HistoryRewriteFact(2.0, 2, ("user",), "compact", ()),
        ControlFact(2.0, FailedControl("model")),
        ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
    )
    for fact in facts:
        assert decode_fact(encode_fact(fact)) == fact


def test_run_request_round_trips_and_enforces_cross_fields() -> None:
    assert SuspensionSelector is KernelSuspensionSelector
    start = StartRequest(_history(Message.user("hello")), _context())
    continued = ContinueRequest(_started())
    suspended = _suspended()
    resumed = ResumeRequest(
        suspended,
        SuspensionSelector(source="approval", metadata={"ticket": 7}),
        (Message.external("approved"),),
        {"actor": "host"},
    )
    for request in (start, continued, resumed):
        encoded = encode_run_request(request)
        _assert_json(encoded)
        assert decode_run_request(encoded) == request

    invalid_append = encode_run_request(ResumeRequest(_suspended(pending=True)))
    invalid_append["append_messages"] = [encode_message(Message.user("not allowed"))]
    with pytest.raises(RequestError, match="planning continuation") as planning_error:
        decode_run_request(invalid_append)
    assert planning_error.value.code == "messages_require_planning"

    selector_mismatch = encode_run_request(ResumeRequest(suspended))
    selector_mismatch["selector"] = {"source": "other"}
    with pytest.raises(RequestError, match="does not match") as selector_error:
        decode_run_request(selector_mismatch)
    assert selector_error.value.code == "suspension_mismatch"

    call = ToolCall("call-1", "lookup")
    with pytest.raises(ValueError, match="unresolved"):
        StartRequest(_history(Message.user("go"), Message.assistant(tool_calls=(call,))))


def test_checkpoint_rejects_fact_snapshot_message_mismatches() -> None:
    suspension = Suspension("host_pause", "host", metadata={"ticket": 1})
    suspended_snapshot = RunSnapshot(
        1,
        _context(),
        _history(Message.user("hello")),
        RunMetrics(),
        Suspended(Planning(), suspension),
    )
    with pytest.raises(ValueError, match="must match Suspended state"):
        Checkpoint(
            "bad-control",
            suspended_snapshot,
            ControlFact(2.0, SuspendedControl("other", "host", None, ("ticket",))),
        )

    call = ToolCall("call-1", "lookup")
    outcome = ToolSuccess((ContentPart.text_part("ok"),))
    tool_history = _history(
        Message.user("go"),
        Message.assistant(tool_calls=(call,)),
        Message.tool(call.id, outcome),
    )
    tool_snapshot = RunSnapshot(
        2,
        _context(),
        tool_history,
        RunMetrics(1, 1),
        Suspended(Planning(), suspension),
    )
    valid_tool_fact = ToolBatchFact(
        2.0,
        "batch-1",
        (call.id,),
        False,
        (ToolOutcomeKind.SUCCESS,),
        SuspensionView("host_pause", "host", None, ("ticket",)),
    )
    assert Checkpoint("valid-tool", tool_snapshot, valid_tool_fact).fact is valid_tool_fact

    with pytest.raises(ValueError, match="suspension must match"):
        Checkpoint(
            "bad-tool-suspension",
            tool_snapshot,
            ToolBatchFact(
                2.0,
                "batch-1",
                (call.id,),
                False,
                (ToolOutcomeKind.SUCCESS,),
                SuspensionView("host_pause", "other", None, ("ticket",)),
            ),
        )
    with pytest.raises(ValueError, match="call ids"):
        Checkpoint(
            "bad-tool-call",
            tool_snapshot,
            ToolBatchFact(
                2.0,
                "batch-1",
                ("other",),
                False,
                (ToolOutcomeKind.SUCCESS,),
                SuspensionView("host_pause", "host", None, ("ticket",)),
            ),
        )
    with pytest.raises(ValueError, match="outcomes"):
        Checkpoint(
            "bad-tool-outcome",
            tool_snapshot,
            ToolBatchFact(
                2.0,
                "batch-1",
                (call.id,),
                False,
                (ToolOutcomeKind.FAILURE,),
                SuspensionView("host_pause", "host", None, ("ticket",)),
            ),
        )

    planning_snapshot = RunSnapshot(
        1,
        _context(),
        _history(Message.user("hello")),
        RunMetrics(),
        Planning(),
    )
    with pytest.raises(ValueError, match="append an external"):
        Checkpoint(
            "bad-insert",
            planning_snapshot,
            ConversationInsertFact(2.0, "host"),
        )
    with pytest.raises(ValueError, match="roles must match"):
        Checkpoint(
            "bad-rewrite",
            planning_snapshot,
            HistoryRewriteFact(2.0, 2, ("system",), "compact", ()),
        )

    assistant = Message.assistant(
        (ContentPart.text_part("partial"),),
        tool_calls=(call,),
    )
    limited_snapshot = RunSnapshot(
        1,
        _context(),
        _history(Message.user("go"), assistant),
        RunMetrics(1, 0, ModelUsage(total_tokens=10)),
        Limited(LimitReason.MAX_TOTAL_TOKENS),
    )
    valid_model_fact = ModelTurnFact(
        2.0,
        ModelTurnResult.LIMITED,
        1,
        (call.id,),
        "length",
        ModelUsage(total_tokens=10),
        LimitReason.MAX_TOTAL_TOKENS,
    )
    assert Checkpoint("valid-model", limited_snapshot, valid_model_fact).fact is valid_model_fact
    with pytest.raises(ValueError, match="part_count"):
        Checkpoint(
            "bad-model",
            limited_snapshot,
            ModelTurnFact(
                2.0,
                ModelTurnResult.LIMITED,
                0,
                (call.id,),
                "length",
                ModelUsage(total_tokens=10),
                LimitReason.MAX_TOTAL_TOKENS,
            ),
        )


def _event_wire(kind: str, data: Mapping[str, Any], *, sequence: int = 1) -> dict[str, Any]:
    return {
        "schema_version": "v0",
        "run_id": "run-1",
        "invocation_id": "inv-1",
        "sequence": sequence,
        "kind": kind,
        "created_at": float(sequence),
        "data": dict(data),
    }


def test_every_event_data_shape_round_trips() -> None:
    checkpoint = _started()
    call = ToolCall("call-1", "lookup", {"q": "x"})
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "total_tokens": 3,
        "reasoning_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    fixtures: tuple[tuple[str, Mapping[str, Any]], ...] = (
        (
            "invocation_started",
            {"request_kind": "start", "starting_checkpoint_id": None, "starting": None},
        ),
        (
            "invocation_started",
            {
                "request_kind": "continue",
                "starting_checkpoint_id": checkpoint.id,
                "starting": run_view(checkpoint.snapshot),
            },
        ),
        ("model_started", {"planning_step": 1}),
        (
            "model_delta",
            {"kind": "content", "index": 0, "part_type": "text", "text_delta": "a", "data": {}},
        ),
        (
            "model_delta",
            {
                "kind": "tool_call",
                "index": 0,
                "id": "call-1",
                "name": "lookup",
                "arguments_delta": "{}",
            },
        ),
        ("model_delta", {"kind": "reasoning", "index": 0, "text_delta": "why"}),
        ("model_delta", {"kind": "usage", "usage": usage}),
        ("model_finished", {"finish_reason": "stop", "tool_call_count": 0, "usage": usage}),
        (
            "tool_batch_selected",
            {
                "batch_id": "batch-1",
                "call_ids": [call.id],
                "parallel": False,
                "remaining_count": 0,
                "remaining_call_id_digest": empty_call_id_suffix_digest().hex(),
            },
        ),
        (
            "approval_requested",
            {
                "batch_id": "batch-1",
                "index": 0,
                "call": {"id": call.id, "name": call.name, "arguments": {"q": "x"}},
                "risk": {"network": "read"},
            },
        ),
        ("approval_decided", {"call_id": call.id, "kind": "allow"}),
        (
            "approval_decided",
            {"call_id": call.id, "kind": "deny", "reason": "policy"},
        ),
        (
            "approval_decided",
            {
                "call_id": call.id,
                "kind": "suspend",
                "suspension": {
                    "reason": "approval",
                    "source": "host",
                    "wait_id": "wait-1",
                    "metadata": {},
                },
            },
        ),
        (
            "tool_started",
            {
                "batch_id": "batch-1",
                "index": 0,
                "call": {"id": call.id, "name": call.name, "arguments": {"q": "x"}},
                "parallel": False,
            },
        ),
        ("tool_progress", {"tool_call_id": call.id, "progress": {"percent": 50}}),
        (
            "tool_finished",
            {"batch_id": "batch-1", "index": 0, "tool_call_id": call.id, "outcome_kind": "success"},
        ),
        ("tool_cancel_requested", {"tool_call_id": call.id}),
        (
            "checkpoint_committed",
            {
                "checkpoint_id": checkpoint.id,
                "fact": encode_fact(checkpoint.fact),
                "after": run_view(checkpoint.snapshot),
            },
        ),
        ("invocation_stopped", {"reason": "terminal", "last_checkpoint_id": checkpoint.id}),
    )
    for sequence, (kind, data) in enumerate(fixtures, start=1):
        raw = _event_wire(kind, data, sequence=sequence)
        decoded = decode_event(raw)
        assert encode_event(decoded) == raw


def test_trace_round_trip_uses_compact_entries() -> None:
    events = (
        Event(
            "run-1",
            "inv-1",
            1,
            EventKind.INVOCATION_STARTED,
            1.0,
            {"request_kind": "start", "starting_checkpoint_id": None, "starting": None},
        ),
        Event(
            "run-1",
            "inv-1",
            2,
            EventKind.INVOCATION_STOPPED,
            2.0,
            {"reason": "cancelled", "last_checkpoint_id": None},
        ),
    )
    trace = RunTrace(
        TraceHeader("run-1", "inv-1", "start", ("tenant",)),
        tuple(TraceEntry(x.sequence, x.kind, x.created_at, x.data) for x in events),
    )
    encoded = encode_trace(trace)
    _assert_json(encoded)
    assert "run_id" not in encoded["entries"][0]
    assert decode_trace(encoded) == trace


_INVALID_PORTABLE_VALUES: tuple[tuple[Callable[[object], object], dict[str, Any]], ...] = (
    (decode_message, {"role": "user", "parts": [], "metadata": {}, "extra": 1}),
    (
        decode_model_response,
        {
            "parts": [{"type": "text", "text": "x", "metadata": {}}],
            "tool_calls": [],
            "finish_reason": None,
            "usage": {
                "input_tokens": True,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
            },
            "model_id": None,
            "response_id": None,
            "metadata": {},
        },
    ),
    (decode_state, {"kind": "tools_pending", "pending": []}),
    (
        decode_run_view,
        {
            "revision": 0,
            "history_count": 1,
            "metrics": {
                "planning_steps": 0,
                "tool_calls": 0,
                "usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "reasoning_tokens": None,
                    "cache_read_tokens": None,
                    "cache_write_tokens": None,
                },
            },
            "state": {
                "kind": "tools_pending",
                "pending_count": 0,
                "call_id_digest": "00" * 32,
            },
        },
    ),
    (
        decode_tool_spec,
        {
            "name": "x",
            "description": "",
            "input_schema": {},
            "output_schema": None,
            "execution": {"concurrency": "parallel", "read_only": False, "idempotent": True},
            "risk": {},
        },
    ),
    (
        decode_tool_result,
        {
            "outcome": {
                "kind": "waiting",
                "parts": [{"type": "text", "text": "wait", "metadata": {}}],
                "task": None,
                "structured_content": None,
            }
        },
    ),
)


@pytest.mark.parametrize(
    "decoder,value",
    _INVALID_PORTABLE_VALUES,
)
def test_invalid_portable_values_are_rejected(
    decoder: Callable[[object], object],
    value: object,
) -> None:
    with pytest.raises(ProtocolError):
        decoder(value)


def test_wire_boundary_rejects_versions_non_finite_values_and_cycles() -> None:
    snapshot = encode_snapshot(_started().snapshot)
    snapshot["schema_version"] = "v0"
    with pytest.raises(ProtocolError, match="unknown field"):
        decode_snapshot(snapshot)

    checkpoint = encode_checkpoint(_started())
    checkpoint["schema_version"] = "v1"
    with pytest.raises(ProtocolError, match="must be v0"):
        decode_checkpoint(checkpoint)

    message = encode_message(Message.user("hello"))
    message["metadata"] = {"score": inf}
    with pytest.raises(ProtocolError, match="non-finite"):
        decode_message(message)

    cyclic: dict[str, object] = {
        "role": "user",
        "parts": [{"type": "text", "text": "hello", "metadata": {}}],
    }
    cyclic["metadata"] = cyclic
    with pytest.raises(ProtocolError, match="cycle"):
        decode_message(cyclic)


def test_wire_boundary_rejects_excessive_depth_and_unsafe_number_conversion() -> None:
    nested: object = None
    for _ in range(129):
        nested = [nested]
    message = encode_message(Message.user("hello"))
    message["metadata"] = {"nested": nested}
    with pytest.raises(ProtocolError, match="maximum JSON nesting depth"):
        decode_message(message)

    context = encode_context(_context())
    context["started_at"] = 10**400
    with pytest.raises(ProtocolError, match="safe JSON number range"):
        decode_context(context)


def test_opaque_data_presence_is_unambiguous() -> None:
    opaque: dict[str, Any] = {
        "type": "provider_extension",
        "data": {},
        "metadata": {},
    }
    with pytest.raises(ProtocolError, match="must not be empty when present"):
        decode_content_part(opaque)

    decoded = decode_content_part({"type": "provider_extension", "metadata": {}})
    assert encode_content_part(decoded) == {"type": "provider_extension", "metadata": {}}


def test_checkpoint_event_and_trace_cross_field_mismatches_are_rejected() -> None:
    checkpoint = encode_checkpoint(_started())
    checkpoint["fact"] = {
        "kind": "control",
        "at": 1.0,
        "data": {"action": "limited", "reason": "deadline"},
    }
    with pytest.raises(ProtocolError, match="revision 0"):
        decode_checkpoint(checkpoint)

    started_event = _event_wire(
        "invocation_started",
        {"request_kind": "start", "starting_checkpoint_id": "cp", "starting": None},
    )
    with pytest.raises(ProtocolError, match="cannot carry"):
        decode_event(started_event)

    trace: dict[str, Any] = {
        "schema_version": "v0",
        "header": {
            "run_id": "run-1",
            "invocation_id": "inv-1",
            "request_kind": "start",
            "metadata_keys": [],
        },
        "entries": [
            {
                "sequence": 1,
                "kind": "invocation_started",
                "created_at": 1.0,
                "data": {"request_kind": "start", "starting_checkpoint_id": None, "starting": None},
            },
            {
                "sequence": 2,
                "kind": "invocation_stopped",
                "created_at": 2.0,
                "data": {"reason": "unknown", "last_checkpoint_id": None},
            },
        ],
    }
    with pytest.raises(ProtocolError, match="unsupported"):
        decode_trace(trace)


def test_wire_scalar_and_container_guards_report_the_boundary_field() -> None:
    def invalid_case(
        decoder: Callable[[object], object],
        value: object,
        message: str,
    ) -> tuple[Callable[[object], object], object, str]:
        return decoder, value, message

    usage: dict[str, object] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
    }
    valid_spec: dict[str, object] = {
        "name": "lookup",
        "description": "lookup",
        "input_schema": {},
        "output_schema": None,
        "execution": {"concurrency": "serial", "read_only": False, "idempotent": False},
        "risk": {},
    }
    valid_context: dict[str, object] = {
        "run_id": "run-1",
        "started_at": 1.0,
        "deadline": None,
        "parent_run_id": None,
        "parent_tool_call_id": None,
        "run_kind": None,
        "metadata": {},
    }
    invalid_cases: tuple[tuple[Callable[[object], object], object, str], ...] = (
        invalid_case(decode_checkpoint, [], "checkpoint must be an object"),
        invalid_case(decode_message, [], "must be an object"),
        invalid_case(decode_message, {"role": "user"}, "missing field"),
        invalid_case(decode_message, {"role": 1}, "must be a string"),
        invalid_case(
            decode_message,
            {"role": "user", "parts": "not-an-array", "metadata": {}},
            "must be an array",
        ),
        invalid_case(decode_tool_spec, {**valid_spec, "name": ""}, "must not be empty"),
        invalid_case(decode_model_usage, {**usage, "input_tokens": -1}, "must be >= 0"),
        invalid_case(
            decode_context,
            {**valid_context, "started_at": True},
            "must be a number",
        ),
        invalid_case(decode_context, {**valid_context, "started_at": -1}, "must be >= 0"),
        invalid_case(
            decode_tool_spec,
            {
                **valid_spec,
                "execution": {
                    "concurrency": "serial",
                    "read_only": 0,
                    "idempotent": False,
                },
            },
            "must be a boolean",
        ),
        invalid_case(
            decode_message,
            {"role": "user", "parts": [], "metadata": []},
            "metadata must be an object",
        ),
        invalid_case(decode_message, cast(Any, {1: "bad"}), "keys must be strings"),
        invalid_case(
            decode_message,
            {"role": "user", "parts": [], "metadata": {"bad": object()}},
            "non-JSON value",
        ),
    )
    for decoder, value, message in invalid_cases:
        with pytest.raises(ProtocolError, match=message):
            decoder(value)

    with pytest.raises(ProtocolError, match="must be finite"):
        number(inf, "number")
    assert optional_number(None, "optional") is None


def test_fact_and_run_view_codecs_reject_empty_or_illegal_variants() -> None:
    invalid_facts: tuple[tuple[dict[str, Any], str], ...] = (
        (
            {"kind": "started", "at": 1.0, "data": {"history_roles": []}},
            "history_roles must not be empty",
        ),
        (
            {
                "kind": "tool_batch",
                "at": 1.0,
                "data": {
                    "batch_id": "batch-1",
                    "call_ids": ["call-1"],
                    "parallel": False,
                    "outcome_kinds": [],
                    "suspension": None,
                },
            },
            "outcome_kinds must not be empty",
        ),
        (
            {
                "kind": "tool_batch",
                "at": 1.0,
                "data": {
                    "batch_id": "batch-1",
                    "call_ids": [],
                    "parallel": False,
                    "outcome_kinds": ["success"],
                    "suspension": None,
                },
            },
            "call_ids must not be empty",
        ),
        (
            {
                "kind": "history_rewrite",
                "at": 1.0,
                "data": {
                    "before_count": 1,
                    "after_roles": [],
                    "reason": "compact",
                    "metadata_keys": [],
                },
            },
            "after_roles must not be empty",
        ),
        ({"kind": "control", "at": 1.0, "data": {}}, "missing field.*action"),
    )
    for value, message in invalid_facts:
        with pytest.raises(ProtocolError, match=message):
            decode_fact(value)

    metrics = {
        "planning_steps": 0,
        "tool_calls": 0,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
        },
    }
    with pytest.raises(ProtocolError, match="state view is missing"):
        decode_run_view({"revision": 1, "history_count": 1, "metrics": metrics, "state": {}})
    with pytest.raises(ProtocolError, match="active state view"):
        decode_run_view(
            {
                "revision": 1,
                "history_count": 1,
                "metrics": metrics,
                "state": {
                    "kind": "suspended",
                    "resume_to": {"kind": "completed", "part_count": 1},
                    "suspension": {
                        "reason": "pause",
                        "source": "host",
                        "wait_id": None,
                        "metadata_keys": [],
                    },
                },
            }
        )
    with pytest.raises(ProtocolError, match="32-byte hex"):
        decode_run_view(
            {
                "revision": 1,
                "history_count": 1,
                "metrics": metrics,
                "state": {
                    "kind": "tools_pending",
                    "pending_count": 1,
                    "call_id_digest": "AB" * 32,
                },
            }
        )

    valid_states: tuple[Mapping[str, Any], ...] = (
        {"kind": "completed", "part_count": 1},
        {"kind": "failed", "code": "model_error"},
        {"kind": "limited", "reason": "deadline"},
        {
            "kind": "suspended",
            "resume_to": {"kind": "planning"},
            "suspension": {
                "reason": "pause",
                "source": "host",
                "wait_id": None,
                "metadata_keys": [],
            },
        },
    )
    for state in valid_states:
        decoded = decode_run_view(
            {"revision": 1, "history_count": 1, "metrics": metrics, "state": state}
        )
        assert decoded["state"] == state


def test_event_codec_rejects_missing_discriminators_and_starting_evidence() -> None:
    wrong_version = _event_wire(
        "invocation_stopped",
        {"reason": "cancelled", "last_checkpoint_id": None},
    )
    wrong_version["schema_version"] = "v1"
    with pytest.raises(ProtocolError, match="schema_version must be v0"):
        decode_event(wrong_version)

    continued = _event_wire(
        "invocation_started",
        {"request_kind": "continue", "starting_checkpoint_id": None, "starting": None},
    )
    with pytest.raises(ProtocolError, match="require a starting checkpoint"):
        decode_event(continued)

    missing_delta_kind = _event_wire("model_delta", {})
    with pytest.raises(ProtocolError, match="model_delta data is missing"):
        decode_event(missing_delta_kind)

    missing_decision_kind = _event_wire("approval_decided", {"call_id": "call-1"})
    with pytest.raises(ProtocolError, match="approval decision is missing"):
        decode_event(missing_decision_kind)

    uppercase_digest = _event_wire(
        "tool_batch_selected",
        {
            "batch_id": "batch-1",
            "call_ids": ["call-1"],
            "parallel": False,
            "remaining_count": 0,
            "remaining_call_id_digest": "AB" * 32,
        },
    )
    with pytest.raises(ProtocolError, match="32-byte hex"):
        decode_event(uppercase_digest)


def test_public_wire_decoders_cover_nested_values_and_empty_documents() -> None:
    assert decode_content_part({"type": "text", "text": "x", "metadata": {}}).text == "x"
    assert decode_tool_call({"id": "call-1", "name": "lookup", "arguments": {}}).id == "call-1"
    assert decode_error_info({"code": "bad", "message": "failed"}).code == "bad"
    assert (
        decode_model_usage(
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
            }
        ).total_tokens
        == 3
    )
    assert decode_metrics(
        {
            "planning_steps": 1,
            "tool_calls": 2,
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "reasoning_tokens": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
            },
        }
    ) == RunMetrics(1, 2)
    assert decode_suspension(
        {"reason": "pause", "source": "host", "wait_id": None, "metadata": {}}
    ) == Suspension("pause", "host")
    decoded_tool_message = decode_message(
        cast(
            object,
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "outcome": {
                    "kind": "success",
                    "parts": [{"type": "text", "text": "ok", "metadata": {}}],
                    "structured_content": None,
                },
                "metadata": {},
            },
        )
    )
    assert decoded_tool_message.tool_call_id == "call-1"
    assert encode_message(decoded_tool_message)["role"] == "tool"
    assert encode_content_part(ContentPart(type="opaque")) == {
        "type": "opaque",
        "metadata": {},
    }

    with pytest.raises(ProtocolError, match="message role has unsupported"):
        decode_message({"role": "unknown"})
    with pytest.raises(ProtocolError, match="tool outcome kind has unsupported"):
        decode_tool_outcome({"kind": "unknown"})
    with pytest.raises(ProtocolError, match="waiting result requires"):
        decode_tool_result(
            cast(
                object,
                {
                    "outcome": {
                        "kind": "success",
                        "parts": [{"type": "text", "text": "ok", "metadata": {}}],
                        "structured_content": None,
                    },
                    "suspension": {
                        "reason": "pause",
                        "source": "host",
                        "wait_id": None,
                        "metadata": {},
                    },
                },
            )
        )
    snapshot = encode_snapshot(_started().snapshot)
    snapshot["history"] = []
    with pytest.raises(ProtocolError, match="history must not be empty"):
        decode_snapshot(snapshot)
    with pytest.raises(ProtocolError, match="run state is missing"):
        decode_state({})
    with pytest.raises(ProtocolError, match="active state"):
        decode_state(
            cast(
                object,
                {
                    "kind": "suspended",
                    "resume_to": {
                        "kind": "completed",
                        "parts": [{"type": "text", "text": "done", "metadata": {}}],
                    },
                    "suspension": {
                        "reason": "pause",
                        "source": "host",
                        "wait_id": None,
                        "metadata": {},
                    },
                },
            )
        )


def test_request_and_trace_codecs_reject_empty_discriminated_documents() -> None:
    assert StartRequest(_history(Message.user("hello"))).context is None
    selector_request = ResumeRequest(
        _suspended(),
        SuspensionSelector(source="approval"),
    )
    assert encode_run_request(selector_request)["selector"] == {"source": "approval"}
    with pytest.raises(ValueError, match="must not be empty"):
        RunHistory(())
    with pytest.raises(ValueError, match="active checkpoint"):
        ContinueRequest(_suspended())
    with pytest.raises(ValueError, match="suspended checkpoint"):
        ResumeRequest(_started())
    with pytest.raises(ValueError, match="regular role"):
        ResumeRequest(
            _suspended(),
            append_messages=(Message.tool("call-1", ToolSuccess((ContentPart.text_part("ok"),))),),
        )

    with pytest.raises(ProtocolError, match="run request is missing"):
        decode_run_request({})
    with pytest.raises(ProtocolError, match="start messages must not be empty"):
        decode_run_request({"kind": "start", "messages": [], "context": None})
    empty_selector = encode_run_request(ResumeRequest(_suspended()))
    empty_selector["selector"] = {}
    with pytest.raises(ProtocolError, match="must set at least one field"):
        decode_run_request(empty_selector)

    trace = encode_trace(
        RunTrace(
            TraceHeader("run-1", "inv-1", "start"),
            (
                TraceEntry(
                    1,
                    EventKind.INVOCATION_STARTED,
                    1.0,
                    {
                        "request_kind": "start",
                        "starting_checkpoint_id": None,
                        "starting": None,
                    },
                ),
                TraceEntry(
                    2,
                    EventKind.INVOCATION_STOPPED,
                    2.0,
                    {"reason": "cancelled", "last_checkpoint_id": None},
                ),
            ),
        )
    )
    trace["schema_version"] = "v1"
    with pytest.raises(ProtocolError, match="schema_version must be v0"):
        decode_trace(trace)
    trace["schema_version"] = "v0"
    trace["entries"] = trace["entries"][:1]
    with pytest.raises(ProtocolError, match="at least two entries"):
        decode_trace(trace)


def test_encoders_defend_against_invalid_trusted_message_values() -> None:
    invalid_message = object.__new__(Message)
    object.__setattr__(invalid_message, "role", "tool")
    object.__setattr__(invalid_message, "outcome", None)
    object.__setattr__(invalid_message, "tool_call_id", None)
    with pytest.raises(ProtocolError, match="invalid trusted tool message"):
        encode_message(invalid_message)

    invalid_part = object.__new__(ContentPart)
    object.__setattr__(invalid_part, "type", "artifact")
    object.__setattr__(invalid_part, "artifact", None)
    with pytest.raises(ProtocolError, match="invalid trusted artifact"):
        encode_content_part(invalid_part)
