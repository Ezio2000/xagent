from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from jharness.kernel import (
    ApprovalAllow,
    ApprovalDeny,
    ApprovalSuspend,
    Checkpoint,
    Completed,
    ContentPart,
    ControlFact,
    ConversationInsertFact,
    ErrorInfo,
    EventKind,
    Failed,
    FailedControl,
    HistoryRewriteFact,
    Limited,
    LimitReason,
    Message,
    ModelTurnFact,
    ModelTurnResult,
    Planning,
    ResumedFact,
    RunContext,
    RunMetrics,
    RunSnapshot,
    SettledResult,
    StartedFact,
    Suspended,
    SuspendedControl,
    Suspension,
    ToolAccepted,
    ToolBatchFact,
    ToolCall,
    ToolExecution,
    ToolFailure,
    ToolOutcomeKind,
    ToolsPending,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.kernel.wire import (
    ContinueRequest,
    ResumeRequest,
    StartRequest,
    encode_fact,
    encode_message,
    encode_run_request,
    encode_state,
    encode_tool_result,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "contracts" / "v0"


def _schema(name: str) -> Mapping[str, Any]:
    value: object = json.loads((SPEC / name).read_text())
    assert isinstance(value, dict)
    return cast(Mapping[str, Any], value)


def _defs(name: str) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], _schema(name)["$defs"])


def _definition(name: str, definition: str) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], _defs(name)[definition])


def _properties(definition: Mapping[str, Any]) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], definition["properties"])


def _enum(value: object) -> set[str]:
    assert isinstance(value, list)
    items = cast(list[object], value)
    assert all(isinstance(item, str) for item in items)
    return set(cast(list[str], items))


def _kind_const(name: str, definition: str) -> str:
    kind = cast(Mapping[str, Any], _properties(_definition(name, definition))["kind"])
    return cast(str, kind["const"])


def _active_checkpoint() -> Checkpoint:
    snapshot = RunSnapshot(
        0,
        RunContext("run-1", 1.0),
        (Message.user("hello"),),
        RunMetrics(),
        Planning(),
    )
    return Checkpoint("checkpoint-0", snapshot, StartedFact(1.0, ("user",)))


def _suspended_checkpoint() -> Checkpoint:
    suspension = Suspension("pause", "host", "wait-1")
    snapshot = RunSnapshot(
        1,
        RunContext("run-1", 1.0),
        (Message.user("hello"),),
        RunMetrics(),
        Suspended(Planning(), suspension),
    )
    fact = ControlFact(2.0, SuspendedControl("pause", "host", "wait-1", ()))
    return Checkpoint("checkpoint-1", snapshot, fact)


def test_closed_lifecycle_and_limit_vocabularies_match_state_schema() -> None:
    call = ToolCall("call-1", "lookup")
    states = (
        Planning(),
        ToolsPending((call,)),
        Suspended(Planning(), Suspension("pause", "host")),
        Completed((ContentPart.text_part("done"),)),
        Failed(ErrorInfo("failed", "failed")),
        Limited(LimitReason.DEADLINE),
    )
    state_names = (
        "planning",
        "tools_pending",
        "suspended",
        "completed",
        "failed",
        "limited",
    )
    assert {cast(str, encode_state(state)["kind"]) for state in states} == {
        _kind_const("state.schema.json", name) for name in state_names
    }
    assert {reason.value for reason in LimitReason} == _enum(
        cast(Mapping[str, Any], _defs("state.schema.json")["limit_reason"])["enum"]
    )


def test_event_request_approval_tool_and_scheduling_vocabularies_match() -> None:
    event_kinds = _definition("events.schema.json", "event_kind")
    assert {kind.value for kind in EventKind} == _enum(event_kinds["enum"])

    request_documents = (
        encode_run_request(StartRequest((Message.user("hello"),))),
        encode_run_request(ContinueRequest(_active_checkpoint())),
        encode_run_request(ResumeRequest(_suspended_checkpoint())),
    )
    assert {cast(str, document["kind"]) for document in request_documents} == {
        _kind_const("run-request.schema.json", name)
        for name in ("start_request", "continue_request", "resume_request")
    }

    call_id = "call-1"
    decisions = (
        (ApprovalAllow(call_id), "allow"),
        (ApprovalDeny(call_id, "denied"), "deny"),
        (ApprovalSuspend(call_id, Suspension("approval", "policy")), "suspend"),
    )
    assert {kind for _, kind in decisions} == {
        _kind_const("approval.schema.json", name) for name in ("allow", "deny", "suspend")
    }

    results = (
        SettledResult(ToolSuccess((ContentPart.text_part("ok"),))),
        SettledResult(ToolFailure.from_error("failed", "failed")),
        SettledResult(ToolAccepted((ContentPart.text_part("accepted"),), "correlation")),
        WaitingResult(
            ToolWaiting((ContentPart.text_part("waiting"),)),
            Suspension("wait", "tool"),
        ),
    )
    encoded_results = tuple(encode_tool_result(result) for result in results)
    assert {
        cast(str, cast(Mapping[str, Any], result["outcome"])["kind"]) for result in encoded_results
    } == {
        _kind_const("messages.schema.json", name)
        for name in ("tool_success", "tool_failure", "tool_accepted", "tool_waiting")
    }
    settled_required = set(
        cast(list[str], _definition("tool-result.schema.json", "settled_result")["required"])
    )
    waiting_required = set(
        cast(list[str], _definition("tool-result.schema.json", "waiting_result")["required"])
    )
    assert all(set(result) == settled_required for result in encoded_results[:-1])
    assert set(encoded_results[-1]) == waiting_required

    concurrency = cast(
        Mapping[str, Any],
        _properties(_definition("tools.schema.json", "tool_execution"))["concurrency"],
    )
    assert {
        ToolExecution().concurrency,
        ToolExecution("parallel", True, True).concurrency,
    } == _enum(concurrency["enum"])


def test_message_roles_and_fact_payloads_match_contracts_exactly() -> None:
    call = ToolCall("call-1", "lookup")
    messages = (
        Message.system("system"),
        Message.user("user"),
        Message.external("external"),
        Message.assistant((ContentPart.text_part("assistant"),)),
        Message.tool(call.id, ToolSuccess((ContentPart.text_part("tool"),))),
    )
    message_defs = _defs("messages.schema.json")
    regular_roles = _enum(
        cast(
            Mapping[str, Any],
            _properties(cast(Mapping[str, Any], message_defs["regular_message"]))["role"],
        )["enum"]
    )
    assistant_role = cast(
        str,
        cast(
            Mapping[str, Any],
            _properties(cast(Mapping[str, Any], message_defs["assistant_message"]))["role"],
        )["const"],
    )
    tool_role = cast(
        str,
        cast(
            Mapping[str, Any],
            _properties(cast(Mapping[str, Any], message_defs["tool_message"]))["role"],
        )["const"],
    )
    assert {cast(str, encode_message(message)["role"]) for message in messages} == (
        regular_roles | {assistant_role, tool_role}
    )

    facts = (
        StartedFact(1.0, ("user",)),
        ResumedFact(1.0, (), ()),
        ModelTurnFact(
            1.0,
            ModelTurnResult.COMPLETED,
            1,
            (),
            "end_turn",
            None,
            None,
        ),
        ToolBatchFact(
            1.0,
            "batch-1",
            ("call-1",),
            False,
            (ToolOutcomeKind.SUCCESS,),
            None,
        ),
        ConversationInsertFact(1.0, "host"),
        HistoryRewriteFact(1.0, 2, ("user",), "compact", ()),
        ControlFact(1.0, FailedControl("failed")),
    )
    encoded_facts = tuple(encode_fact(fact) for fact in facts)
    fact_kind_schema = cast(
        Mapping[str, Any],
        _properties(_definition("checkpoint.schema.json", "fact"))["kind"],
    )
    assert {cast(str, fact["kind"]) for fact in encoded_facts} == _enum(fact_kind_schema["enum"])
    definition_by_kind = {
        "started": "started_data",
        "resumed": "resumed_data",
        "model_turn": "model_turn_data",
        "tool_batch": "tool_batch_data",
        "conversation_insert": "conversation_insert_data",
        "history_rewrite": "history_rewrite_data",
    }
    for fact in encoded_facts[:-1]:
        kind = cast(str, fact["kind"])
        data = cast(Mapping[str, Any], fact["data"])
        definition = _definition("checkpoint.schema.json", definition_by_kind[kind])
        assert set(data) == set(cast(list[str], definition["required"]))

    control = cast(Mapping[str, Any], encoded_facts[-1]["data"])
    failed_action = cast(
        Mapping[str, Any],
        _properties(_definition("checkpoint.schema.json", "failed_control"))["action"],
    )
    assert control["action"] == failed_action["const"]
