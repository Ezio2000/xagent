"""Exact durable and selected live-event fixture assertions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from conformance._execution import InvocationOutcome
from conformance._values import boolean, integer, mapping, sequence, string, string_list
from jharness.kernel import (
    Completed,
    EventKind,
    Failed,
    Limited,
    Suspended,
    ToolsPending,
    thaw_json_value,
)
from jharness.kernel.wire import encode_content_part, encode_model_usage, encode_suspension


@dataclass(slots=True)
class _EventProjection:
    counts: Counter[str]
    approval_call_ids: list[str]
    progress: list[Any]
    tool_activity: list[dict[str, str]]
    active: int
    peak: int
    unmatched_finish: bool


def assert_invocation(
    outcome: InvocationOutcome,
    expected: Mapping[str, Any],
) -> None:
    snapshot = outcome.snapshot
    _equal(snapshot.status, string(expected["status"], "expected status"), "status")
    _equal(snapshot.revision, integer(expected["revision"], "expected revision"), "revision")
    _equal(
        [checkpoint.fact.kind for checkpoint in outcome.commits],
        string_list(expected["fact_kinds"], "expected fact_kinds"),
        "fact_kinds",
    )
    _equal(
        [message.role for message in snapshot.history],
        string_list(expected["message_roles"], "expected message_roles"),
        "message_roles",
    )
    _equal(
        snapshot.metrics.planning_steps,
        integer(expected["planning_steps"], "expected planning_steps"),
        "planning_steps",
    )
    _equal(
        snapshot.metrics.tool_calls,
        integer(expected["tool_calls"], "expected tool_calls"),
        "tool_calls",
    )
    _equal(
        list(_pending_call_ids(snapshot.state)),
        string_list(expected.get("pending_call_ids", ()), "expected pending_call_ids"),
        "pending_call_ids",
    )
    _equal(
        [cast(Any, message.outcome).kind for message in snapshot.history if message.role == "tool"],
        string_list(
            expected.get("tool_outcome_kinds", ()),
            "expected tool_outcome_kinds",
        ),
        "tool_outcome_kinds",
    )
    if "usage" in expected:
        _equal(encode_model_usage(snapshot.metrics.usage), expected["usage"], "usage")
    _optional_outcome_assertions(outcome, expected)
    _optional_event_assertions(outcome, expected)
    _optional_model_assertions(outcome, expected)
    if outcome.trace is None and outcome.request_error is None:
        raise AssertionError("completed invocation did not produce a trace")
    if "repository_idempotent" in expected:
        _equal(
            outcome.repository_idempotent,
            boolean(expected["repository_idempotent"], "expected repository_idempotent"),
            "repository_idempotent",
        )


def _optional_outcome_assertions(
    outcome: InvocationOutcome,
    expected: Mapping[str, Any],
) -> None:
    _assert_final_output(outcome, expected)
    _assert_suspension(outcome, expected)
    _assert_error(outcome, expected)
    _assert_limit(outcome, expected)
    _assert_repository_error(outcome, expected)
    _assert_request_error(outcome, expected)


def _assert_final_output(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    state = outcome.snapshot.state
    parts = state.parts if isinstance(state, Completed) else ()
    if "final_text" in expected:
        text = "".join(part.text or "" for part in parts if part.type == "text")
        _equal(text, string(expected["final_text"], "expected final_text"), "final_text")
    if "final_part_types" in expected:
        _equal(
            [part.type for part in parts],
            string_list(expected["final_part_types"], "expected final_part_types"),
            "final_part_types",
        )
    if "final_parts" in expected:
        _equal(
            [encode_content_part(part) for part in parts],
            list(sequence(expected["final_parts"], "expected final_parts")),
            "final_parts",
        )


def _assert_suspension(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    if "suspension" not in expected:
        return
    state = outcome.snapshot.state
    actual = encode_suspension(state.suspension) if isinstance(state, Suspended) else None
    _equal(actual, expected["suspension"], "suspension")


def _assert_error(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    if "error_code" not in expected:
        return
    state = outcome.snapshot.state
    actual = state.error.code if isinstance(state, Failed) else None
    _equal(actual, string(expected["error_code"], "expected error_code"), "error_code")


def _assert_limit(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    if "limit_reason" not in expected:
        return
    state = outcome.snapshot.state
    actual = state.reason.value if isinstance(state, Limited) else None
    _equal(actual, string(expected["limit_reason"], "expected limit_reason"), "limit_reason")


def _assert_repository_error(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    if "repository_error" in expected:
        wanted = string(expected["repository_error"], "expected repository_error")
        if outcome.repository_error is None or wanted not in outcome.repository_error:
            raise AssertionError(
                f"repository_error: expected substring {wanted!r}, got {outcome.repository_error!r}"
            )
    elif outcome.repository_error is not None:
        raise AssertionError(f"unexpected repository_error: {outcome.repository_error}")


def _assert_request_error(outcome: InvocationOutcome, expected: Mapping[str, Any]) -> None:
    if "request_error" in expected:
        _equal(
            outcome.request_error,
            string(expected["request_error"], "expected request_error"),
            "request_error",
        )
    elif outcome.request_error is not None:
        raise AssertionError(f"unexpected request_error: {outcome.request_error}")


def _optional_event_assertions(
    outcome: InvocationOutcome,
    expected: Mapping[str, Any],
) -> None:
    projection = _project_events(outcome)
    _assert_event_inventory(projection, expected)
    _assert_event_values(projection, expected)


def _project_events(outcome: InvocationOutcome) -> _EventProjection:
    result = _EventProjection(Counter(), [], [], [], 0, 0, False)
    for event in outcome.events:
        kind = event.kind.value
        result.counts[kind] += 1
        if event.kind is EventKind.APPROVAL_REQUESTED:
            call = mapping(event.data["call"], "approval call")
            result.approval_call_ids.append(string(call["id"], "approval call id"))
        elif event.kind is EventKind.TOOL_PROGRESS:
            result.progress.append(thaw_json_value(event.data["progress"]))
        elif event.kind is EventKind.TOOL_STARTED:
            call = mapping(event.data["call"], "started tool call")
            call_id = string(call["id"], "started tool call id")
            result.tool_activity.append({"kind": kind, "tool_call_id": call_id})
            result.active += 1
            result.peak = max(result.peak, result.active)
        elif event.kind is EventKind.TOOL_FINISHED:
            call_id = string(event.data["tool_call_id"], "finished tool call id")
            result.tool_activity.append({"kind": kind, "tool_call_id": call_id})
            result.active -= 1
            result.unmatched_finish |= result.active < 0
    return result


def _assert_event_inventory(
    projection: _EventProjection,
    expected: Mapping[str, Any],
) -> None:
    if "event_counts" in expected:
        for kind, raw_count in mapping(expected["event_counts"], "event_counts").items():
            _equal(
                projection.counts[kind],
                integer(raw_count, f"event count {kind}"),
                f"event_counts.{kind}",
            )
    if "forbidden_event_kinds" in expected:
        forbidden = set(string_list(expected["forbidden_event_kinds"], "forbidden_event_kinds"))
        present = forbidden.intersection(projection.counts)
        if present:
            raise AssertionError(f"forbidden event kind(s) observed: {sorted(present)}")


def _assert_event_values(
    projection: _EventProjection,
    expected: Mapping[str, Any],
) -> None:
    if "approval_call_ids" in expected:
        _equal(
            projection.approval_call_ids,
            string_list(expected["approval_call_ids"], "approval_call_ids"),
            "approval_call_ids",
        )
    if "progress" in expected:
        _equal(
            projection.progress,
            list(sequence(expected["progress"], "expected progress")),
            "progress",
        )
    if "tool_activity" in expected:
        _equal(
            projection.tool_activity,
            list(sequence(expected["tool_activity"], "expected tool_activity")),
            "tool_activity",
        )
    if "max_active_tools" in expected:
        if projection.unmatched_finish:
            raise AssertionError("tool_finished observed without active tool")
        if projection.active != 0:
            raise AssertionError(f"{projection.active} tool(s) remained active after invocation")
        _equal(
            projection.peak,
            integer(expected["max_active_tools"], "expected max_active_tools"),
            "max_active_tools",
        )


def _optional_model_assertions(
    outcome: InvocationOutcome,
    expected: Mapping[str, Any],
) -> None:
    if "model_request_roles" not in expected:
        return
    actual = (
        []
        if outcome.model is None
        else [[message.role for message in request.messages] for request in outcome.model.requests]
    )
    wanted = [
        string_list(roles, "model request roles")
        for roles in sequence(expected["model_request_roles"], "model_request_roles")
    ]
    _equal(actual, wanted, "model_request_roles")


def _pending_call_ids(state: object) -> tuple[str, ...]:
    pending = state.resume_to if isinstance(state, Suspended) else state
    if isinstance(pending, ToolsPending):
        return tuple(call.id for call in pending.pending)
    return ()


def _equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
