from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import pytest

from jharness.kernel._engine.verification import fact_data, run_view, verify_change
from jharness.kernel.checkpoint import (
    Checkpoint,
    ModelTurnFact,
    ModelTurnResult,
    StartedFact,
    ToolBatchFact,
    ToolOutcomeKind,
)
from jharness.kernel.context import RunContext
from jharness.kernel.diagnostics import (
    RunTrace,
    TraceEntry,
    TraceError,
    TraceHeader,
    build_trace,
    verify_trace,
)
from jharness.kernel.events import Event, EventKind
from jharness.kernel.messages import ContentPart, Message
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import Completed, Planning, RunMetrics

_USAGE = {
    "input_tokens": None,
    "output_tokens": None,
    "total_tokens": None,
    "reasoning_tokens": None,
    "cache_read_tokens": None,
    "cache_write_tokens": None,
}


def _events(items: Sequence[tuple[EventKind, Mapping[str, Any]]]) -> tuple[Event, ...]:
    return tuple(
        Event("run-1", "inv-1", sequence, kind, float(sequence), data)
        for sequence, (kind, data) in enumerate(items, start=1)
    )


def _checkpoint_data(checkpoint: Checkpoint) -> Mapping[str, Any]:
    return {
        "checkpoint_id": checkpoint.id,
        "fact": fact_data(checkpoint.fact),
        "after": run_view(checkpoint.snapshot),
    }


def _completed_events(*, delta_count: int = 1) -> tuple[Event, ...]:
    context = RunContext("run-1", 0)
    user = Message.user("hi")
    started = Checkpoint(
        "cp-0",
        RunSnapshot(0, context, (user,), RunMetrics(), Planning()),
        StartedFact(1, ("user",)),
    )
    part = ContentPart.text_part("done")
    completed = Checkpoint(
        "cp-1",
        RunSnapshot(
            1,
            context,
            (user, Message.assistant((part,))),
            RunMetrics(planning_steps=1),
            Completed((part,)),
        ),
        ModelTurnFact(
            4,
            ModelTurnResult.COMPLETED,
            1,
            (),
            "end_turn",
            None,
            None,
        ),
    )
    deltas: list[tuple[EventKind, Mapping[str, Any]]] = [
        (
            EventKind.MODEL_DELTA,
            {
                "kind": "content",
                "index": 0,
                "part_type": "text",
                "text_delta": "x",
                "data": {},
            },
        )
        for _ in range(delta_count)
    ]
    return _events(
        [
            (
                EventKind.INVOCATION_STARTED,
                {
                    "request_kind": "start",
                    "starting_checkpoint_id": None,
                    "starting": None,
                },
            ),
            (EventKind.CHECKPOINT_COMMITTED, _checkpoint_data(started)),
            (EventKind.MODEL_STARTED, {"planning_step": 1}),
            *deltas,
            (
                EventKind.MODEL_FINISHED,
                {"finish_reason": "end_turn", "tool_call_count": 0, "usage": None},
            ),
            (EventKind.CHECKPOINT_COMMITTED, _checkpoint_data(completed)),
            (
                EventKind.INVOCATION_STOPPED,
                {"reason": "terminal", "last_checkpoint_id": "cp-1"},
            ),
        ]
    )


def _parallel_tool_events() -> tuple[Event, ...]:
    starting = {
        "revision": 2,
        "history_count": 2,
        "metrics": {"planning_steps": 1, "tool_calls": 0, "usage": dict(_USAGE)},
        "state": {"kind": "tools_pending", "call_ids": ["call-1", "call-2"]},
    }
    fact = ToolBatchFact(
        5,
        "batch-1",
        ("call-1", "call-2"),
        True,
        (ToolOutcomeKind.SUCCESS, ToolOutcomeKind.FAILURE),
        None,
    )
    after = {
        "revision": 3,
        "history_count": 4,
        "metrics": {"planning_steps": 1, "tool_calls": 2, "usage": dict(_USAGE)},
        "state": {"kind": "planning"},
    }
    return _events(
        [
            (
                EventKind.INVOCATION_STARTED,
                {
                    "request_kind": "continue",
                    "starting_checkpoint_id": "cp-2",
                    "starting": starting,
                },
            ),
            (
                EventKind.TOOL_STARTED,
                {
                    "batch_id": "batch-1",
                    "index": 0,
                    "call": {"id": "call-1", "name": "one", "arguments": {}},
                    "parallel": True,
                },
            ),
            (
                EventKind.TOOL_STARTED,
                {
                    "batch_id": "batch-1",
                    "index": 1,
                    "call": {"id": "call-2", "name": "two", "arguments": {}},
                    "parallel": True,
                },
            ),
            (
                EventKind.TOOL_FINISHED,
                {
                    "batch_id": "batch-1",
                    "index": 1,
                    "tool_call_id": "call-2",
                    "outcome_kind": "failure",
                },
            ),
            (
                EventKind.TOOL_FINISHED,
                {
                    "batch_id": "batch-1",
                    "index": 0,
                    "tool_call_id": "call-1",
                    "outcome_kind": "success",
                },
            ),
            (
                EventKind.CHECKPOINT_COMMITTED,
                {"checkpoint_id": "cp-3", "fact": fact_data(fact), "after": after},
            ),
            (
                EventKind.INVOCATION_STOPPED,
                {"reason": "consumer_closed", "last_checkpoint_id": "cp-3"},
            ),
        ]
    )


def _replace_entry(trace: RunTrace, index: int, entry: TraceEntry) -> RunTrace:
    entries = list(trace.entries)
    entries[index] = entry
    return RunTrace(trace.header, tuple(entries))


def _renumber(trace: RunTrace, entries: Sequence[TraceEntry]) -> RunTrace:
    return RunTrace(
        trace.header,
        tuple(replace(entry, sequence=index) for index, entry in enumerate(entries, start=1)),
    )


def _pending_trace(
    middle: Sequence[tuple[EventKind, Mapping[str, Any]]],
) -> RunTrace:
    starting = {
        "revision": 2,
        "history_count": 2,
        "metrics": {"planning_steps": 1, "tool_calls": 0, "usage": dict(_USAGE)},
        "state": {"kind": "tools_pending", "call_ids": ["call-1", "call-2"]},
    }
    events = _events(
        [
            (
                EventKind.INVOCATION_STARTED,
                {
                    "request_kind": "continue",
                    "starting_checkpoint_id": "cp-2",
                    "starting": starting,
                },
            ),
            *middle,
            (
                EventKind.INVOCATION_STOPPED,
                {"reason": "consumer_closed", "last_checkpoint_id": "cp-2"},
            ),
        ]
    )
    return build_trace(events, "continue")


def test_build_trace_compacts_identity_and_verifies_shared_fact_rules() -> None:
    events = _completed_events()
    trace = build_trace(events, "start", ("tenant", "request"))
    result = verify_trace(trace)

    assert trace.header.run_id == "run-1"
    assert trace.header.invocation_id == "inv-1"
    assert trace.header.metadata_keys == ("tenant", "request")
    assert len(trace.entries) == len(events)
    assert all(entry.data is event.data for entry, event in zip(trace.entries, events, strict=True))
    assert not hasattr(trace.entries[0], "run_id")
    assert not hasattr(trace.entries[0], "invocation_id")
    assert not hasattr(trace.entries[-2], "before")
    assert not hasattr(trace, "final_summary")
    assert result.checkpoint_count == 2
    assert result.live_event_count == len(events) - 2
    assert result.final_checkpoint_id == "cp-1"
    assert result.final_view is trace.entries[-2].data["after"]

    with pytest.raises(TypeError, match="immutable"):
        cast(Any, trace.entries[0].data)["extra"] = True


def test_build_trace_rejects_incomplete_or_mixed_invocations() -> None:
    events = _completed_events()
    with pytest.raises(ValueError, match="end with invocation_stopped"):
        build_trace(events[:-1], "start")

    last = events[-1]
    mixed = (
        *events[:-1],
        Event(
            "other-run",
            last.invocation_id,
            last.sequence,
            last.kind,
            last.created_at,
            last.data,
        ),
    )
    with pytest.raises(ValueError, match="run_id"):
        build_trace(mixed, "start")


def test_trace_values_reject_invalid_shapes() -> None:
    header = TraceHeader("run-1", "inv-1", "start")
    started = TraceEntry(
        1,
        EventKind.INVOCATION_STARTED,
        1.0,
        {"request_kind": "start", "starting_checkpoint_id": None, "starting": None},
    )
    stopped = TraceEntry(
        2,
        EventKind.INVOCATION_STOPPED,
        2.0,
        {"reason": "cancelled", "last_checkpoint_id": None},
    )

    with pytest.raises(ValueError, match="run_id"):
        TraceHeader("", "inv-1", "start")
    with pytest.raises(ValueError, match="invocation_id"):
        TraceHeader("run-1", "", "start")
    with pytest.raises(ValueError, match="request kind"):
        TraceHeader("run-1", "inv-1", cast(Any, "invalid"))
    with pytest.raises(TypeError, match="metadata_keys must be a sequence"):
        TraceHeader("run-1", "inv-1", "start", cast(Any, "tenant"))
    with pytest.raises(TypeError, match="contain strings"):
        TraceHeader("run-1", "inv-1", "start", cast(Any, (1,)))
    with pytest.raises(ValueError, match="must be unique"):
        TraceHeader("run-1", "inv-1", "start", ("tenant", "tenant"))

    with pytest.raises(TypeError, match="sequence must be an integer"):
        TraceEntry(cast(Any, True), EventKind.MODEL_DELTA, 1.0)
    with pytest.raises(ValueError, match="sequence must be >= 1"):
        TraceEntry(0, EventKind.MODEL_DELTA, 1.0)
    with pytest.raises(TypeError, match="kind must be EventKind"):
        TraceEntry(1, cast(Any, "model_delta"), 1.0)
    with pytest.raises(TypeError, match="created_at must be a number"):
        TraceEntry(1, EventKind.MODEL_DELTA, cast(Any, True))
    with pytest.raises(ValueError, match="finite"):
        TraceEntry(1, EventKind.MODEL_DELTA, float("inf"))
    with pytest.raises(TypeError, match="data must be a mapping"):
        TraceEntry(1, EventKind.MODEL_DELTA, 1.0, cast(Any, []))
    with pytest.raises(TypeError, match="data keys must be strings"):
        TraceEntry(1, EventKind.MODEL_DELTA, 1.0, cast(Any, {1: "x"}))

    with pytest.raises(TypeError, match="header must be TraceHeader"):
        RunTrace(cast(Any, object()), (started, stopped))
    with pytest.raises(TypeError, match="entries must be a sequence"):
        RunTrace(header, cast(Any, "entries"))
    with pytest.raises(ValueError, match="at least two entries"):
        RunTrace(header, (started,))
    with pytest.raises(TypeError, match="contain TraceEntry"):
        RunTrace(header, cast(Any, (started, object())))


def test_build_trace_rejects_every_invalid_envelope_shape() -> None:
    events = _completed_events()
    with pytest.raises(ValueError, match="unsupported trace request"):
        build_trace(events, cast(Any, "invalid"))
    with pytest.raises(ValueError, match="at least two events"):
        build_trace(events[:1], "start")
    with pytest.raises(TypeError, match="requires Event"):
        build_trace(cast(Any, (events[0], object(), events[-1])), "start")
    with pytest.raises(ValueError, match="start with"):
        build_trace(events[1:], "start")
    with pytest.raises(ValueError, match="end with"):
        build_trace(events[:2], "start")

    last = events[-1]
    wrong_invocation = (
        *events[:-1],
        Event(
            last.run_id,
            "other-invocation",
            last.sequence,
            last.kind,
            last.created_at,
            last.data,
        ),
    )
    with pytest.raises(ValueError, match="invocation_id"):
        build_trace(wrong_invocation, "start")
    unordered = (*events[:-1], replace(last, sequence=events[-2].sequence))
    with pytest.raises(ValueError, match="strictly increase"):
        build_trace(unordered, "start")
    with pytest.raises(ValueError, match="must match"):
        build_trace(events, "continue")


def test_verifier_rejects_invalid_lifecycle_and_start_evidence() -> None:
    trace = build_trace(_completed_events(), "start")
    with pytest.raises(TypeError, match="trace must be RunTrace"):
        verify_trace(cast(Any, object()))
    with pytest.raises(ValueError, match="code must not be empty"):
        TraceError("", "bad")

    wrong_first = replace(trace.entries[0], kind=EventKind.MODEL_STARTED)
    with pytest.raises(TraceError) as first_error:
        verify_trace(_replace_entry(trace, 0, wrong_first))
    assert first_error.value.code == "lifecycle"

    wrong_last = replace(trace.entries[-1], kind=EventKind.MODEL_FINISHED)
    with pytest.raises(TraceError) as last_error:
        verify_trace(_replace_entry(trace, len(trace.entries) - 1, wrong_last))
    assert last_error.value.code == "lifecycle"

    header_mismatch = RunTrace(
        replace(trace.header, request_kind="continue"),
        trace.entries,
    )
    with pytest.raises(TraceError) as header_error:
        verify_trace(header_mismatch)
    assert header_error.value.code == "request_mismatch"

    carried_checkpoint = replace(
        trace.entries[0],
        data={
            "request_kind": "start",
            "starting_checkpoint_id": "cp-0",
            "starting": None,
        },
    )
    with pytest.raises(TraceError) as start_error:
        verify_trace(_replace_entry(trace, 0, carried_checkpoint))
    assert start_error.value.code == "request_mismatch"

    missing_view = _pending_trace(())
    invalid_continue = replace(
        missing_view.entries[0],
        data={
            "request_kind": "continue",
            "starting_checkpoint_id": None,
            "starting": None,
        },
    )
    with pytest.raises(TraceError) as continue_error:
        verify_trace(_replace_entry(missing_view, 0, invalid_continue))
    assert continue_error.value.code == "request_mismatch"


def test_model_lifecycle_and_fact_mismatches_are_rejected() -> None:
    trace = build_trace(_completed_events(), "start")
    without_initial_checkpoint = _renumber(
        trace,
        (trace.entries[0], *trace.entries[2:]),
    )
    with pytest.raises(TraceError) as state_error:
        verify_trace(without_initial_checkpoint)
    assert state_error.value.code == "model_lifecycle"

    entries = list(trace.entries)
    entries.insert(3, trace.entries[2])
    with pytest.raises(TraceError) as active_error:
        verify_trace(_renumber(trace, entries))
    assert active_error.value.code == "model_lifecycle"

    wrong_step = replace(trace.entries[2], data={"planning_step": 2})
    with pytest.raises(TraceError) as step_error:
        verify_trace(_replace_entry(trace, 2, wrong_step))
    assert step_error.value.code == "model_lifecycle"

    finished_index = len(trace.entries) - 3
    wrong_count = replace(
        trace.entries[finished_index],
        data={"finish_reason": "end_turn", "tool_call_count": 1, "usage": None},
    )
    with pytest.raises(TraceError) as count_error:
        verify_trace(_replace_entry(trace, finished_index, wrong_count))
    assert count_error.value.code == "model_fact_mismatch"

    wrong_reason = replace(
        trace.entries[finished_index],
        data={"finish_reason": "other", "tool_call_count": 0, "usage": None},
    )
    with pytest.raises(TraceError) as reason_error:
        verify_trace(_replace_entry(trace, finished_index, wrong_reason))
    assert reason_error.value.code == "model_fact_mismatch"


def test_approval_and_tool_lifecycle_evidence_is_checked() -> None:
    approval_request: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.APPROVAL_REQUESTED,
        {"call": {"id": "call-1", "name": "one", "arguments": {}}},
    )
    approval_decision: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.APPROVAL_DECIDED,
        {"call_id": "call-1"},
    )
    assert verify_trace(_pending_trace((approval_request, approval_decision))).checkpoint_count == 0

    with pytest.raises(TraceError) as decision_error:
        verify_trace(_pending_trace((approval_decision,)))
    assert decision_error.value.code == "approval_lifecycle"

    nonpending_request: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.APPROVAL_REQUESTED,
        {"call": {"id": "other", "name": "one", "arguments": {}}},
    )
    with pytest.raises(TraceError) as pending_error:
        verify_trace(_pending_trace((nonpending_request,)))
    assert pending_error.value.code == "approval_lifecycle"

    with pytest.raises(TraceError) as duplicate_error:
        verify_trace(
            _pending_trace(
                (approval_request, approval_decision, approval_request),
            )
        )
    assert duplicate_error.value.code == "approval_lifecycle"

    tool_started: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.TOOL_STARTED,
        {
            "batch_id": "batch-1",
            "index": 0,
            "call": {"id": "call-1", "name": "one", "arguments": {}},
            "parallel": False,
        },
    )
    tool_finished: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.TOOL_FINISHED,
        {
            "batch_id": "batch-1",
            "index": 0,
            "tool_call_id": "call-1",
            "outcome_kind": "success",
        },
    )
    active_trace = _pending_trace(
        (
            tool_started,
            (EventKind.TOOL_PROGRESS, {"tool_call_id": "call-1"}),
            (EventKind.TOOL_CANCEL_REQUESTED, {"tool_call_id": "call-1"}),
            tool_finished,
        )
    )
    assert verify_trace(active_trace).checkpoint_count == 0

    with pytest.raises(TraceError) as active_tool_error:
        verify_trace(_pending_trace(((EventKind.TOOL_PROGRESS, {"tool_call_id": "call-1"}),)))
    assert active_tool_error.value.code == "tool_lifecycle"

    bad_finish: tuple[EventKind, Mapping[str, Any]] = (
        EventKind.TOOL_FINISHED,
        {
            "batch_id": "other-batch",
            "index": 0,
            "tool_call_id": "call-1",
            "outcome_kind": "success",
        },
    )
    with pytest.raises(TraceError) as finish_error:
        verify_trace(_pending_trace((tool_started, bad_finish)))
    assert finish_error.value.code == "tool_lifecycle"


def test_checkpoint_and_stop_evidence_rejects_conflicts() -> None:
    trace = build_trace(_completed_events(), "start")
    final_commit_index = len(trace.entries) - 2
    final_commit = trace.entries[final_commit_index]
    duplicate_id = replace(final_commit, data={**final_commit.data, "checkpoint_id": "cp-0"})
    with pytest.raises(TraceError) as duplicate_error:
        verify_trace(_replace_entry(trace, final_commit_index, duplicate_id))
    assert duplicate_error.value.code == "duplicate_checkpoint_id"

    wrong_endpoint = replace(
        trace.entries[-1], data={"reason": "terminal", "last_checkpoint_id": "cp-0"}
    )
    with pytest.raises(TraceError) as endpoint_error:
        verify_trace(_replace_entry(trace, len(trace.entries) - 1, wrong_endpoint))
    assert endpoint_error.value.code == "stop_mismatch"

    unsupported = replace(
        trace.entries[-1],
        data={"reason": "unknown", "last_checkpoint_id": "cp-1"},
    )
    with pytest.raises(TraceError) as reason_error:
        verify_trace(_replace_entry(trace, len(trace.entries) - 1, unsupported))
    assert reason_error.value.code == "stop_mismatch"

    nonterminal = _pending_trace(())
    terminal_stop = replace(
        nonterminal.entries[-1],
        data={"reason": "terminal", "last_checkpoint_id": "cp-2"},
    )
    with pytest.raises(TraceError) as terminal_error:
        verify_trace(_replace_entry(nonterminal, len(nonterminal.entries) - 1, terminal_stop))
    assert terminal_error.value.code == "stop_mismatch"


def test_verify_trace_reports_stable_sequence_and_revision_codes() -> None:
    trace = build_trace(_completed_events(), "start")
    duplicate_sequence = replace(trace.entries[2], sequence=trace.entries[1].sequence)
    with pytest.raises(TraceError) as sequence_error:
        verify_trace(_replace_entry(trace, 2, duplicate_sequence))
    assert sequence_error.value.code == "sequence_order"
    assert sequence_error.value.sequence == trace.entries[1].sequence

    commit_index = len(trace.entries) - 2
    commit = trace.entries[commit_index]
    after = dict(cast(Mapping[str, Any], commit.data["after"]))
    after["revision"] = cast(int, after["revision"]) + 1
    revision_gap = replace(
        commit,
        data={
            "checkpoint_id": commit.data["checkpoint_id"],
            "fact": commit.data["fact"],
            "after": after,
        },
    )
    with pytest.raises(TraceError) as revision_error:
        verify_trace(_replace_entry(trace, commit_index, revision_gap))
    assert revision_error.value.code == "revision_gap"
    assert revision_error.value.sequence == commit.sequence


def test_model_delta_requires_an_active_model_and_turn_requires_finish() -> None:
    trace = build_trace(_completed_events(), "start")
    without_start = RunTrace(trace.header, (*trace.entries[:2], *trace.entries[3:]))
    with pytest.raises(TraceError) as delta_error:
        verify_trace(without_start)
    assert delta_error.value.code == "model_lifecycle"

    finished_index = len(trace.entries) - 3
    without_finish = RunTrace(
        trace.header,
        (*trace.entries[:finished_index], *trace.entries[finished_index + 1 :]),
    )
    with pytest.raises(TraceError) as finish_error:
        verify_trace(without_finish)
    assert finish_error.value.code == "model_lifecycle"


def test_parallel_tool_completion_order_may_differ_but_fact_keeps_model_order() -> None:
    trace = build_trace(_parallel_tool_events(), "continue")

    result = verify_trace(trace)

    assert result.checkpoint_count == 1
    fact = cast(Mapping[str, Any], trace.entries[-2].data["fact"])
    data = cast(Mapping[str, Any], fact["data"])
    assert list(cast(Sequence[str], data["call_ids"])) == ["call-1", "call-2"]


def test_tool_completion_must_match_the_atomic_batch_fact() -> None:
    trace = build_trace(_parallel_tool_events(), "continue")
    finished = trace.entries[3]
    mismatch = replace(
        finished,
        data={**finished.data, "outcome_kind": "success"},
    )

    with pytest.raises(TraceError) as error:
        verify_trace(_replace_entry(trace, 3, mismatch))

    assert error.value.code == "tool_fact_mismatch"


def test_tool_fact_rejects_a_missing_observed_completion() -> None:
    trace = build_trace(_parallel_tool_events(), "continue")
    commit_index = len(trace.entries) - 2
    commit = trace.entries[commit_index]
    fact = dict(cast(Mapping[str, Any], commit.data["fact"]))
    data = dict(cast(Mapping[str, Any], fact["data"]))
    data["outcome_kinds"] = ["success", "success"]
    fact["data"] = data
    both_successful = _replace_entry(
        trace,
        commit_index,
        replace(commit, data={**commit.data, "fact": fact}),
    )
    without_second_tool = RunTrace(
        both_successful.header,
        (*both_successful.entries[:2], *both_successful.entries[4:]),
    )

    with pytest.raises(TraceError) as error:
        verify_trace(without_second_tool)

    assert error.value.code == "tool_fact_mismatch"


def test_tool_fact_rejects_an_extra_observed_completion() -> None:
    trace = build_trace(_parallel_tool_events(), "continue")
    commit_index = len(trace.entries) - 2
    commit = trace.entries[commit_index]
    fact = dict(cast(Mapping[str, Any], commit.data["fact"]))
    data = dict(cast(Mapping[str, Any], fact["data"]))
    data["call_ids"] = ["call-1"]
    data["outcome_kinds"] = ["success"]
    fact["data"] = data
    after = {
        "revision": 3,
        "history_count": 3,
        "metrics": {"planning_steps": 1, "tool_calls": 1, "usage": dict(_USAGE)},
        "state": {"kind": "tools_pending", "call_ids": ["call-2"]},
    }
    partial_commit = replace(
        commit,
        data={"checkpoint_id": "cp-3", "fact": fact, "after": after},
    )

    with pytest.raises(TraceError) as error:
        verify_trace(_replace_entry(trace, commit_index, partial_commit))

    assert error.value.code == "tool_fact_mismatch"


def test_tool_fact_rejects_duplicate_call_ids_with_a_stable_code() -> None:
    trace = build_trace(_parallel_tool_events(), "continue")
    commit_index = len(trace.entries) - 2
    commit = trace.entries[commit_index]
    fact = dict(cast(Mapping[str, Any], commit.data["fact"]))
    data = dict(cast(Mapping[str, Any], fact["data"]))
    data["call_ids"] = ["call-1", "call-1"]
    fact["data"] = data
    duplicate_fact = replace(commit, data={**commit.data, "fact": fact})

    with pytest.raises(TraceError) as error:
        verify_trace(_replace_entry(trace, commit_index, duplicate_fact))

    assert error.value.code == "tool_fact_mismatch"


def test_trace_growth_is_one_entry_per_event_without_payload_duplication() -> None:
    small_events = _completed_events(delta_count=8)
    large_events = _completed_events(delta_count=512)
    small = build_trace(small_events, "start")
    large = build_trace(large_events, "start")

    assert len(small.entries) == len(small_events)
    assert len(large.entries) == len(large_events)
    assert all(
        entry.data is event.data for entry, event in zip(large.entries, large_events, strict=True)
    )
    assert verify_trace(large).entry_count == len(large_events)


def test_history_rewrite_fact_must_name_the_actual_before_count() -> None:
    before: dict[str, Any] = {
        "revision": 3,
        "history_count": 2,
        "metrics": {"planning_steps": 1, "tool_calls": 0, "usage": dict(_USAGE)},
        "state": {"kind": "planning"},
    }
    after: dict[str, Any] = {
        "revision": 4,
        "history_count": 1,
        "metrics": before["metrics"],
        "state": {"kind": "planning"},
    }
    fact: dict[str, Any] = {
        "kind": "history_rewrite",
        "at": 4.0,
        "data": {
            "before_count": 3,
            "after_roles": ["user"],
            "reason": "compact",
            "metadata_keys": [],
        },
    }

    with pytest.raises(ValueError, match="rewrite_before_mismatch"):
        verify_change(before, fact, after)
