from __future__ import annotations

import gc
from dataclasses import FrozenInstanceError
from typing import cast
from weakref import ref

import pytest

from jharness.kernel import (
    ApprovalAllow,
    ApprovalDeny,
    ApprovalSuspend,
    ArtifactRef,
    Checkpoint,
    Completed,
    ContentPart,
    ControlFact,
    EphemeralRepository,
    ErrorInfo,
    Failed,
    FailedControl,
    HistoryRewriteFact,
    Limited,
    LimitedControl,
    LimitReason,
    Message,
    ModelCapabilities,
    ModelOptions,
    ModelResponse,
    ModelTurnFact,
    ModelTurnResult,
    ModelUsage,
    Planning,
    ResponseFormat,
    RevisionConflict,
    RunContext,
    RunMetrics,
    RunSnapshot,
    SettledResult,
    StartedFact,
    Suspended,
    SuspendedControl,
    Suspension,
    SuspensionSelector,
    TaskRef,
    ToolAccepted,
    ToolBatchFact,
    ToolCall,
    ToolExecution,
    ToolFailure,
    ToolOutcomeKind,
    ToolRisk,
    ToolSpec,
    ToolsPending,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
    freeze_json_value,
    thaw_json_value,
)
from jharness.kernel._engine.change import Change
from jharness.kernel._engine.change import reduce as reduce_change
from jharness.kernel.errors import RepositoryError
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint


def context() -> RunContext:
    return RunContext("run-1", 1.0, metadata={"tenant": "a"})


def started() -> Checkpoint:
    message = Message.user("hello")
    snapshot = RunSnapshot(0, context(), (message,), RunMetrics(), Planning())
    return Checkpoint("cp-0", snapshot, StartedFact(1.0, ("user",)))


def started_with_metadata(metadata: dict[str, object]) -> Checkpoint:
    message = Message.user("hello")
    snapshot = RunSnapshot(
        0,
        RunContext("run-1", 1.0, metadata=metadata),
        (message,),
        RunMetrics(),
        Planning(),
    )
    return Checkpoint("cp-metadata", snapshot, StartedFact(1.0, ("user",)))


def tool_checkpoints() -> tuple[Checkpoint, Checkpoint, Checkpoint]:
    first = started()
    call = ToolCall("call-1", "lookup", {"key": "value"})
    pending = reduce_change(
        first.snapshot,
        Change(
            ModelTurnFact(
                2.0,
                ModelTurnResult.TOOLS_PENDING,
                0,
                (call.id,),
                None,
                None,
                None,
            ),
            ToolsPending((call,)),
            append=(Message.assistant(tool_calls=(call,)),),
            planning_steps=1,
        ),
        checkpoint_id="cp-1",
    )
    outcome = ToolSuccess((ContentPart.text_part("ok"),), {"count": 1})
    settled = reduce_change(
        pending.snapshot,
        Change(
            ToolBatchFact(
                3.0,
                "batch-1",
                (call.id,),
                False,
                (ToolOutcomeKind.SUCCESS,),
                None,
            ),
            Planning(),
            append=(Message.tool(call.id, outcome),),
            tool_calls=1,
        ),
        checkpoint_id="cp-2",
    )
    return first, pending, settled


def test_message_content_and_tool_result_have_one_authoritative_shape() -> None:
    artifact = ArtifactRef("artifact:1", media_type="text/plain")
    assert ContentPart.artifact_part(artifact).artifact is artifact
    call = ToolCall("call-1", "search", {"q": "x"})
    assistant = Message.assistant(tool_calls=(call,))
    success = ToolSuccess((ContentPart.text_part("ok"),), {"count": 1})
    result = SettledResult(success)
    tool = Message.tool(call.id, result.outcome)
    assert assistant.tool_calls == (call,)
    assert tool.parts == ()
    assert tool.outcome is success
    assert thaw_json_value(success.structured_content) == {"count": 1}


def test_closed_tool_results_and_approval_decisions() -> None:
    failure = ToolFailure.from_error("bad_input", "bad")
    accepted = ToolAccepted(
        (ContentPart.text_part("queued"),),
        "job-1",
        TaskRef("job-1", "queued"),
    )
    suspension = Suspension("waiting", "tool", wait_id="wait-1")
    waiting = WaitingResult(ToolWaiting((ContentPart.text_part("waiting"),)), suspension)
    failed_result = SettledResult(failure)
    accepted_result = SettledResult(accepted)
    assert isinstance(failed_result.outcome, ToolFailure)
    assert isinstance(accepted_result.outcome, ToolAccepted)
    assert failed_result.outcome.error.code == "bad_input"
    assert accepted_result.outcome.correlation_id == "job-1"
    assert waiting.suspension is suspension
    assert ApprovalAllow("c").call_id == "c"
    assert ApprovalDeny("c", "risk").reason == "risk"
    assert ApprovalSuspend("c", suspension).suspension is suspension


def test_flat_state_and_selector_invariants() -> None:
    call = ToolCall("c", "tool")
    pending = ToolsPending((call,))
    suspension = Suspension("approval", "policy", metadata={"ticket": 7})
    state = Suspended(pending, suspension)
    assert state.resume_to is pending
    assert SuspensionSelector(source="policy", metadata={"ticket": 7}).matches(state)
    assert not SuspensionSelector(reason="other").matches(state)
    assert not SuspensionSelector(metadata={"missing": None}).matches(state)
    with pytest.raises(ValueError, match="at least one call"):
        ToolsPending(())
    with pytest.raises(ValueError, match="at least one field"):
        SuspensionSelector()


def test_model_tool_and_limit_values_are_strict() -> None:
    usage = ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5)
    assert ModelUsage(total_tokens=1).add(usage).total_tokens == 6
    response = ModelResponse((ContentPart.text_part("done"),), usage=usage)
    assert response.to_assistant_message().role == "assistant"
    assert ModelCapabilities(streaming=True).streaming
    assert ModelOptions(max_output_tokens=1).max_output_tokens == 1
    assert ResponseFormat("json_schema", {"type": "object"}, True).strict
    spec = ToolSpec(
        "lookup",
        "Lookup",
        {"type": "object"},
        execution=ToolExecution("parallel", read_only=True, idempotent=True),
        risk=ToolRisk(network="read"),
    )
    assert spec.parallel_safe
    with pytest.raises(ValueError, match="read-only"):
        ToolExecution("parallel")
    with pytest.raises(TypeError):
        ModelUsage(total_tokens=True)  # type: ignore[arg-type]


def test_snapshot_validates_history_against_flat_state() -> None:
    call = ToolCall("c", "tool")
    history = (Message.user("go"), Message.assistant(tool_calls=(call,)))
    snapshot = RunSnapshot(1, context(), history, RunMetrics(), ToolsPending((call,)))
    assert snapshot.status == "tools_pending"
    with pytest.raises(ValueError, match="unresolved"):
        RunSnapshot(1, context(), history, RunMetrics(), Planning())
    failed = RunSnapshot(
        1,
        context(),
        history,
        RunMetrics(),
        Failed(ErrorInfo("model", "failed")),
    )
    assert failed.status == "failed"
    completed = RunSnapshot(
        1,
        context(),
        (Message.user("go"), Message.assistant((ContentPart.text_part("done"),))),
        RunMetrics(),
        Completed((ContentPart.text_part("done"),)),
    )
    assert completed.status == "completed"


def test_change_without_history_edit_reuses_tuple_and_proof_without_full_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = started()
    proof = object.__getattribute__(first.snapshot, "_history_proof")

    def reject_full_analysis(*args: object) -> None:
        _ = args
        raise AssertionError("internal no-op change must not analyze full history")

    monkeypatch.setattr("jharness.kernel._history.analyze_history", reject_full_analysis)
    next_checkpoint = reduce_change(
        first.snapshot,
        Change(
            ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
            Limited(LimitReason.DEADLINE),
        ),
        checkpoint_id="cp-1",
    )

    assert next_checkpoint.snapshot.history is first.snapshot.history
    assert object.__getattribute__(next_checkpoint.snapshot, "_history_proof") is proof


def test_change_append_uses_incremental_proof_and_rejects_old_tool_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending, settled = tool_checkpoints()

    def reject_full_analysis(*args: object) -> None:
        _ = args
        raise AssertionError("internal append must not analyze full history")

    monkeypatch.setattr("jharness.kernel._history.analyze_history", reject_full_analysis)
    reused = ToolCall("call-1", "lookup", {"key": "other"})
    with pytest.raises(ValueError, match="tool call id reused"):
        reduce_change(
            settled.snapshot,
            Change(
                ModelTurnFact(
                    4.0,
                    ModelTurnResult.TOOLS_PENDING,
                    0,
                    (reused.id,),
                    None,
                    None,
                    None,
                ),
                ToolsPending((reused,)),
                append=(Message.assistant(tool_calls=(reused,)),),
                planning_steps=1,
            ),
            checkpoint_id="cp-3",
        )

    pending_proof = object.__getattribute__(pending.snapshot, "_history_proof")
    settled_proof = object.__getattribute__(settled.snapshot, "_history_proof")
    assert settled.snapshot.history[:-1] == pending.snapshot.history
    assert settled_proof.seen_call_ids is pending_proof.seen_call_ids


def test_change_replacement_rebuilds_proof_without_external_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jharness.kernel._history as history_internals

    first = started()

    def reject_external_analysis(*args: object) -> None:
        _ = args
        raise AssertionError("trusted replacement must not normalize history again")

    monkeypatch.setattr(history_internals, "analyze_history", reject_external_analysis)
    replacement = (Message.user("condensed"),)
    rewritten = reduce_change(
        first.snapshot,
        Change(
            HistoryRewriteFact(2.0, 1, ("user",), "compact", ()),
            Planning(),
            replace=replacement,
        ),
        checkpoint_id="cp-1",
    )

    assert rewritten.snapshot.history == replacement


async def test_ephemeral_repository_enforces_cas_and_all_checkpoint_idempotency() -> None:
    first = started()
    repository = EphemeralRepository()
    await repository.commit(first)
    await repository.commit(first)
    limited_snapshot = RunSnapshot(
        1,
        first.snapshot.context,
        first.snapshot.history,
        first.snapshot.metrics,
        Limited(LimitReason.DEADLINE),
    )
    second = Checkpoint(
        "cp-1",
        limited_snapshot,
        ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
    )
    await repository.commit(second)

    # A lost response can cause an exact retry after the repository has advanced.
    await repository.commit(first)
    ledger = cast(dict[str, bytes], object.__getattribute__(repository, "_by_id"))
    assert set(ledger) == {"cp-0", "cp-1"}
    assert all(len(fingerprint) == 32 for fingerprint in ledger.values())

    collision = Checkpoint(first.id, second.snapshot, second.fact)
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(collision)
    stale = Checkpoint("cp-stale", second.snapshot, second.fact)
    with pytest.raises(RevisionConflict):
        await repository.commit(stale)


async def test_ephemeral_repository_fingerprint_is_order_independent_and_type_strict() -> None:
    first = started_with_metadata(
        {
            "z": {"second": 2, "first": 1},
            "values": [None, False, 1, 1.5, "1"],
        }
    )
    reordered = started_with_metadata(
        {
            "values": [None, False, 1, 1.5, "1"],
            "z": {"first": 1, "second": 2},
        }
    )
    repository = EphemeralRepository()
    await repository.commit(first)
    await repository.commit(reordered)

    changed_type = started_with_metadata(
        {
            "values": [None, False, 1.0, 1.5, "1"],
            "z": {"first": 1, "second": 2},
        }
    )
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(changed_type)


async def test_ephemeral_repository_accepts_equivalent_wire_reconstruction() -> None:
    first, pending, settled = tool_checkpoints()
    repository = EphemeralRepository()
    await repository.commit(first)
    await repository.commit(pending)
    await repository.commit(settled)

    rebuilt = decode_checkpoint(encode_checkpoint(settled))
    assert rebuilt == settled
    assert rebuilt is not settled
    await repository.commit(rebuilt)


async def test_ephemeral_repository_rejects_same_id_for_each_changed_content_area() -> None:
    first = started()
    changed_context = RunSnapshot(
        0,
        RunContext("run-1", 1.0, metadata={"tenant": "other"}),
        first.snapshot.history,
        first.snapshot.metrics,
        Planning(),
    )
    changed_history = RunSnapshot(
        0,
        first.snapshot.context,
        (Message.user("different"),),
        first.snapshot.metrics,
        Planning(),
    )
    changed_metrics = RunSnapshot(
        0,
        first.snapshot.context,
        first.snapshot.history,
        RunMetrics(planning_steps=1),
        Planning(),
    )
    collisions = (
        Checkpoint(first.id, changed_context, first.fact),
        Checkpoint(first.id, changed_history, first.fact),
        Checkpoint(first.id, changed_metrics, first.fact),
        Checkpoint(first.id, first.snapshot, StartedFact(2.0, ("user",))),
    )
    repository = EphemeralRepository()
    await repository.commit(first)

    for collision in collisions:
        with pytest.raises(RepositoryError, match="reused"):
            await repository.commit(collision)


async def test_ephemeral_repository_ledger_does_not_retain_old_checkpoint() -> None:
    class WeakCheckpoint(Checkpoint):
        pass

    base = started()
    first = WeakCheckpoint(base.id, base.snapshot, base.fact)
    old_checkpoint = ref(first)
    repository = EphemeralRepository()
    await repository.commit(first)
    second = Checkpoint(
        "cp-1",
        RunSnapshot(
            1,
            first.snapshot.context,
            first.snapshot.history,
            first.snapshot.metrics,
            Limited(LimitReason.DEADLINE),
        ),
        ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
    )
    await repository.commit(second)

    del first
    gc.collect()

    ledger = cast(dict[str, bytes], object.__getattribute__(repository, "_by_id"))
    assert old_checkpoint() is None
    assert all(
        type(fingerprint) is bytes and len(fingerprint) == 32 for fingerprint in ledger.values()
    )


async def test_ephemeral_repository_seeds_idempotency_ledger_from_initial_checkpoint() -> None:
    first = started()
    repository = EphemeralRepository(first)
    await repository.commit(first)
    collision = Checkpoint(
        first.id,
        RunSnapshot(
            1,
            first.snapshot.context,
            first.snapshot.history,
            first.snapshot.metrics,
            Limited(LimitReason.DEADLINE),
        ),
        ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
    )
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(collision)


async def test_ephemeral_repository_rejects_nonconsecutive_first_commit() -> None:
    first = started()
    limited_snapshot = RunSnapshot(
        1,
        first.snapshot.context,
        first.snapshot.history,
        first.snapshot.metrics,
        Limited(LimitReason.DEADLINE),
    )
    with pytest.raises(RevisionConflict):
        await EphemeralRepository().commit(
            Checkpoint(
                "cp-1",
                limited_snapshot,
                ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
            )
        )


def test_checkpoint_rejects_fact_state_mismatch() -> None:
    first = started()
    failed_snapshot = RunSnapshot(
        1,
        first.snapshot.context,
        first.snapshot.history,
        first.snapshot.metrics,
        Failed(ErrorInfo("boom", "broken")),
    )
    with pytest.raises(ValueError, match="match Failed"):
        Checkpoint(
            "cp-1",
            failed_snapshot,
            ControlFact(2.0, FailedControl("other")),
        )
    suspended = Suspended(Planning(), Suspension("pause", "host"))
    suspended_snapshot = RunSnapshot(
        1,
        first.snapshot.context,
        first.snapshot.history,
        first.snapshot.metrics,
        suspended,
    )
    checkpoint = Checkpoint(
        "cp-1",
        suspended_snapshot,
        ControlFact(2.0, SuspendedControl("pause", "host", None, ())),
    )
    assert checkpoint.snapshot.state is suspended


def test_json_and_public_values_are_deeply_immutable() -> None:
    raw = {"nested": [1, {"x": 2}]}
    frozen = freeze_json_value(raw)
    raw["nested"] = []
    assert thaw_json_value(frozen) == {"nested": [1, {"x": 2}]}
    with pytest.raises(TypeError):
        frozen["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        context().run_id = "other"  # type: ignore[misc]
