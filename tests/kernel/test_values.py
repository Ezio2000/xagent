from __future__ import annotations

import gc
from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from typing import Any, cast
from weakref import ref

import pytest

import jharness.kernel.history as history_module
from jharness.kernel import (
    MAX_JSON_NESTING_DEPTH,
    ApprovalAllow,
    ApprovalDeny,
    ApprovalSuspend,
    ArtifactRef,
    Checkpoint,
    Completed,
    ContentPart,
    ControlFact,
    ConversationInsertFact,
    DurableCommit,
    EphemeralRepository,
    ErrorInfo,
    Failed,
    FailedControl,
    HistoryAppend,
    HistoryReplace,
    HistoryRewriteFact,
    HistoryUnchanged,
    InitialHistory,
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
    PendingToolCalls,
    Planning,
    ResponseFormat,
    RevisionConflict,
    RunContext,
    RunHistory,
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
    checkpoint_digest,
    freeze_json_value,
    thaw_json_value,
)
from jharness.kernel._digest import compose_call_id_digest
from jharness.kernel._engine.change import Change
from jharness.kernel._engine.change import reduce as reduce_change
from jharness.kernel.errors import RepositoryError
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint


def context() -> RunContext:
    return RunContext("run-1", 1.0, metadata={"tenant": "a"})


def history(*messages: Message) -> RunHistory:
    return RunHistory(messages)


def pending_calls(*calls: ToolCall) -> PendingToolCalls:
    return PendingToolCalls(calls)


def durable(checkpoint: Checkpoint, previous: Checkpoint | None = None) -> DurableCommit:
    if previous is None:
        return DurableCommit(checkpoint, None, InitialHistory(checkpoint.snapshot.history))
    base = previous.snapshot.history
    target = checkpoint.snapshot.history
    if len(base) == len(target) and base.digest == target.digest:
        change = HistoryUnchanged(len(base), base.digest)
    elif len(target) > len(base) and target[: len(base)] == tuple(base):
        change = HistoryAppend(len(base), base.digest, target[len(base) :])
    else:
        change = HistoryReplace(len(base), base.digest, target)
    return DurableCommit(checkpoint, previous.id, change)


def started() -> Checkpoint:
    message = Message.user("hello")
    snapshot = RunSnapshot(0, context(), history(message), RunMetrics(), Planning())
    return Checkpoint("cp-0", snapshot, StartedFact(1.0, ("user",)))


def started_with_metadata(metadata: dict[str, object]) -> Checkpoint:
    message = Message.user("hello")
    snapshot = RunSnapshot(
        0,
        RunContext("run-1", 1.0, metadata=metadata),
        history(message),
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
            ToolsPending(pending_calls(call)),
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
    pending_state = ToolsPending(pending_calls(call))
    suspension = Suspension("approval", "policy", metadata={"ticket": 7})
    state = Suspended(pending_state, suspension)
    assert state.resume_to is pending_state
    assert SuspensionSelector(source="policy", metadata={"ticket": 7}).matches(state)
    assert not SuspensionSelector(reason="other").matches(state)
    assert not SuspensionSelector(metadata={"missing": None}).matches(state)
    with pytest.raises(ValueError, match="at least one call"):
        PendingToolCalls(())
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
    messages = history(Message.user("go"), Message.assistant(tool_calls=(call,)))
    snapshot = RunSnapshot(1, context(), messages, RunMetrics(), ToolsPending(pending_calls(call)))
    assert snapshot.status == "tools_pending"
    with pytest.raises(ValueError, match="unresolved"):
        RunSnapshot(1, context(), messages, RunMetrics(), Planning())
    failed = RunSnapshot(
        1,
        context(),
        messages,
        RunMetrics(),
        Failed(ErrorInfo("model", "failed")),
    )
    assert failed.status == "failed"
    completed = RunSnapshot(
        1,
        context(),
        history(Message.user("go"), Message.assistant((ContentPart.text_part("done"),))),
        RunMetrics(),
        Completed((ContentPart.text_part("done"),)),
    )
    assert completed.status == "completed"


def test_run_history_append_shares_skew_digits_and_tail_windows_are_bounded() -> None:
    checkpoint = started()
    initial_history = checkpoint.snapshot.history
    for revision in range(1, 130):
        previous = checkpoint.snapshot.history
        previous_digits = object.__getattribute__(previous, "_digits")
        checkpoint = reduce_change(
            checkpoint.snapshot,
            Change(
                ConversationInsertFact(float(revision + 1), "test"),
                Planning(),
                append=(Message.external(f"message-{revision}"),),
            ),
            checkpoint_id=f"cp-{revision}",
        )
        digits = object.__getattribute__(checkpoint.snapshot.history, "_digits")
        second = previous_digits.next
        if second is not None and previous_digits.weight == second.weight:
            assert digits.weight == previous_digits.weight * 2 + 1
            assert digits.next is second.next
            assert digits.tree.left is previous_digits.tree
            assert digits.tree.right is second.tree
        else:
            assert digits.weight == 1
            assert digits.next is previous_digits
            assert digits.tree.message is checkpoint.snapshot.history[-1]

    history_value = checkpoint.snapshot.history
    assert history_value.first is initial_history.first
    assert [message.parts[0].text for message in history_value.iter_tail(3)] == [
        "message-127",
        "message-128",
        "message-129",
    ]
    assert tuple(history_value.iter_window(1, 3)) == history_value[1:3]
    assert tuple(reversed(history_value)) == tuple(history_value)[::-1]
    assert history_value.index(history_value[-1]) == len(history_value) - 1
    assert history_value.count(history_value.first) == 1

    linear = tuple(history_value)
    assert all(history_value[index] is message for index, message in enumerate(linear))

    weights: list[int] = []
    digit = object.__getattribute__(history_value, "_digits")
    while digit is not None:
        weights.append(digit.weight)
        digit = digit.next
    assert sum(weights) == len(history_value)
    assert weights == sorted(weights)
    assert [index for index in range(len(weights) - 1) if weights[index] == weights[index + 1]] in (
        [],
        [0],
    )


def test_run_history_tail_never_locates_a_window_or_visits_older_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = tuple(Message.external(f"message-{index}") for index in range(1_000))
    history_value = RunHistory(messages)
    head = object.__getattribute__(history_value, "_digits")
    assert head.weight == 3
    assert head.next is not None
    older_tree = head.next.tree

    original_iter_tree = history_module._iter_tree  # pyright: ignore[reportPrivateUsage]

    def guarded_iter_tree(tree: Any) -> Iterator[Message]:
        if tree is older_tree:
            raise AssertionError("tail iteration visited an older digit")
        yield from original_iter_tree(tree)

    def fail_window(_: RunHistory, _start: int, _stop: int) -> Iterator[Message]:
        raise AssertionError("tail iteration used general window location")

    monkeypatch.setattr(history_module, "_iter_tree", guarded_iter_tree)
    monkeypatch.setattr(RunHistory, "_iter_window", fail_window)
    assert tuple(history_value.iter_tail(3)) == messages[-3:]
    assert history_value[-1] is messages[-1]


def test_pending_tool_calls_advance_shares_backing_and_composes_suffix_digest() -> None:
    calls = tuple(ToolCall(f"call-{index}", "lookup") for index in range(100))
    pending = PendingToolCalls(calls)
    advanced = pending.advance(37)
    assert advanced is not None
    assert object.__getattribute__(advanced, "_calls") is object.__getattribute__(pending, "_calls")
    assert object.__getattribute__(advanced, "_digests") is object.__getattribute__(
        pending, "_digests"
    )
    assert advanced.prefix(3) == calls[37:40]
    window = advanced.limit(3)
    assert len(window) == 3
    assert tuple(window) == calls[37:40]
    assert window[::-1] == tuple(reversed(calls[37:40]))
    assert object.__getattribute__(window, "_pending") is advanced
    assert advanced[::-1] == tuple(reversed(calls[37:]))
    assert (
        compose_call_id_digest(
            tuple(call.id for call in calls[:37]),
            advanced.call_id_digest,
        )
        == pending.call_id_digest
    )
    assert advanced.advance(len(advanced)) is None


def test_recovered_pending_proof_canonicalizes_once_then_advances_without_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_calls = tuple(ToolCall(f"call-{index}", "lookup") for index in range(40))
    state_calls = tuple(ToolCall(call.id, call.name) for call in history_calls)
    pending = PendingToolCalls(state_calls)
    state = ToolsPending(pending)

    original_call_equality = ToolCall.__eq__
    comparisons = 0

    def counted_call_equality(left: ToolCall, right: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original_call_equality(left, right)

    monkeypatch.setattr(ToolCall, "__eq__", counted_call_equality)
    snapshot = RunSnapshot(
        1,
        context(),
        history(Message.user("go"), Message.assistant(tool_calls=history_calls)),
        RunMetrics(),
        state,
    )
    assert comparisons == len(history_calls)
    proof = object.__getattribute__(snapshot, "_history_proof")
    assert proof.unresolved is pending

    monkeypatch.setattr(ToolCall, "__eq__", original_call_equality)

    def fail_pending_iteration(_: PendingToolCalls) -> object:
        raise AssertionError("incremental pending validation scanned the remaining suffix")

    monkeypatch.setattr(PendingToolCalls, "__iter__", fail_pending_iteration)
    batch_number = 0
    while isinstance(snapshot.state, ToolsPending):
        current = snapshot.state.pending
        batch = current.prefix(3)
        remaining = current.advance(len(batch))
        next_state = ToolsPending(remaining) if remaining is not None else Planning()
        outcome = ToolSuccess((ContentPart.text_part("ok"),))
        checkpoint = reduce_change(
            snapshot,
            Change(
                ToolBatchFact(
                    float(batch_number + 2),
                    f"batch-{batch_number}",
                    tuple(call.id for call in batch),
                    False,
                    (ToolOutcomeKind.SUCCESS,) * len(batch),
                    None,
                ),
                next_state,
                append=tuple(Message.tool(call.id, outcome) for call in batch),
                tool_calls=len(batch),
            ),
            checkpoint_id=f"cp-batch-{batch_number}",
        )
        snapshot = checkpoint.snapshot
        if isinstance(next_state, ToolsPending):
            proof = object.__getattribute__(snapshot, "_history_proof")
            assert proof.unresolved is next_state.pending
        batch_number += 1

    assert batch_number == 14


def test_durable_commit_derives_digest_and_validates_minimal_history_change() -> None:
    first = started()
    initial = durable(first)
    assert initial.digest == checkpoint_digest(first)
    assert initial.history_count == 1
    assert initial.history_digest == first.snapshot.history.digest
    assert initial.base_history_count is None

    second = reduce_change(
        first.snapshot,
        Change(
            ConversationInsertFact(2.0, "test"),
            Planning(),
            append=(Message.external("new"),),
        ),
        checkpoint_id="cp-1",
    )
    appended = durable(second, first)
    assert isinstance(appended.history, HistoryAppend)
    assert appended.base_history_count == 1
    assert appended.base_history_digest == first.snapshot.history.digest

    with pytest.raises(ValueError, match="does not produce"):
        DurableCommit(
            second,
            first.id,
            HistoryAppend(1, b"\x00" * 32, (Message.external("new"),)),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        HistoryAppend(1, first.snapshot.history.digest, ())


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
                ToolsPending(pending_calls(reused)),
                append=(Message.assistant(tool_calls=(reused,)),),
                planning_steps=1,
            ),
            checkpoint_id="cp-3",
        )

    pending_proof = object.__getattribute__(pending.snapshot, "_history_proof")
    settled_proof = object.__getattribute__(settled.snapshot, "_history_proof")
    assert settled.snapshot.history[:-1] == tuple(pending.snapshot.history)
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
    replacement = history(Message.user("condensed"))
    rewritten = reduce_change(
        first.snapshot,
        Change(
            HistoryRewriteFact(2.0, 1, ("user",), "compact", ()),
            Planning(),
            replace=replacement,
        ),
        checkpoint_id="cp-1",
    )

    assert rewritten.snapshot.history is replacement


async def test_ephemeral_repository_enforces_cas_and_all_checkpoint_idempotency() -> None:
    first = started()
    repository = EphemeralRepository()
    await repository.commit(durable(first))
    await repository.commit(durable(first))
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
    await repository.commit(durable(second, first))

    # A lost response can cause an exact retry after the repository has advanced.
    await repository.commit(durable(first))
    ledger = cast(dict[tuple[str, str], bytes], object.__getattribute__(repository, "_by_id"))
    assert set(ledger) == {("run-1", "cp-0"), ("run-1", "cp-1")}
    assert all(len(fingerprint) == 32 for fingerprint in ledger.values())

    collision = Checkpoint(first.id, second.snapshot, second.fact)
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(durable(collision, first))
    stale = Checkpoint("cp-stale", second.snapshot, second.fact)
    with pytest.raises(RevisionConflict):
        await repository.commit(durable(stale, first))


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
    await repository.commit(durable(first))
    await repository.commit(durable(reordered))

    changed_type = started_with_metadata(
        {
            "values": [None, False, 1.0, 1.5, "1"],
            "z": {"first": 1, "second": 2},
        }
    )
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(durable(changed_type))


async def test_ephemeral_repository_accepts_equivalent_wire_reconstruction() -> None:
    first, pending, settled = tool_checkpoints()
    repository = EphemeralRepository()
    await repository.commit(durable(first))
    await repository.commit(durable(pending, first))
    await repository.commit(durable(settled, pending))

    rebuilt = decode_checkpoint(encode_checkpoint(settled))
    assert rebuilt == settled
    assert rebuilt is not settled
    await repository.commit(durable(rebuilt, pending))


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
        history(Message.user("different")),
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
    await repository.commit(durable(first))

    for collision in collisions:
        with pytest.raises(RepositoryError, match="reused"):
            await repository.commit(durable(collision))


async def test_ephemeral_repository_ledger_does_not_retain_old_checkpoint() -> None:
    class WeakCheckpoint(Checkpoint):
        pass

    base = started()
    first = WeakCheckpoint(base.id, base.snapshot, base.fact)
    old_checkpoint = ref(first)
    repository = EphemeralRepository()
    await repository.commit(durable(first))
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
    await repository.commit(durable(second, first))

    del first
    gc.collect()

    ledger = cast(dict[tuple[str, str], bytes], object.__getattribute__(repository, "_by_id"))
    assert old_checkpoint() is None
    assert all(
        type(fingerprint) is bytes and len(fingerprint) == 32 for fingerprint in ledger.values()
    )


async def test_ephemeral_repository_seeds_idempotency_ledger_from_initial_checkpoint() -> None:
    first = started()
    repository = EphemeralRepository(first)
    await repository.commit(durable(first))
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
        await repository.commit(durable(collision, first))


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
        attempted = Checkpoint(
            "cp-1",
            limited_snapshot,
            ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
        )
        await EphemeralRepository().commit(durable(attempted, first))


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


def test_json_values_have_one_portable_nesting_limit() -> None:
    accepted: object = None
    for _ in range(MAX_JSON_NESTING_DEPTH):
        accepted = [accepted]
    assert thaw_json_value(freeze_json_value(accepted)) == accepted

    rejected: object = [accepted]
    with pytest.raises(ValueError, match="maximum JSON nesting depth"):
        freeze_json_value(rejected)
