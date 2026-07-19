from __future__ import annotations

import json

import pytest

from jharness.kernel import (
    Checkpoint,
    ControlFact,
    DurableCommit,
    HistoryAppend,
    HistoryUnchanged,
    Message,
    ModelTurnFact,
    ModelTurnResult,
    PendingToolCalls,
    RunHistory,
    RunSnapshot,
    Suspended,
    SuspendedControl,
    Suspension,
    ToolCall,
    ToolsPending,
)
from jharness.repository import _codec
from tests.repository_backends.support import started


def _pending_commit(call_count: int) -> tuple[DurableCommit, DurableCommit]:
    initial = started("run-a", "cp-0")
    calls = tuple(
        ToolCall(f"call-{index}", "tool", {"index": index}) for index in range(call_count)
    )
    assistant = Message.assistant((), tool_calls=calls)
    before = initial.checkpoint.snapshot.history
    history = RunHistory((*before, assistant))
    snapshot = RunSnapshot(
        1,
        initial.checkpoint.snapshot.context,
        history,
        initial.checkpoint.snapshot.metrics,
        ToolsPending(PendingToolCalls(calls)),
    )
    checkpoint = Checkpoint(
        "cp-1",
        snapshot,
        ModelTurnFact(
            at=2.0,
            result=ModelTurnResult.TOOLS_PENDING,
            part_count=0,
            tool_call_ids=tuple(call.id for call in calls),
            finish_reason="tool_calls",
            usage=None,
            limit_reason=None,
        ),
    )
    return initial, DurableCommit(
        checkpoint,
        initial.checkpoint_id,
        HistoryAppend(len(before), before.digest, (assistant,)),
    )


def test_pending_state_core_is_compact_and_reconstructs_from_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial, pending = _pending_commit(256)

    def forbidden(_: object) -> dict[str, object]:
        raise AssertionError("pending core must not call the full state wire encoder")

    monkeypatch.setattr(_codec, "encode_state", forbidden)
    identity = _codec.commit_identity(pending)  # pyright: ignore[reportPrivateUsage]
    core = _codec.encode_core(identity)  # pyright: ignore[reportPrivateUsage]
    document = json.loads(core.payload)
    state = pending.checkpoint.snapshot.state
    assert isinstance(state, ToolsPending)
    assert document["state"] == {
        "kind": "tools_pending",
        "pending_count": 256,
        "pending_digest": state.pending.digest.hex(),
    }
    assert b'"pending":[' not in core.payload

    suspension = Suspension("host_wait", "test")
    suspended_snapshot = RunSnapshot(
        2,
        pending.checkpoint.snapshot.context,
        pending.checkpoint.snapshot.history,
        pending.checkpoint.snapshot.metrics,
        Suspended(state, suspension),
    )
    suspended_checkpoint = Checkpoint(
        "cp-2",
        suspended_snapshot,
        ControlFact(3.0, SuspendedControl("host_wait", "test", None, ())),
    )
    suspended = DurableCommit(
        suspended_checkpoint,
        pending.checkpoint_id,
        HistoryUnchanged(len(suspended_snapshot.history), suspended_snapshot.history.digest),
    )
    suspended_core = _codec.encode_core(  # pyright: ignore[reportPrivateUsage]
        _codec.commit_identity(suspended)  # pyright: ignore[reportPrivateUsage]
    )
    assert b"call-0" not in suspended_core.payload
    assert b"call-255" not in suspended_core.payload

    chunks = (
        *_codec.encode_history_change(initial),  # pyright: ignore[reportPrivateUsage]
        *_codec.encode_history_change(pending),  # pyright: ignore[reportPrivateUsage]
    )
    checkpoint, _ = _codec.reconstruct_checkpoint(  # pyright: ignore[reportPrivateUsage]
        core_payload=suspended_core.payload,
        core_digest=suspended_core.digest,
        chunks=tuple((chunk.payload, chunk.digest, chunk.message_count) for chunk in chunks),
        expected_checkpoint_digest=suspended.digest,
    )
    assert checkpoint == suspended.checkpoint


def test_history_codec_chunks_only_the_appended_delta() -> None:
    initial = started("run-a", "cp-0")
    before = initial.checkpoint.snapshot.history
    messages = tuple(Message.external(str(index)) for index in range(130))
    history = RunHistory((*before, *messages))
    snapshot = RunSnapshot(
        1,
        initial.checkpoint.snapshot.context,
        history,
        initial.checkpoint.snapshot.metrics,
        initial.checkpoint.snapshot.state,
    )
    from jharness.kernel import ConversationInsertFact

    checkpoint = Checkpoint("cp-1", snapshot, ConversationInsertFact(2.0, "test"))
    commit = DurableCommit(
        checkpoint,
        initial.checkpoint_id,
        HistoryAppend(len(before), before.digest, messages),
    )

    chunks = _codec.encode_history_change(commit)  # pyright: ignore[reportPrivateUsage]
    assert tuple(chunk.message_count for chunk in chunks) == (64, 64, 2)
    assert sum(chunk.message_count for chunk in chunks) == len(messages)
