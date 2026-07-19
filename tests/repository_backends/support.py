from __future__ import annotations

from jharness.kernel import (
    Checkpoint,
    ControlFact,
    ConversationInsertFact,
    DurableCommit,
    HistoryAppend,
    HistoryReplace,
    HistoryRewriteFact,
    HistoryUnchanged,
    InitialHistory,
    Limited,
    LimitedControl,
    LimitReason,
    Message,
    Planning,
    RunContext,
    RunHistory,
    RunMetrics,
    RunSnapshot,
    StartedFact,
)


def started(
    run_id: str,
    checkpoint_id: str,
    *,
    text: str = "hello",
    metadata: dict[str, object] | None = None,
) -> DurableCommit:
    history = RunHistory((Message.user(text),))
    snapshot = RunSnapshot(
        0,
        RunContext(run_id, 1.0, metadata={} if metadata is None else metadata),
        history,
        RunMetrics(),
        Planning(),
    )
    checkpoint = Checkpoint(checkpoint_id, snapshot, StartedFact(1.0, ("user",)))
    return DurableCommit(checkpoint, None, InitialHistory(history))


def append_external(
    previous: Checkpoint,
    checkpoint_id: str,
    *,
    text: str = "external",
) -> DurableCommit:
    message = Message.external(text)
    before = previous.snapshot.history
    history = RunHistory((*before, message))
    snapshot = RunSnapshot(
        previous.snapshot.revision + 1,
        previous.snapshot.context,
        history,
        previous.snapshot.metrics,
        Planning(),
    )
    checkpoint = Checkpoint(
        checkpoint_id,
        snapshot,
        ConversationInsertFact(2.0 + previous.snapshot.revision, "test"),
    )
    return DurableCommit(
        checkpoint,
        previous.id,
        HistoryAppend(len(before), before.digest, (message,)),
    )


def replace_history(
    previous: Checkpoint,
    checkpoint_id: str,
    *,
    messages: tuple[Message, ...] | None = None,
) -> DurableCommit:
    before = previous.snapshot.history
    history = RunHistory((Message.user("summary"),) if messages is None else messages)
    snapshot = RunSnapshot(
        previous.snapshot.revision + 1,
        previous.snapshot.context,
        history,
        previous.snapshot.metrics,
        Planning(),
    )
    checkpoint = Checkpoint(
        checkpoint_id,
        snapshot,
        HistoryRewriteFact(
            at=2.0 + previous.snapshot.revision,
            before_count=len(before),
            after_roles=tuple(message.role for message in history),
            reason="compact",
            metadata_keys=(),
        ),
    )
    return DurableCommit(
        checkpoint,
        previous.id,
        HistoryReplace(len(before), before.digest, history),
    )


def limited(previous: Checkpoint, checkpoint_id: str) -> DurableCommit:
    history = previous.snapshot.history
    snapshot = RunSnapshot(
        previous.snapshot.revision + 1,
        previous.snapshot.context,
        history,
        previous.snapshot.metrics,
        Limited(LimitReason.DEADLINE),
    )
    checkpoint = Checkpoint(
        checkpoint_id,
        snapshot,
        ControlFact(2.0 + previous.snapshot.revision, LimitedControl(LimitReason.DEADLINE)),
    )
    return DurableCommit(
        checkpoint,
        previous.id,
        HistoryUnchanged(len(history), history.digest),
    )
