"""Single Checkpoint construction and persistence path."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from jharness.kernel._engine.change import Change, reduce
from jharness.kernel._validation import expect_instance
from jharness.kernel.checkpoint import Checkpoint
from jharness.kernel.errors import CommitError
from jharness.kernel.repository import (
    DurableCommit,
    HistoryAppend,
    HistoryChange,
    HistoryReplace,
    HistoryUnchanged,
    InitialHistory,
    RunRepository,
)


class WorkCommitDeadlineReached(Exception):
    pass


class Committer:
    __slots__ = ("_repository", "_timeout")

    def __init__(self, repository: RunRepository, *, timeout_seconds: float) -> None:
        self._repository = expect_instance(repository, RunRepository, "repository")
        if timeout_seconds <= 0:
            raise ValueError("repository timeout must be > 0")
        self._timeout = timeout_seconds

    async def persist_start(
        self,
        checkpoint: Checkpoint,
        *,
        work_timeout_seconds: float | None = None,
    ) -> Checkpoint:
        checkpoint = expect_instance(checkpoint, Checkpoint, "start checkpoint")
        if checkpoint.snapshot.revision != 0:
            raise ValueError("start checkpoint revision must be 0")
        return await self._persist(
            DurableCommit(
                checkpoint,
                None,
                InitialHistory(checkpoint.snapshot.history),
            ),
            previous=None,
            work_timeout_seconds=work_timeout_seconds,
        )

    async def apply(
        self,
        previous: Checkpoint,
        change: Change,
        *,
        work_timeout_seconds: float | None = None,
    ) -> Checkpoint:
        previous = expect_instance(previous, Checkpoint, "previous checkpoint")
        checkpoint = reduce(previous.snapshot, change, checkpoint_id=str(uuid4()))
        return await self._persist(
            DurableCommit(
                checkpoint,
                previous.id,
                _history_change(previous, change, checkpoint),
            ),
            previous=previous,
            work_timeout_seconds=work_timeout_seconds,
        )

    async def _persist(
        self,
        commit: DurableCommit,
        *,
        previous: Checkpoint | None,
        work_timeout_seconds: float | None = None,
    ) -> Checkpoint:
        work_limited = work_timeout_seconds is not None and work_timeout_seconds <= self._timeout
        timeout = (
            self._timeout
            if work_timeout_seconds is None
            else min(self._timeout, max(0.0, work_timeout_seconds))
        )
        if timeout <= 0:
            raise WorkCommitDeadlineReached
        try:
            async with asyncio.timeout(timeout):
                await self._repository.commit(commit)
        except TimeoutError as exc:
            if work_limited:
                raise WorkCommitDeadlineReached from exc
            raise CommitError("repository commit timed out", last_checkpoint=previous) from exc
        except Exception as exc:
            raise CommitError(str(exc) or exc.__class__.__name__, last_checkpoint=previous) from exc
        return commit.checkpoint


def _history_change(
    previous: Checkpoint,
    change: Change,
    checkpoint: Checkpoint,
) -> HistoryChange:
    base_count = len(previous.snapshot.history)
    base_digest = previous.snapshot._history_digest()  # pyright: ignore[reportPrivateUsage]
    if change.replace is not None:
        return HistoryReplace(base_count, base_digest, checkpoint.snapshot.history)
    if change.append:
        return HistoryAppend(base_count, base_digest, change.append)
    return HistoryUnchanged(base_count, base_digest)
