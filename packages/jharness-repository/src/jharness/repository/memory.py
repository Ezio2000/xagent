"""Process-local, multi-run durable-commit repository."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import cast

from jharness.kernel import (
    Checkpoint,
    DurableCommit,
    RepositoryError,
    RevisionConflict,
)

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _StoredHead:
    checkpoint: Checkpoint
    digest: bytes
    history_count: int
    history_digest: bytes


@dataclass(frozen=True, slots=True)
class _LedgerEntry:
    revision: int
    digest: bytes


class MemoryRunRepository:
    """Thread-safe in-memory CAS repository for any number of runs.

    Values remain domain objects and structurally share immutable histories; this
    backend never serializes a checkpoint merely to retain or return it.
    """

    __slots__ = ("_heads", "_ledger", "_lock")

    def __init__(self) -> None:
        self._heads: dict[str, _StoredHead] = {}
        self._ledger: dict[str, dict[str, _LedgerEntry]] = {}
        self._lock = Lock()

    async def commit(self, commit: DurableCommit) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""

        if not isinstance(cast(object, commit), DurableCommit):
            raise TypeError("commit must be a DurableCommit")
        run_id = _plain_string(commit.run_id, "run id")
        checkpoint_id = _plain_string(commit.checkpoint_id, "checkpoint id")
        parent_id = commit.parent_checkpoint_id
        parent_id = None if parent_id is None else _plain_string(parent_id, "parent checkpoint id")

        with self._lock:
            ledger = self._ledger.get(run_id)
            existing = None if ledger is None else ledger.get(checkpoint_id)
            head = self._heads.get(run_id)
            if existing is not None:
                _accept_existing(head, existing, commit, checkpoint_id, run_id)
                return
            _validate_new_commit(head, commit, run_id, parent_id)
            self._store_new(head, ledger, commit, run_id, checkpoint_id)

    def _store_new(
        self,
        previous_head: _StoredHead | None,
        ledger: dict[str, _LedgerEntry] | None,
        commit: DurableCommit,
        run_id: str,
        checkpoint_id: str,
    ) -> None:
        checkpoint = Checkpoint(
            checkpoint_id,
            commit.checkpoint.snapshot,
            commit.checkpoint.fact,
        )
        head = _StoredHead(
            checkpoint,
            commit.digest,
            commit.history_count,
            commit.history_digest,
        )
        entry = _LedgerEntry(commit.revision, commit.digest)
        created_ledger = ledger is None
        if ledger is None:
            ledger = {}
        previous_entry: _LedgerEntry | object = ledger.get(checkpoint_id, _MISSING)
        try:
            if created_ledger:
                self._ledger[run_id] = ledger
            ledger[checkpoint_id] = entry
            self._heads[run_id] = head
        except BaseException:
            if previous_entry is _MISSING:
                ledger.pop(checkpoint_id, None)
            else:
                ledger[checkpoint_id] = cast(_LedgerEntry, previous_entry)
            if created_ledger:
                self._ledger.pop(run_id, None)
            if previous_head is None:
                self._heads.pop(run_id, None)
            else:
                self._heads[run_id] = previous_head
            raise

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""

        run_id = _plain_string(run_id, "run_id")
        with self._lock:
            head = self._heads.get(run_id)
            if head is None:
                return None
            checkpoint = head.checkpoint
            return Checkpoint(checkpoint.id, checkpoint.snapshot, checkpoint.fact)


def _accept_existing(
    head: _StoredHead | None,
    entry: _LedgerEntry,
    commit: DurableCommit,
    checkpoint_id: str,
    run_id: str,
) -> None:
    if entry.digest != commit.digest:
        raise RepositoryError(
            f"checkpoint id {checkpoint_id!r} was reused with new content in run {run_id!r}"
        )
    if entry.revision != commit.revision:
        raise RepositoryError("stored memory checkpoint ledger is inconsistent")
    if head is None or head.checkpoint.snapshot.revision < entry.revision:
        raise RepositoryError("stored memory checkpoint ledger is orphaned")
    if head.checkpoint.snapshot.revision == entry.revision and (
        head.checkpoint.id != checkpoint_id or head.digest != entry.digest
    ):
        raise RepositoryError("stored memory checkpoint ledger is orphaned")


def _validate_new_commit(
    head: _StoredHead | None,
    commit: DurableCommit,
    run_id: str,
    parent_id: str | None,
) -> None:
    actual_revision = None if head is None else head.checkpoint.snapshot.revision
    if actual_revision != commit.expected_revision:
        raise RevisionConflict(run_id, commit.expected_revision, actual_revision)
    if head is None:
        if parent_id is not None or commit.base_history_count is not None:
            raise RepositoryError("first durable commit has an invalid history base")
        return
    if parent_id != head.checkpoint.id:
        raise RepositoryError("parent checkpoint does not match the authoritative head")
    if (
        commit.base_history_count != head.history_count
        or commit.base_history_digest != head.history_digest
    ):
        raise RepositoryError("history change base does not match the authoritative head")


def _plain_string(value: str, label: str) -> str:
    if not isinstance(cast(object, value), str):
        raise TypeError(f"{label} must be a string")
    normalized = str.__str__(value)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized
