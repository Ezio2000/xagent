"""Process-local, multi-run checkpoint repository."""

from __future__ import annotations

from threading import Lock
from typing import cast

from jharness.kernel import Checkpoint, RepositoryError, RevisionConflict

from ._codec import EncodedCheckpoint, decode_checkpoint, encode_checkpoint


class MemoryRunRepository:
    """Thread-safe in-memory CAS repository for any number of runs."""

    __slots__ = ("_by_id", "_heads", "_lock")

    def __init__(self) -> None:
        self._heads: dict[str, EncodedCheckpoint] = {}
        self._by_id: dict[str, bytes] = {}
        self._lock = Lock()

    async def commit(self, checkpoint: Checkpoint) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""
        encoded = encode_checkpoint(checkpoint)
        with self._lock:
            existing_digest = self._by_id.get(encoded.checkpoint_id)
            if existing_digest is not None:
                if existing_digest == encoded.digest:
                    return
                raise RepositoryError(
                    f"checkpoint id {encoded.checkpoint_id!r} was reused with new content"
                )

            head = self._heads.get(encoded.run_id)
            actual_revision = None if head is None else head.revision
            if actual_revision != encoded.expected_revision:
                raise RevisionConflict(
                    encoded.run_id,
                    encoded.expected_revision,
                    actual_revision,
                )

            previous_head = self._heads.get(encoded.run_id)
            try:
                self._by_id[encoded.checkpoint_id] = encoded.digest
                self._heads[encoded.run_id] = encoded
            except BaseException:
                self._by_id.pop(encoded.checkpoint_id, None)
                if previous_head is None:
                    self._heads.pop(encoded.run_id, None)
                else:
                    self._heads[encoded.run_id] = previous_head
                raise

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""
        run_id = _validate_run_id(run_id)
        with self._lock:
            head = self._heads.get(run_id)
        return None if head is None else decode_checkpoint(head.payload)


def _validate_run_id(run_id: str) -> str:
    if not isinstance(cast(object, run_id), str):
        raise TypeError("run_id must be a string")
    run_id = str.__str__(run_id)
    if not run_id:
        raise ValueError("run_id must not be empty")
    return run_id
