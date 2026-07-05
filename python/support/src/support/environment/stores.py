"""Run store implementations for controlled runtime scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from kernel import CheckpointSummary, RunSnapshot, StoredCheckpoint


class MemoryRunStore:
    """In-memory run store that records checkpoint copies."""

    def __init__(self) -> None:
        self.checkpoints: list[StoredCheckpoint] = []

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        self.checkpoints.append(StoredCheckpoint.from_dict(checkpoint.to_dict()))

    async def load_checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> RunSnapshot:
        matches = [checkpoint for checkpoint in self.checkpoints if checkpoint.run_id == run_id]
        if checkpoint_id is not None:
            matches = [
                checkpoint for checkpoint in matches if checkpoint.checkpoint_id == checkpoint_id
            ]
        if not matches:
            raise KeyError(run_id)
        return RunSnapshot.from_dict(matches[-1].snapshot.to_dict())

    async def list_checkpoints(self, run_id: str) -> Sequence[CheckpointSummary]:
        return [
            checkpoint.summary() for checkpoint in self.checkpoints if checkpoint.run_id == run_id
        ]


class FailingRunStore(MemoryRunStore):
    """Run store that fails on every checkpoint save."""

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        _ = checkpoint
        raise RuntimeError("store unavailable")


class FailingSecondCheckpointStore(MemoryRunStore):
    """Run store that accepts the first checkpoint and fails subsequent saves."""

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        if self.checkpoints:
            raise RuntimeError("store unavailable")
        await super().save_checkpoint(checkpoint)


class SlowRunStore(MemoryRunStore):
    """Run store that blocks long enough for runtime deadline tests."""

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        _ = checkpoint
        await asyncio.sleep(10)
