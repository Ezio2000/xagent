"""Fake run journals for controlled runtime tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from kernel import EventTypes, JournalRecord

from harness.observation.timeline import timeline_event_label


class MemoryRunJournal:
    """In-memory run journal that records journal record copies."""

    def __init__(self) -> None:
        self.records: list[JournalRecord] = []

    async def append(self, record: JournalRecord) -> None:
        self.records.append(JournalRecord.from_dict(record.to_dict()))

    async def read(
        self, run_id: str, *, after_sequence: int | None = None
    ) -> AsyncIterator[JournalRecord]:
        for record in self.records:
            if record.run_id != run_id:
                continue
            if after_sequence is not None and record.sequence <= after_sequence:
                continue
            yield JournalRecord.from_dict(record.to_dict())


class TimelineRunJournal(MemoryRunJournal):
    """Run journal that also appends event labels to a shared timeline."""

    def __init__(self, timeline: list[str]) -> None:
        super().__init__()
        self.timeline = timeline

    async def append(self, record: JournalRecord) -> None:
        self.timeline.append(timeline_event_label("journal", record.event))
        await super().append(record)


class FailingCheckpointJournal(MemoryRunJournal):
    """Run journal that fails when appending checkpoint events."""

    async def append(self, record: JournalRecord) -> None:
        if record.event_type == EventTypes.CHECKPOINT:
            raise RuntimeError("journal unavailable")
        await super().append(record)


class SlowRunJournal(MemoryRunJournal):
    """Run journal that blocks long enough for runtime deadline tests."""

    async def append(self, record: JournalRecord) -> None:
        _ = record
        await asyncio.sleep(10)
