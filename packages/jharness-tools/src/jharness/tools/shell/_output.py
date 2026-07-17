"""Bounded, concurrent subprocess-output capture."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

_READ_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class CapturedOutput:
    """One decoded stream plus its raw-byte accounting."""

    text: str
    bytes: int
    truncated: bool


@dataclass(slots=True)
class BoundedOutput:
    """Keep a bounded head and tail while counting and draining every byte."""

    limit: int
    _head: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _tail: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _total: int = field(default=0, init=False, repr=False)

    def feed(self, chunk: bytes) -> None:
        """Consume one chunk without retaining more than ``limit`` bytes."""

        if not chunk:
            return
        self._total += len(chunk)
        head_limit = (self.limit + 1) // 2
        if len(self._head) < head_limit:
            take = min(head_limit - len(self._head), len(chunk))
            self._head.extend(chunk[:take])
            chunk = chunk[take:]
        if not chunk:
            return
        tail_limit = self.limit - head_limit
        if tail_limit == 0:
            return
        self._tail.extend(chunk)
        overflow = len(self._tail) - tail_limit
        if overflow > 0:
            del self._tail[:overflow]

    def capture(self, *, incomplete: bool = False) -> CapturedOutput:
        """Return an immutable UTF-8 observation of the retained raw bytes."""

        retained = bytes(self._head + self._tail)
        return CapturedOutput(
            retained.decode("utf-8", errors="replace"),
            self._total,
            self._total > self.limit or incomplete,
        )


async def drain_stream(reader: asyncio.StreamReader, output: BoundedOutput) -> CapturedOutput:
    """Continuously drain one pipe so the subprocess cannot block on it."""

    while chunk := await reader.read(_READ_SIZE):
        output.feed(chunk)
    return output.capture()
