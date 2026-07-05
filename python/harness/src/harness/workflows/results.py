"""Result containers for high-level agent harness workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from diagnostics import ReplayResult
from kernel import AgentEvent, AgentResult, AgentStatus, PauseState, RunSnapshot


def _final_text(result: AgentResult) -> str:
    return "".join(part.text or "" for part in result.final_parts if part.type == "text")


@dataclass(slots=True, frozen=True)
class PausedHarnessRun:
    """A paused run with a durable snapshot ready for resume."""

    result: AgentResult
    snapshot: RunSnapshot
    pause: PauseState
    events: tuple[AgentEvent, ...] = ()

    @classmethod
    def from_result(
        cls,
        result: AgentResult,
        *,
        events: tuple[AgentEvent, ...] = (),
    ) -> PausedHarnessRun:
        """Build a paused wrapper or raise if the result is not resumable."""

        if result.status is not AgentStatus.PAUSED:
            raise ValueError(f"expected paused result, got {result.status.value}")
        if result.snapshot is None:
            raise ValueError("paused result is missing a snapshot")
        pause = result.snapshot.state.pause
        if pause is None:
            raise ValueError("paused result snapshot is missing pause metadata")
        return cls(
            result=result,
            snapshot=RunSnapshot.from_dict(result.snapshot.to_dict()),
            pause=PauseState.from_dict(pause.to_dict()),
            events=events,
        )

    @property
    def run_id(self) -> str:
        return self.result.run_id


@dataclass(slots=True, frozen=True)
class TraceHarnessRun:
    """A run result paired with diagnostics replay validation."""

    result: AgentResult
    trace: Mapping[str, Any]
    replay: ReplayResult
    events: tuple[AgentEvent, ...] = ()

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def status(self) -> AgentStatus:
        return self.result.status

    @property
    def final_text(self) -> str:
        return _final_text(self.result)
