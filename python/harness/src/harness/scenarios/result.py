"""Scenario result containers for controlled runtime scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from kernel import AgentEvent, AgentResult


@dataclass(slots=True, frozen=True)
class HarnessRunResult:
    """Observed output from a controlled harness run."""

    result: AgentResult | None = None
    events: tuple[AgentEvent, ...] = ()

    @property
    def run_id(self) -> str | None:
        if self.result is not None:
            return self.result.run_id
        if self.events:
            return self.events[0].run_id
        return None
