"""Immutable durable run aggregate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from jharness.kernel._history import (
    HistoryProof,
    analyze_messages,
    evolve_history,
)
from jharness.kernel._validation import expect_instance, expect_int
from jharness.kernel.context import RunContext
from jharness.kernel.history import RunHistory
from jharness.kernel.messages import Message
from jharness.kernel.state import RunMetrics, RunState


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Complete immutable recovery state at one committed revision."""

    revision: int
    context: RunContext
    history: RunHistory
    metrics: RunMetrics
    state: RunState
    _history_proof: HistoryProof = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if expect_int(self.revision, "snapshot revision") < 0:
            raise ValueError("snapshot revision must be >= 0")
        expect_instance(self.context, RunContext, "snapshot context")
        history = expect_instance(self.history, RunHistory, "snapshot history")
        expect_instance(self.metrics, RunMetrics, "snapshot metrics")
        if not isinstance(cast(object, self.state), RunState):
            raise TypeError("snapshot state must be a RunState")
        object.__setattr__(self, "_history_proof", analyze_messages(history, self.state))

    @property
    def status(self) -> str:
        """Display status derived from the lifecycle variant."""

        return self.state.kind

    def _evolve(
        self,
        *,
        append: tuple[Message, ...],
        replace: RunHistory | None,
        metrics: RunMetrics,
        state: RunState,
    ) -> RunSnapshot:
        """Construct one internally proven snapshot without rescanning old history."""

        history, proof = evolve_history(
            self.history,
            self._history_proof,
            append=append,
            replace=replace,
            state=state,
        )
        next_snapshot = object.__new__(RunSnapshot)
        object.__setattr__(next_snapshot, "revision", self.revision + 1)
        object.__setattr__(next_snapshot, "context", self.context)
        object.__setattr__(next_snapshot, "history", history)
        object.__setattr__(next_snapshot, "metrics", metrics)
        object.__setattr__(next_snapshot, "state", state)
        object.__setattr__(next_snapshot, "_history_proof", proof)
        return next_snapshot

    def _history_digest(self) -> bytes:
        return self._history_proof.digest
