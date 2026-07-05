"""Checkpoint assertions for controlled runtime tests."""

from __future__ import annotations

from collections.abc import Sequence

from kernel import AgentEvent, AgentStatus, EventTypes, RunSnapshot


def checkpoint_statuses(events: Sequence[AgentEvent]) -> list[AgentStatus]:
    """Return checkpoint snapshot statuses in event order."""

    statuses: list[AgentStatus] = []
    for event in events:
        if event.type != EventTypes.CHECKPOINT:
            continue
        statuses.append(RunSnapshot.from_dict(event.data).state.status)
    return statuses


def assert_checkpoint_statuses(
    events: Sequence[AgentEvent], expected: Sequence[AgentStatus]
) -> None:
    """Assert checkpoint statuses in event order."""

    assert checkpoint_statuses(events) == list(expected)
