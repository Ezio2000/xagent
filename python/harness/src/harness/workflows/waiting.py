"""Helpers for extracting pause and background-work state from harness runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from kernel import AgentEvent, AgentResult, BackgroundTask, EventTypes, PauseState


@dataclass(slots=True, frozen=True)
class WaitingState:
    """Host-visible waiting state surfaced by a run."""

    pause: PauseState | None = None
    background_tasks: tuple[BackgroundTask, ...] = ()

    @property
    def has_waiting_work(self) -> bool:
        return self.pause is not None or bool(self.background_tasks)


def waiting_from_result(
    result: AgentResult,
    *,
    events: Sequence[AgentEvent] = (),
) -> WaitingState:
    """Extract waiting state from a result, optionally enriched by events."""

    pause = None
    if result.snapshot is not None:
        snapshot_pause = result.snapshot.state.pause
        if snapshot_pause is not None:
            pause = PauseState.from_dict(snapshot_pause.to_dict())
    event_state = waiting_from_events(events)
    result_tasks = () if pause is None else _background_tasks_from_pause(pause)
    message_tasks = _background_tasks_from_messages(result)
    return WaitingState(
        pause=pause or event_state.pause,
        background_tasks=_merge_background_tasks(
            result_tasks,
            message_tasks,
            event_state.background_tasks,
        ),
    )


def waiting_from_events(events: Sequence[AgentEvent]) -> WaitingState:
    """Extract waiting state from runtime events."""

    pause: PauseState | None = None
    tasks: list[BackgroundTask] = []
    for event in events:
        if event.type == EventTypes.RUN_PAUSED:
            raw_pause = event.data.get("pause")
            if isinstance(raw_pause, Mapping):
                pause = PauseState.from_dict(cast(Mapping[str, Any], raw_pause))
        if event.type in {
            EventTypes.BACKGROUND_TASK_STARTED,
            EventTypes.BACKGROUND_TASK_UPDATED,
            EventTypes.BACKGROUND_TASK_COMPLETED,
        }:
            raw_task = event.data.get("task")
            if isinstance(raw_task, Mapping):
                tasks.append(BackgroundTask.from_dict(cast(Mapping[str, Any], raw_task)))
    return WaitingState(pause=pause, background_tasks=tuple(tasks))


def _background_tasks_from_pause(pause: PauseState) -> tuple[BackgroundTask, ...]:
    raw_task = pause.metadata.get("background_task")
    if not isinstance(raw_task, Mapping):
        return ()
    return (BackgroundTask.from_dict(cast(Mapping[str, Any], raw_task)),)


def _background_tasks_from_messages(result: AgentResult) -> tuple[BackgroundTask, ...]:
    tasks: list[BackgroundTask] = []
    for message in result.messages:
        if message.role != "tool":
            continue
        raw_task = message.metadata.get("background_task")
        if isinstance(raw_task, Mapping):
            tasks.append(BackgroundTask.from_dict(cast(Mapping[str, Any], raw_task)))
    return tuple(tasks)


def _merge_background_tasks(*groups: Sequence[BackgroundTask]) -> tuple[BackgroundTask, ...]:
    merged: list[BackgroundTask] = []
    seen: set[tuple[str, str, str | None]] = set()
    for group in groups:
        for task in group:
            key = (task.id, task.status, task.lifecycle)
            if key in seen:
                continue
            seen.add(key)
            merged.append(BackgroundTask.from_dict(task.to_dict()))
    return tuple(merged)
