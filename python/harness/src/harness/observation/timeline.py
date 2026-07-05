"""Timeline labeling helpers for controlled runtime scenarios."""

from __future__ import annotations

from kernel import AgentEvent, EventTypes


def timeline_event_label(prefix: str, event: AgentEvent) -> str:
    """Return a stable event label for ordering assertions."""

    if event.type == EventTypes.STATE_CHANGED:
        return f"{prefix}:{event.type}:{event.data['to']}"
    return f"{prefix}:{event.type}"
