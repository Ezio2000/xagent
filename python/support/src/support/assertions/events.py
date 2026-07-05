"""Event assertions for controlled runtime scenarios."""

from __future__ import annotations

from collections.abc import Sequence

from kernel import AgentEvent, EventType


def assert_event_types(events: Sequence[AgentEvent], expected: Sequence[EventType]) -> None:
    """Assert that events have exactly the expected event types."""

    actual = [event.type for event in events]
    assert actual == list(expected)


def assert_event_type_order(events: Sequence[AgentEvent], expected: Sequence[EventType]) -> None:
    """Assert that event types appear in order, allowing unrelated events between them."""

    actual = [event.type for event in events]
    cursor = 0
    for event_type in actual:
        if cursor < len(expected) and event_type == expected[cursor]:
            cursor += 1
    assert cursor == len(expected), f"expected ordered event types {list(expected)}, got {actual}"
