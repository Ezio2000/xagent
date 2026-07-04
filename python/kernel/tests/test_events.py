from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest
from kernel import AgentEvent, EventEmitter, EventTypes, QueuedEvent


def test_event_to_dict() -> None:
    event = AgentEvent(
        EventTypes.MODEL_STARTED,
        {"iteration": 1},
        run_id="run-1",
        sequence=7,
        created_at=1.5,
    )

    assert event.to_dict() == {
        "type": "model_started",
        "data": {"iteration": 1},
        "run_id": "run-1",
        "sequence": 7,
        "created_at": 1.5,
        "schema_version": "v0",
    }


def test_custom_event_type_is_allowed() -> None:
    event = AgentEvent("memory_compacted", {"tokens": 120}, run_id="run-1")

    assert event.type == "memory_compacted"


def test_event_constructor_rejects_invalid_envelope_types() -> None:
    with pytest.raises(TypeError, match="event type"):
        AgentEvent(cast(Any, 123))

    with pytest.raises(TypeError, match="event sequence"):
        AgentEvent("custom", sequence=cast(Any, True))

    with pytest.raises(TypeError, match="event created_at"):
        AgentEvent("custom", created_at=cast(Any, "now"))

    with pytest.raises(TypeError, match="event schema_version"):
        AgentEvent("custom", run_id="run-1", schema_version=cast(Any, 1))

    with pytest.raises(ValueError, match="run_id"):
        AgentEvent("custom", run_id="")


def test_event_emitter_rejects_core_event_types() -> None:
    emitter = EventEmitter()

    with pytest.raises(ValueError, match="runtime-owned"):
        emitter.emit(EventTypes.CHECKPOINT, {})


def test_event_emitter_drains_queued_events() -> None:
    emitter = EventEmitter()
    emitter.emit("custom_progress", {"phase": "model"})

    events = emitter.drain()

    assert events == (QueuedEvent("custom_progress", {"phase": "model"}),)
    assert emitter.drain() == ()


def test_event_data_is_defensively_copied() -> None:
    data = {"nested": {"value": 1}}
    event = AgentEvent("custom", data, run_id="run-1")

    data["nested"]["value"] = 2

    assert event.data == {"nested": {"value": 1}}


def test_event_data_is_immutable() -> None:
    event = AgentEvent(
        "custom",
        {"nested": {"value": 1}, "items": [{"value": 1}]},
        run_id="run-1",
    )

    with pytest.raises(TypeError, match="event data is immutable"):
        event.data["nested"]["value"] = 2
    with pytest.raises(TypeError, match="event data is immutable"):
        event.data["items"].append({"value": 2})

    data = event.to_dict()
    data["data"]["nested"]["value"] = 3

    assert event.data["nested"]["value"] == 1


def test_event_data_can_be_deepcopied_by_consumers() -> None:
    event = AgentEvent("custom", {"items": [{"value": 1}]}, run_id="run-1")

    copied = deepcopy(event.data)
    copied["items"].append({"value": 2})

    assert copied == {"items": [{"value": 1}, {"value": 2}]}
    assert event.data == {"items": [{"value": 1}]}
