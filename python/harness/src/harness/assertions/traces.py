"""Trace assertions for controlled runtime tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from diagnostics import RunTrace, replay_trace


def assert_trace_replays(trace_payload: Mapping[str, Any]) -> RunTrace:
    """Assert that a trace payload can be constructed and replayed."""

    trace = RunTrace.from_dict(trace_payload)
    replay = replay_trace(trace)
    assert replay.valid, replay.message
    return trace
