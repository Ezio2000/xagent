"""Explicit codecs for run context and snapshots."""

from __future__ import annotations

from typing import Any

from jharness.kernel.context import RunContext
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.wire._helpers import (
    array,
    decode_document,
    integer,
    json_object,
    number,
    object_fields,
    optional_number,
    optional_string,
    string,
    thaw_object,
)
from jharness.kernel.wire.messages import decode_message_value, encode_message
from jharness.kernel.wire.state import (
    decode_metrics_value,
    decode_state_value,
    encode_metrics,
    encode_state,
)

__all__ = ["decode_context", "decode_snapshot", "encode_context", "encode_snapshot"]


def encode_context(value: RunContext) -> dict[str, Any]:
    return {
        "run_id": value.run_id,
        "started_at": value.started_at,
        "deadline": value.deadline,
        "parent_run_id": value.parent_run_id,
        "parent_tool_call_id": value.parent_tool_call_id,
        "run_kind": value.run_kind,
        "metadata": thaw_object(value.metadata),
    }


def decode_context(value: object) -> RunContext:
    return decode_document(value, "run context", decode_context_value)


def decode_context_value(value: object) -> RunContext:
    data = object_fields(
        value,
        "run context",
        frozenset(
            {
                "run_id",
                "started_at",
                "deadline",
                "parent_run_id",
                "parent_tool_call_id",
                "run_kind",
                "metadata",
            }
        ),
    )
    return RunContext(
        run_id=string(data["run_id"], "run_id", non_empty=True),
        started_at=number(data["started_at"], "started_at", minimum=0),
        deadline=optional_number(data["deadline"], "deadline", minimum=0),
        parent_run_id=optional_string(data["parent_run_id"], "parent_run_id"),
        parent_tool_call_id=optional_string(
            data["parent_tool_call_id"],
            "parent_tool_call_id",
        ),
        run_kind=optional_string(data["run_kind"], "run_kind"),
        metadata=json_object(data["metadata"], "run metadata"),
    )


def encode_snapshot(value: RunSnapshot) -> dict[str, Any]:
    return {
        "revision": value.revision,
        "context": encode_context(value.context),
        "history": [encode_message(message) for message in value.history],
        "metrics": encode_metrics(value.metrics),
        "state": encode_state(value.state),
    }


def decode_snapshot(value: object) -> RunSnapshot:
    return decode_document(value, "run snapshot", decode_snapshot_value)


def decode_snapshot_value(value: object) -> RunSnapshot:
    data = object_fields(
        value,
        "run snapshot",
        frozenset({"revision", "context", "history", "metrics", "state"}),
    )
    history = tuple(decode_message_value(item) for item in array(data["history"], "history"))
    if not history:
        from jharness.kernel.errors import ProtocolError

        raise ProtocolError("history must not be empty")
    return RunSnapshot(
        revision=integer(data["revision"], "snapshot revision", minimum=0),
        context=decode_context_value(data["context"]),
        history=history,
        metrics=decode_metrics_value(data["metrics"]),
        state=decode_state_value(data["state"]),
    )
