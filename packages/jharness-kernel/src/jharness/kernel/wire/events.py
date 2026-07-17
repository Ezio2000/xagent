"""Explicit codec for invocation events and trace entries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jharness.kernel.events import Event, EventKind
from jharness.kernel.wire._helpers import (
    boolean,
    decode_document,
    enum_string,
    integer,
    json_object,
    number,
    object_fields,
    optional_string,
    string,
    thaw_object,
)
from jharness.kernel.wire.checkpoint import (
    decode_fact_value,
    decode_run_view_value,
    encode_fact,
    encode_run_view,
)
from jharness.kernel.wire.messages import decode_tool_call_value, encode_tool_call
from jharness.kernel.wire.models import decode_model_usage_value, encode_model_usage
from jharness.kernel.wire.state import decode_suspension_value, encode_suspension
from jharness.kernel.wire.tools import decode_risk_value, encode_risk_value

__all__ = ["decode_event", "encode_event"]

_REQUEST_KINDS = frozenset({"start", "continue", "resume"})
_DELTA_KINDS = frozenset({"content", "tool_call", "reasoning", "usage"})
_OUTCOME_KINDS = frozenset({"success", "failure", "accepted", "waiting"})
_STOP_REASONS = frozenset(
    {"terminal", "suspended", "cancelled", "consumer_closed", "repository_error"}
)


def encode_event(value: Event) -> dict[str, Any]:
    return {
        "schema_version": "v0",
        "run_id": value.run_id,
        "invocation_id": value.invocation_id,
        "sequence": value.sequence,
        "kind": value.kind.value,
        "created_at": value.created_at,
        "data": decode_event_data_value(value.kind, thaw_object(value.data)),
    }


def decode_event(value: object) -> Event:
    return decode_document(value, "invocation event", _decode_event)


def _decode_event(value: object) -> Event:
    data = object_fields(
        value,
        "invocation event",
        frozenset(
            {"schema_version", "run_id", "invocation_id", "sequence", "kind", "created_at", "data"}
        ),
    )
    if string(data["schema_version"], "event schema_version") != "v0":
        from jharness.kernel.errors import ProtocolError

        raise ProtocolError("event schema_version must be v0")
    kind = EventKind(enum_string(data["kind"], "event kind", frozenset(x.value for x in EventKind)))
    return Event(
        run_id=string(data["run_id"], "event run_id", non_empty=True),
        invocation_id=string(data["invocation_id"], "event invocation_id", non_empty=True),
        sequence=integer(data["sequence"], "event sequence", minimum=1),
        kind=kind,
        created_at=number(data["created_at"], "event created_at", minimum=0),
        data=decode_event_data_value(kind, data["data"]),
    )


EventDataDecoder = Callable[[object], dict[str, Any]]

_DATA_DECODERS: dict[EventKind, EventDataDecoder] = {
    EventKind.INVOCATION_STARTED: lambda value: _decode_invocation_started(value),
    EventKind.MODEL_STARTED: lambda value: _decode_model_started(value),
    EventKind.MODEL_DELTA: lambda value: _decode_model_delta(value),
    EventKind.MODEL_FINISHED: lambda value: _decode_model_finished(value),
    EventKind.APPROVAL_REQUESTED: lambda value: _decode_approval_requested(value),
    EventKind.APPROVAL_DECIDED: lambda value: _decode_approval_decided(value),
    EventKind.TOOL_STARTED: lambda value: _decode_tool_started(value),
    EventKind.TOOL_PROGRESS: lambda value: _decode_tool_progress(value),
    EventKind.TOOL_FINISHED: lambda value: _decode_tool_finished(value),
    EventKind.TOOL_CANCEL_REQUESTED: lambda value: _decode_tool_cancel(value),
    EventKind.CHECKPOINT_COMMITTED: lambda value: _decode_checkpoint_committed(value),
    EventKind.INVOCATION_STOPPED: lambda value: _decode_invocation_stopped(value),
}


def decode_event_data_value(kind: EventKind, value: object) -> dict[str, Any]:
    return _DATA_DECODERS[kind](value)


def _decode_invocation_started(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "invocation_started data",
        frozenset({"request_kind", "starting_checkpoint_id", "starting"}),
    )
    request_kind = enum_string(data["request_kind"], "request kind", _REQUEST_KINDS)
    checkpoint_id = optional_string(
        data["starting_checkpoint_id"],
        "starting checkpoint id",
        non_empty=True,
    )
    raw_starting = data["starting"]
    if request_kind == "start":
        if checkpoint_id is not None or raw_starting is not None:
            from jharness.kernel.errors import ProtocolError

            raise ProtocolError("start event cannot carry a starting checkpoint")
        starting = None
    else:
        if checkpoint_id is None or raw_starting is None:
            from jharness.kernel.errors import ProtocolError

            raise ProtocolError("continue and resume events require a starting checkpoint")
        starting = decode_run_view_value(raw_starting)
    return {
        "request_kind": request_kind,
        "starting_checkpoint_id": checkpoint_id,
        "starting": starting,
    }


def _decode_model_started(value: object) -> dict[str, Any]:
    data = object_fields(value, "model_started data", frozenset({"planning_step"}))
    return {"planning_step": integer(data["planning_step"], "planning_step", minimum=1)}


def _decode_model_delta(value: object) -> dict[str, Any]:
    raw = json_object(value, "model_delta data")
    if "kind" not in raw:
        from jharness.kernel.errors import ProtocolError

        raise ProtocolError("model_delta data is missing field(s): kind")
    kind = enum_string(raw["kind"], "model delta kind", _DELTA_KINDS)
    if kind == "content":
        data = object_fields(
            raw,
            "content delta",
            frozenset({"kind", "index", "part_type", "text_delta", "data"}),
        )
        return {
            "kind": "content",
            "index": integer(data["index"], "content delta index", minimum=0),
            "part_type": string(data["part_type"], "content part_type", non_empty=True),
            "text_delta": string(data["text_delta"], "content text_delta"),
            "data": thaw_object(json_object(data["data"], "content delta data")),
        }
    if kind == "tool_call":
        data = object_fields(
            raw,
            "tool call delta",
            frozenset({"kind", "index", "id", "name", "arguments_delta"}),
        )
        return {
            "kind": "tool_call",
            "index": integer(data["index"], "tool call delta index", minimum=0),
            "id": optional_string(data["id"], "tool call delta id", non_empty=True),
            "name": optional_string(data["name"], "tool call delta name", non_empty=True),
            "arguments_delta": string(data["arguments_delta"], "arguments_delta"),
        }
    if kind == "reasoning":
        data = object_fields(raw, "reasoning delta", frozenset({"kind", "index", "text_delta"}))
        return {
            "kind": "reasoning",
            "index": integer(data["index"], "reasoning delta index", minimum=0),
            "text_delta": string(data["text_delta"], "reasoning text_delta"),
        }
    data = object_fields(raw, "usage delta", frozenset({"kind", "usage"}))
    return {
        "kind": "usage",
        "usage": encode_model_usage(decode_model_usage_value(data["usage"])),
    }


def _decode_model_finished(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "model_finished data",
        frozenset({"finish_reason", "tool_call_count", "usage"}),
    )
    raw_usage = data["usage"]
    return {
        "finish_reason": optional_string(data["finish_reason"], "finish_reason"),
        "tool_call_count": integer(data["tool_call_count"], "tool_call_count", minimum=0),
        "usage": (
            None if raw_usage is None else encode_model_usage(decode_model_usage_value(raw_usage))
        ),
    }


def _decode_approval_requested(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "approval request",
        frozenset({"batch_id", "index", "call", "risk"}),
    )
    return {
        "batch_id": string(data["batch_id"], "approval batch_id", non_empty=True),
        "index": integer(data["index"], "approval index", minimum=0),
        "call": encode_tool_call(decode_tool_call_value(data["call"])),
        "risk": encode_risk_value(decode_risk_value(data["risk"])),
    }


def _decode_approval_decided(value: object) -> dict[str, Any]:
    raw = json_object(value, "approval decision")
    if "kind" not in raw:
        from jharness.kernel.errors import ProtocolError

        raise ProtocolError("approval decision is missing field(s): kind")
    kind = enum_string(
        raw["kind"], "approval decision kind", frozenset({"allow", "deny", "suspend"})
    )
    if kind == "allow":
        data = object_fields(raw, "approval allow", frozenset({"call_id", "kind"}))
        return {"call_id": string(data["call_id"], "call_id", non_empty=True), "kind": "allow"}
    if kind == "deny":
        data = object_fields(raw, "approval deny", frozenset({"call_id", "kind", "reason"}))
        return {
            "call_id": string(data["call_id"], "call_id", non_empty=True),
            "kind": "deny",
            "reason": string(data["reason"], "approval reason", non_empty=True),
        }
    data = object_fields(raw, "approval suspend", frozenset({"call_id", "kind", "suspension"}))
    return {
        "call_id": string(data["call_id"], "call_id", non_empty=True),
        "kind": "suspend",
        "suspension": encode_suspension(decode_suspension_value(data["suspension"])),
    }


def _decode_tool_started(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "tool_started data",
        frozenset({"batch_id", "index", "call", "parallel"}),
    )
    return {
        "batch_id": string(data["batch_id"], "tool batch_id", non_empty=True),
        "index": integer(data["index"], "tool index", minimum=0),
        "call": encode_tool_call(decode_tool_call_value(data["call"])),
        "parallel": boolean(data["parallel"], "tool parallel"),
    }


def _decode_tool_progress(value: object) -> dict[str, Any]:
    data = object_fields(value, "tool_progress data", frozenset({"tool_call_id", "progress"}))
    return {
        "tool_call_id": string(data["tool_call_id"], "tool_call_id", non_empty=True),
        "progress": thaw_object(json_object(data["progress"], "tool progress")),
    }


def _decode_tool_finished(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "tool_finished data",
        frozenset({"batch_id", "index", "tool_call_id", "outcome_kind"}),
    )
    return {
        "batch_id": string(data["batch_id"], "tool batch_id", non_empty=True),
        "index": integer(data["index"], "tool index", minimum=0),
        "tool_call_id": string(data["tool_call_id"], "tool_call_id", non_empty=True),
        "outcome_kind": enum_string(data["outcome_kind"], "tool outcome kind", _OUTCOME_KINDS),
    }


def _decode_tool_cancel(value: object) -> dict[str, Any]:
    data = object_fields(value, "tool_cancel_requested data", frozenset({"tool_call_id"}))
    return {"tool_call_id": string(data["tool_call_id"], "tool_call_id", non_empty=True)}


def _decode_checkpoint_committed(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "checkpoint_committed data",
        frozenset({"checkpoint_id", "fact", "after"}),
    )
    return {
        "checkpoint_id": string(data["checkpoint_id"], "checkpoint_id", non_empty=True),
        "fact": encode_fact(decode_fact_value(data["fact"])),
        "after": encode_run_view(decode_run_view_value(data["after"])),
    }


def _decode_invocation_stopped(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "invocation_stopped data",
        frozenset({"reason", "last_checkpoint_id"}),
    )
    return {
        "reason": enum_string(data["reason"], "invocation stop reason", _STOP_REASONS),
        "last_checkpoint_id": optional_string(
            data["last_checkpoint_id"],
            "last checkpoint id",
            non_empty=True,
        ),
    }
