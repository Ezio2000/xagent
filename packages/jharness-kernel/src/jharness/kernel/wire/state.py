"""Explicit codecs for run state, suspension, and metrics."""

from __future__ import annotations

from typing import Any

from jharness.kernel.errors import ProtocolError
from jharness.kernel.limits import LimitReason
from jharness.kernel.state import (
    ActiveState,
    Completed,
    Failed,
    Limited,
    Planning,
    RunMetrics,
    RunState,
    Suspended,
    Suspension,
    ToolsPending,
)
from jharness.kernel.wire._helpers import (
    array,
    decode_document,
    enum_string,
    integer,
    json_object,
    object_fields,
    optional_string,
    string,
    thaw_object,
)
from jharness.kernel.wire.messages import (
    decode_content_part_value,
    decode_error_info_value,
    decode_tool_call_value,
    encode_content_part,
    encode_error_info,
    encode_tool_call,
)
from jharness.kernel.wire.models import decode_model_usage_value, encode_model_usage

__all__ = [
    "decode_metrics",
    "decode_state",
    "decode_suspension",
    "encode_metrics",
    "encode_state",
    "encode_suspension",
]

_STATE_KINDS = frozenset(
    {"planning", "tools_pending", "suspended", "completed", "failed", "limited"}
)
_LIMIT_REASONS = frozenset(item.value for item in LimitReason)


def encode_suspension(value: Suspension) -> dict[str, Any]:
    return {
        "reason": value.reason,
        "source": value.source,
        "wait_id": value.wait_id,
        "metadata": thaw_object(value.metadata),
    }


def decode_suspension(value: object) -> Suspension:
    return decode_document(value, "suspension", decode_suspension_value)


def decode_suspension_value(value: object) -> Suspension:
    data = object_fields(
        value,
        "suspension",
        frozenset({"reason", "source", "wait_id", "metadata"}),
    )
    return Suspension(
        string(data["reason"], "suspension reason", non_empty=True),
        string(data["source"], "suspension source", non_empty=True),
        optional_string(data["wait_id"], "suspension wait_id", non_empty=True),
        json_object(data["metadata"], "suspension metadata"),
    )


def encode_state(value: RunState) -> dict[str, Any]:
    if isinstance(value, Planning):
        return {"kind": "planning"}
    if isinstance(value, ToolsPending):
        return {"kind": "tools_pending", "pending": [encode_tool_call(x) for x in value.pending]}
    if isinstance(value, Suspended):
        return {
            "kind": "suspended",
            "resume_to": encode_state(value.resume_to),
            "suspension": encode_suspension(value.suspension),
        }
    if isinstance(value, Completed):
        return {"kind": "completed", "parts": [encode_content_part(x) for x in value.parts]}
    if isinstance(value, Failed):
        return {"kind": "failed", "error": encode_error_info(value.error)}
    return {"kind": "limited", "reason": value.reason.value}


def decode_state(value: object) -> RunState:
    return decode_document(value, "run state", decode_state_value)


def decode_state_value(value: object) -> RunState:
    raw = json_object(value, "run state")
    if "kind" not in raw:
        raise ProtocolError("run state is missing field(s): kind")
    kind = enum_string(raw["kind"], "run state kind", _STATE_KINDS)
    if kind == "planning":
        object_fields(raw, "planning state", frozenset({"kind"}))
        return Planning()
    if kind == "tools_pending":
        data = object_fields(raw, "tools_pending state", frozenset({"kind", "pending"}))
        pending = tuple(decode_tool_call_value(x) for x in array(data["pending"], "pending calls"))
        return ToolsPending(pending)
    if kind == "suspended":
        data = object_fields(
            raw,
            "suspended state",
            frozenset({"kind", "resume_to", "suspension"}),
        )
        return Suspended(
            _decode_active_state(data["resume_to"]),
            decode_suspension_value(data["suspension"]),
        )
    if kind == "completed":
        data = object_fields(raw, "completed state", frozenset({"kind", "parts"}))
        parts = tuple(decode_content_part_value(x) for x in array(data["parts"], "completed parts"))
        return Completed(parts)
    if kind == "failed":
        data = object_fields(raw, "failed state", frozenset({"kind", "error"}))
        return Failed(decode_error_info_value(data["error"]))
    data = object_fields(raw, "limited state", frozenset({"kind", "reason"}))
    reason = enum_string(data["reason"], "limit reason", _LIMIT_REASONS)
    return Limited(LimitReason(reason))


def _decode_active_state(value: object) -> ActiveState:
    state = decode_state_value(value)
    if not isinstance(state, Planning | ToolsPending):
        raise ProtocolError("active state must be planning or tools_pending")
    return state


def encode_metrics(value: RunMetrics) -> dict[str, Any]:
    return {
        "planning_steps": value.planning_steps,
        "tool_calls": value.tool_calls,
        "usage": encode_model_usage(value.usage),
    }


def decode_metrics(value: object) -> RunMetrics:
    return decode_document(value, "run metrics", decode_metrics_value)


def decode_metrics_value(value: object) -> RunMetrics:
    data = object_fields(
        value,
        "run metrics",
        frozenset({"planning_steps", "tool_calls", "usage"}),
    )
    return RunMetrics(
        integer(data["planning_steps"], "planning_steps", minimum=0),
        integer(data["tool_calls"], "tool_calls", minimum=0),
        decode_model_usage_value(data["usage"]),
    )
