"""Portable start, continue, and resume request documents."""

from __future__ import annotations

from typing import Any

from jharness.kernel.commands import (
    ContinueRequest,
    ResumeRequest,
    RunRequest,
    StartRequest,
    SuspensionSelector,
)
from jharness.kernel.errors import ProtocolError
from jharness.kernel.history import RunHistory
from jharness.kernel.wire._helpers import (
    array,
    decode_document,
    enum_string,
    json_object,
    object_fields,
    optional_string,
    thaw_object,
)
from jharness.kernel.wire.checkpoint import decode_checkpoint_value, encode_checkpoint
from jharness.kernel.wire.messages import decode_message_value, encode_message
from jharness.kernel.wire.snapshot import decode_context_value, encode_context

__all__ = [
    "ContinueRequest",
    "ResumeRequest",
    "RunRequest",
    "StartRequest",
    "SuspensionSelector",
    "decode_run_request",
    "encode_run_request",
]


def encode_run_request(value: RunRequest) -> dict[str, Any]:
    if isinstance(value, StartRequest):
        return {
            "kind": "start",
            "messages": [encode_message(message) for message in value.messages],
            "context": None if value.context is None else encode_context(value.context),
        }
    if isinstance(value, ContinueRequest):
        return {"kind": "continue", "checkpoint": encode_checkpoint(value.checkpoint)}
    return {
        "kind": "resume",
        "checkpoint": encode_checkpoint(value.checkpoint),
        "selector": None if value.selector is None else _encode_selector(value.selector),
        "append_messages": [encode_message(message) for message in value.append_messages],
        "metadata": thaw_object(value.metadata),
    }


def decode_run_request(value: object) -> RunRequest:
    return decode_document(value, "run request", _decode_run_request)


def _decode_run_request(value: object) -> RunRequest:
    raw = json_object(value, "run request")
    if "kind" not in raw:
        raise ProtocolError("run request is missing field(s): kind")
    kind = enum_string(raw["kind"], "run request kind", frozenset({"start", "continue", "resume"}))
    if kind == "start":
        data = object_fields(raw, "start request", frozenset({"kind", "messages", "context"}))
        messages = tuple(
            decode_message_value(item) for item in array(data["messages"], "start messages")
        )
        if not messages:
            raise ProtocolError("start messages must not be empty")
        context = data["context"]
        return StartRequest(
            RunHistory(messages),
            None if context is None else decode_context_value(context),
        )
    if kind == "continue":
        data = object_fields(raw, "continue request", frozenset({"kind", "checkpoint"}))
        return ContinueRequest(decode_checkpoint_value(data["checkpoint"]))
    data = object_fields(
        raw,
        "resume request",
        frozenset({"kind", "checkpoint", "selector", "append_messages", "metadata"}),
    )
    selector = data["selector"]
    return ResumeRequest(
        checkpoint=decode_checkpoint_value(data["checkpoint"]),
        selector=None if selector is None else _decode_selector(selector),
        append_messages=tuple(
            decode_message_value(item) for item in array(data["append_messages"], "append_messages")
        ),
        metadata=json_object(data["metadata"], "resume metadata"),
    )


def _encode_selector(value: SuspensionSelector) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: item
        for key, item in (
            ("reason", value.reason),
            ("source", value.source),
            ("wait_id", value.wait_id),
        )
        if item is not None
    }
    if value.metadata:
        result["metadata"] = thaw_object(value.metadata)
    return result


def _decode_selector(value: object) -> SuspensionSelector:
    data = object_fields(
        value,
        "suspension selector",
        frozenset(),
        frozenset({"reason", "source", "wait_id", "metadata"}),
    )
    metadata = json_object(data.get("metadata", {}), "selector metadata")
    if not any(key in data for key in ("reason", "source", "wait_id")) and not metadata:
        raise ProtocolError("suspension selector must set at least one field")
    return SuspensionSelector(
        reason=optional_string(data.get("reason"), "selector reason", non_empty=True),
        source=optional_string(data.get("source"), "selector source", non_empty=True),
        wait_id=optional_string(data.get("wait_id"), "selector wait_id", non_empty=True),
        metadata=metadata,
    )
