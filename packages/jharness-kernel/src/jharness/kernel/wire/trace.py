"""Explicit codec for compact diagnostic traces."""

from __future__ import annotations

from typing import Any, cast

from jharness.kernel.diagnostics import RequestKind, RunTrace, TraceEntry, TraceHeader
from jharness.kernel.errors import ProtocolError
from jharness.kernel.events import EventKind
from jharness.kernel.wire._helpers import (
    array,
    decode_document,
    enum_string,
    integer,
    number,
    object_fields,
    string,
    thaw_object,
    unique_strings,
)
from jharness.kernel.wire.events import decode_event_data_value

__all__ = ["decode_trace", "encode_trace"]

_REQUEST_KINDS = frozenset({"start", "continue", "resume"})
_EVENT_KINDS = frozenset(item.value for item in EventKind)


def encode_trace(value: RunTrace) -> dict[str, Any]:
    return {
        "schema_version": "v0",
        "header": {
            "run_id": value.header.run_id,
            "invocation_id": value.header.invocation_id,
            "request_kind": value.header.request_kind,
            "metadata_keys": list(value.header.metadata_keys),
        },
        "entries": [_encode_entry(entry) for entry in value.entries],
    }


def decode_trace(value: object) -> RunTrace:
    return decode_document(value, "run trace", _decode_trace)


def _decode_trace(value: object) -> RunTrace:
    data = object_fields(
        value,
        "run trace",
        frozenset({"schema_version", "header", "entries"}),
    )
    if string(data["schema_version"], "trace schema_version") != "v0":
        raise ProtocolError("trace schema_version must be v0")
    entries = tuple(_decode_entry(item) for item in array(data["entries"], "trace entries"))
    if len(entries) < 2:
        raise ProtocolError("trace requires at least two entries")
    return RunTrace(_decode_header(data["header"]), entries)


def _decode_header(value: object) -> TraceHeader:
    data = object_fields(
        value,
        "trace header",
        frozenset({"run_id", "invocation_id", "request_kind", "metadata_keys"}),
    )
    return TraceHeader(
        run_id=string(data["run_id"], "trace run_id", non_empty=True),
        invocation_id=string(
            data["invocation_id"],
            "trace invocation_id",
            non_empty=True,
        ),
        request_kind=cast(
            RequestKind,
            enum_string(data["request_kind"], "trace request_kind", _REQUEST_KINDS),
        ),
        metadata_keys=unique_strings(data["metadata_keys"], "trace metadata_keys"),
    )


def _encode_entry(value: TraceEntry) -> dict[str, Any]:
    return {
        "sequence": value.sequence,
        "kind": value.kind.value,
        "created_at": value.created_at,
        "data": decode_event_data_value(value.kind, thaw_object(value.data)),
    }


def _decode_entry(value: object) -> TraceEntry:
    data = object_fields(
        value,
        "trace entry",
        frozenset({"sequence", "kind", "created_at", "data"}),
    )
    kind = EventKind(enum_string(data["kind"], "trace event kind", _EVENT_KINDS))
    return TraceEntry(
        sequence=integer(data["sequence"], "trace sequence", minimum=1),
        kind=kind,
        created_at=number(data["created_at"], "trace created_at", minimum=0),
        data=decode_event_data_value(kind, data["data"]),
    )
