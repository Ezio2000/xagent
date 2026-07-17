"""Explicit codecs for tool specifications and invocation results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from jharness.kernel.errors import ProtocolError
from jharness.kernel.tools import (
    SettledResult,
    ToolAccepted,
    ToolExecution,
    ToolFailure,
    ToolRisk,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.kernel.wire._helpers import (
    boolean,
    decode_document,
    enum_string,
    json_object,
    object_fields,
    string,
    thaw_object,
)
from jharness.kernel.wire.messages import decode_tool_outcome_value, encode_tool_outcome
from jharness.kernel.wire.state import decode_suspension_value, encode_suspension

__all__ = [
    "decode_tool_result",
    "decode_tool_spec",
    "encode_tool_result",
    "encode_tool_spec",
]

_RISK_FIELDS = frozenset(
    {"filesystem", "network", "subprocess", "destructive", "requires_approval"}
)


def encode_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    """Encode one portable tool specification."""

    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": _encode_schema(spec.input_schema),
        "output_schema": (
            None if spec.output_schema is None else _encode_schema(spec.output_schema)
        ),
        "execution": {
            "concurrency": spec.execution.concurrency,
            "read_only": spec.execution.read_only,
            "idempotent": spec.execution.idempotent,
        },
        "risk": encode_risk_value(spec.risk),
    }


def decode_tool_spec(value: object) -> ToolSpec:
    """Decode one portable tool specification."""

    return decode_document(value, "tool spec", decode_tool_spec_value)


def decode_tool_spec_value(value: object) -> ToolSpec:
    fields = object_fields(
        value,
        "tool spec",
        {"name", "description", "input_schema", "output_schema", "execution", "risk"},
    )
    raw_output = fields["output_schema"]
    return ToolSpec(
        name=string(fields["name"], "tool name", non_empty=True),
        description=string(fields["description"], "tool description"),
        input_schema=_decode_schema(fields["input_schema"], "input_schema"),
        output_schema=(None if raw_output is None else _decode_schema(raw_output, "output_schema")),
        execution=_decode_execution(fields["execution"]),
        risk=decode_risk_value(fields["risk"]),
    )


def _encode_schema(value: Mapping[str, Any] | bool) -> Mapping[str, Any] | bool:
    return value if isinstance(value, bool) else thaw_object(value)


def _decode_schema(value: object, label: str) -> Mapping[str, Any] | bool:
    return value if isinstance(value, bool) else json_object(value, label)


def _decode_execution(value: object) -> ToolExecution:
    fields = object_fields(
        value,
        "tool execution",
        {"concurrency", "read_only", "idempotent"},
    )
    return ToolExecution(
        concurrency=cast(
            Literal["serial", "parallel"],
            enum_string(fields["concurrency"], "tool concurrency", {"serial", "parallel"}),
        ),
        read_only=boolean(fields["read_only"], "tool read_only"),
        idempotent=boolean(fields["idempotent"], "tool idempotent"),
    )


def encode_risk_value(risk: ToolRisk) -> dict[str, Any]:
    encoded = thaw_object(risk.extra)
    for key, value in (
        ("filesystem", risk.filesystem),
        ("network", risk.network),
        ("subprocess", risk.subprocess),
        ("destructive", risk.destructive),
        ("requires_approval", risk.requires_approval),
    ):
        if value is not None:
            encoded[key] = value
    return encoded


def decode_risk_value(value: object) -> ToolRisk:
    mapping = json_object(value, "tool risk")
    return ToolRisk(
        filesystem=_present_risk_string(mapping, "filesystem"),
        network=_present_risk_string(mapping, "network"),
        subprocess=_present_risk_bool(mapping, "subprocess"),
        destructive=_present_risk_bool(mapping, "destructive"),
        requires_approval=_present_risk_bool(mapping, "requires_approval"),
        extra={key: item for key, item in mapping.items() if key not in _RISK_FIELDS},
    )


def _present_risk_string(mapping: Mapping[str, Any], key: str) -> str | None:
    if key not in mapping:
        return None
    return string(mapping[key], f"risk {key}", non_empty=True)


def _present_risk_bool(mapping: Mapping[str, Any], key: str) -> bool | None:
    if key not in mapping:
        return None
    return boolean(mapping[key], f"risk {key}")


def encode_tool_result(result: SettledResult | WaitingResult) -> dict[str, Any]:
    """Encode one settled or waiting tool result."""

    encoded = {"outcome": encode_tool_outcome(result.outcome)}
    if isinstance(result, WaitingResult):
        encoded["suspension"] = encode_suspension(result.suspension)
    return encoded


def decode_tool_result(value: object) -> SettledResult | WaitingResult:
    """Decode one settled or waiting tool result."""

    return decode_document(value, "tool result", decode_tool_result_value)


def decode_tool_result_value(value: object) -> SettledResult | WaitingResult:
    mapping = json_object(value, "tool result")
    if "suspension" in mapping:
        fields = object_fields(mapping, "waiting result", {"outcome", "suspension"})
        outcome = decode_tool_outcome_value(fields["outcome"])
        if not isinstance(outcome, ToolWaiting):
            raise ProtocolError("waiting result requires a waiting outcome")
        return WaitingResult(outcome, decode_suspension_value(fields["suspension"]))
    fields = object_fields(mapping, "settled result", {"outcome"})
    outcome = decode_tool_outcome_value(fields["outcome"])
    if not isinstance(outcome, ToolSuccess | ToolFailure | ToolAccepted):
        raise ProtocolError("settled result requires a settled outcome")
    return SettledResult(outcome)
