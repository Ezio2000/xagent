"""Portable structural-plus-semantic validation cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from conformance._schemas import SchemaSuite, SchemaValidationError
from conformance._values import boolean, string
from jharness.kernel.diagnostics import TraceError, verify_trace
from jharness.kernel.wire import (
    decode_checkpoint,
    decode_event,
    decode_message,
    decode_model_response,
    decode_run_request,
    decode_snapshot,
    decode_state,
    decode_tool_result,
    decode_tool_spec,
    decode_trace,
)

_SCHEMAS = {
    "message": "messages.schema.json",
    "model_response": "model-response.schema.json",
    "tool_spec": "tools.schema.json",
    "tool_result": "tool-result.schema.json",
    "state": "state.schema.json",
    "snapshot": "run-snapshot.schema.json",
    "request": "run-request.schema.json",
    "checkpoint": "checkpoint.schema.json",
    "event": "events.schema.json",
    "trace": "run-trace.schema.json",
}


def run_validation_case(case: Mapping[str, Any], schemas: SchemaSuite) -> None:
    target = string(case["target"], "validation target")
    value = case["value"]
    expected_valid = boolean(case["expected_valid"], "expected_valid")
    expected_code = case["expected_error_code"]
    try:
        schemas.validate(_SCHEMAS[target], value)
        _semantic_validator(target)(value)
    except Exception as exc:
        if expected_valid:
            raise AssertionError(f"expected valid {target}, got {exc}") from exc
        code = _validation_error_code(exc)
        if code != expected_code:
            raise AssertionError(
                f"expected validation error {expected_code!r}, got {code!r}: {exc}"
            ) from exc
        return
    if not expected_valid:
        raise AssertionError(f"expected invalid {target}, but validation succeeded")


def _semantic_validator(target: str) -> Callable[[object], object]:
    return {
        "message": decode_message,
        "model_response": decode_model_response,
        "tool_spec": decode_tool_spec,
        "tool_result": decode_tool_result,
        "state": decode_state,
        "snapshot": decode_snapshot,
        "request": decode_run_request,
        "checkpoint": decode_checkpoint,
        "event": decode_event,
        "trace": _trace,
    }[target]


def _trace(value: object) -> object:
    trace = decode_trace(value)
    return verify_trace(trace)


def _validation_error_code(exc: Exception) -> str:
    if isinstance(exc, SchemaValidationError):
        return "schema_validation"
    if isinstance(exc, TraceError):
        return exc.code
    message = str(exc)
    if "messages require a planning" in message or "messages_require_planning" in message:
        return "messages_require_planning"
    if "tool call ids must be unique" in message or (
        "tool call id" in message and "unique" in message
    ):
        return "duplicate_tool_call_id"
    return "schema_validation"
