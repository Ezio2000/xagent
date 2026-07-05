"""Conformance case constants and primitive validation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

CASE_KEYS = {
    "name",
    "case_type",
    "limits",
    "pause_request",
    "pause_request_timing",
    "conversation_insert",
    "conversation_insert_timing",
    "model_steps",
    "resume_model_steps",
    "resume_append_messages",
    "resume_expected_pause",
    "resume_checkpoint_status",
    "resume_checkpoint_total_tool_calls",
    "retry_model_errors",
    "approval_decisions",
    "approval_metadata",
    "expected_approval_requests",
    "runtime_context",
    "stream_model_steps",
    "expected_status",
    "expected_final_text",
    "expected_final_part_types",
    "expected_final_parts",
    "expected_tool_calls",
    "expected_resume_status",
    "expected_resume_final_text",
    "expected_resume_tool_calls",
    "expected_resume_message_roles",
    "expected_resume_tool_texts",
    "expected_resume_error",
    "expected_resume_trace_prefix",
    "expected_message_roles",
    "expected_tool_texts",
    "expected_tool_text_contains",
    "expected_pending_tool_call_ids",
    "expected_pause",
    "expected_tool_progress",
    "expected_child_run",
    "expected_model_deltas",
    "expected_event_types",
    "expected_trace_kinds",
    "forbidden_event_types",
    "forbidden_journal_event_types",
    "forbidden_checkpoint_statuses",
    "forbidden_checkpoint_tool_counts",
    "forbidden_checkpoint_status_tool_counts",
    "forbidden_unpaused_checkpoint_tool_counts",
    "forbidden_checkpoint_message_roles",
    "message",
    "model_response",
    "expected_error",
}
NEGATIVE_CASE_TYPES = {"message_negative", "model_response_negative"}
MODEL_STEP_RESPONSE_KEYS = {
    "parts",
    "tool_calls",
    "finish_reason",
    "usage",
    "model",
    "response_id",
    "metadata",
}
MODEL_STEP_KEYS = {
    "error",
    "parts",
    "tool_calls",
    "finish_reason",
    "usage",
    "model",
    "response_id",
    "metadata",
}
STREAM_STEP_KEYS = {"events"}
STREAM_EVENT_KEYS_BY_TYPE: dict[str, set[str]] = {
    "text_delta": {"type", "index", "text_delta", "part_type", "metadata"},
    "reasoning_delta": {"type", "index", "text_delta", "metadata"},
    "tool_call_delta": {
        "type",
        "index",
        "id",
        "name",
        "arguments_delta",
        "mode",
        "metadata",
    },
    "usage_delta": {"type", "usage", "metadata"},
    "sleep": {"type", "seconds"},
    "pause_request": {"type"},
}
LIMIT_KEYS = {
    "max_iterations",
    "max_total_tool_calls",
    "timeout_seconds",
    "stop_on_tool_error",
    "max_parallel_tool_calls",
    "max_total_tokens",
    "max_model_retries",
}
APPROVAL_DECISION_KEYS = {"action", "reason", "metadata"}
APPROVAL_REQUEST_EXPECTATION_KEYS = {"risk", "metadata"}
STREAM_EVENT_REQUIRED_KEYS: dict[str, set[str]] = {
    "text_delta": {"index", "text_delta", "part_type"},
    "reasoning_delta": {"index", "text_delta"},
    "tool_call_delta": {"index"},
    "usage_delta": {"usage"},
    "sleep": {"seconds"},
    "pause_request": set(),
}


def reject_unknown_keys(keys: set[str], allowed: set[str], label: str) -> None:
    unknown = keys - allowed
    if unknown:
        raise AssertionError(f"{label} has unknown key(s): {', '.join(sorted(unknown))}")


def check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_case_list(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return cast(list[dict[str, Any]], value)


def expect_case_list_of_strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    items = cast(list[object], value)
    return [expect_case_str(item, f"{label} item") for item in items]


def expect_case_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def expect_case_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be an array")
    return cast(Sequence[object], value)


def expect_case_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def expect_case_optional_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    return expect_case_str(value, label)


def expect_case_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value


def expect_case_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return expect_case_int(value, label)


def expect_case_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    return float(value)
