"""Strict schemas and defensive normalization for the Agent preset tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jharness.kernel import JsonValue
from jharness.tools.agent.models import AgentRequest, AgentSnapshot, AgentStatus

SCHEMA_VERSION = 1
DEFAULT_MAX_AGENT_ID_CHARS = 512
DEFAULT_MAX_DESCRIPTION_CHARS = 200
DEFAULT_MAX_PROMPT_CHARS = 100_000
DEFAULT_MAX_RESULT_CHARS = 65_536
DEFAULT_MAX_ERROR_CODE_CHARS = 128
DEFAULT_MAX_ERROR_MESSAGE_CHARS = 4_096

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_TERMINAL_STATUSES = frozenset[AgentStatus]({"completed", "failed", "cancelled"})


class AgentContractError(ValueError):
    """A recoverable validation error at an Agent tool contract boundary."""


def positive_int(value: object, label: str) -> int:
    """Validate one positive Host-configured integer limit."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def build_wait_id(run_id: str, tool_call_id: str) -> str:
    """Build an unambiguous stable identifier for one Agent wait."""

    run_id = _contract_non_empty_string(run_id, "run_id")
    tool_call_id = _contract_non_empty_string(tool_call_id, "tool_call_id")
    return f"agent-wait:{len(run_id)}:{run_id}:{len(tool_call_id)}:{tool_call_id}"


def agent_input_schema(
    *,
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> dict[str, Any]:
    """Build the strict Draft 2020-12 input schema for ``Agent``."""

    max_description_chars = positive_int(max_description_chars, "max_description_chars")
    max_prompt_chars = positive_int(max_prompt_chars, "max_prompt_chars")
    return {
        "$schema": _DRAFT_2020_12,
        "type": "object",
        "required": ["description", "prompt"],
        "properties": {
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_description_chars,
            },
            "prompt": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_prompt_chars,
            },
            "background": {
                "type": "boolean",
                "default": False,
            },
        },
        "additionalProperties": False,
    }


def agent_id_input_schema(
    *,
    max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
) -> dict[str, Any]:
    """Build the strict shared input schema for Agent id operations."""

    max_agent_id_chars = positive_int(max_agent_id_chars, "max_agent_id_chars")
    return {
        "$schema": _DRAFT_2020_12,
        "type": "object",
        "required": ["agent_id"],
        "properties": {
            "agent_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": max_agent_id_chars,
            }
        },
        "additionalProperties": False,
    }


def snapshot_output_schema(
    *,
    max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
    max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
) -> dict[str, Any]:
    """Build the strict shared Agent snapshot-or-null output schema."""

    max_agent_id_chars = positive_int(max_agent_id_chars, "max_agent_id_chars")
    max_description_chars = positive_int(max_description_chars, "max_description_chars")
    max_result_chars = positive_int(max_result_chars, "max_result_chars")
    max_error_code_chars = positive_int(max_error_code_chars, "max_error_code_chars")
    max_error_message_chars = positive_int(
        max_error_message_chars,
        "max_error_message_chars",
    )
    branches = [
        _snapshot_branch(
            status,
            max_agent_id_chars=max_agent_id_chars,
            max_description_chars=max_description_chars,
            max_result_chars=max_result_chars,
            max_error_code_chars=max_error_code_chars,
            max_error_message_chars=max_error_message_chars,
        )
        for status in ("queued", "running", "completed", "failed", "cancelled")
    ]
    return {
        "$schema": _DRAFT_2020_12,
        "oneOf": [*branches, {"type": "null"}],
    }


def normalize_agent_request(
    arguments: object,
    *,
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> AgentRequest:
    """Validate and normalize one complete ``Agent`` argument object."""

    max_description_chars = _contract_positive_int(
        max_description_chars,
        "max_description_chars",
    )
    max_prompt_chars = _contract_positive_int(max_prompt_chars, "max_prompt_chars")
    root = _contract_mapping(arguments, "Agent arguments")
    _require_exact_keys(
        root,
        "Agent arguments",
        required=frozenset({"description", "prompt"}),
        allowed=frozenset({"description", "prompt", "background"}),
    )
    description = _bounded_non_empty_string(
        root["description"],
        "description",
        max_description_chars,
    )
    prompt = _bounded_non_empty_string(root["prompt"], "prompt", max_prompt_chars)
    background = root.get("background", False)
    if not isinstance(background, bool):
        raise AgentContractError("background must be bool")
    return AgentRequest(description, prompt, background)


def normalize_agent_id(
    arguments: object,
    *,
    max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
) -> str:
    """Validate one complete Agent id operation argument object."""

    max_agent_id_chars = _contract_positive_int(max_agent_id_chars, "max_agent_id_chars")
    root = _contract_mapping(arguments, "Agent id arguments")
    _require_exact_keys(
        root,
        "Agent id arguments",
        required=frozenset({"agent_id"}),
        allowed=frozenset({"agent_id"}),
    )
    return _bounded_non_empty_string(root["agent_id"], "agent_id", max_agent_id_chars)


def snapshot_payload(
    snapshot: object,
    *,
    max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
    max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
    max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
) -> dict[str, JsonValue]:
    """Return a bounded JSON payload for one validated Agent snapshot."""

    max_agent_id_chars = _contract_positive_int(max_agent_id_chars, "max_agent_id_chars")
    max_description_chars = _contract_positive_int(
        max_description_chars,
        "max_description_chars",
    )
    max_result_chars = _contract_positive_int(max_result_chars, "max_result_chars")
    max_error_code_chars = _contract_positive_int(
        max_error_code_chars,
        "max_error_code_chars",
    )
    max_error_message_chars = _contract_positive_int(
        max_error_message_chars,
        "max_error_message_chars",
    )
    snapshot = _validated_snapshot(snapshot)

    agent_id = _bounded_non_empty_string(
        snapshot.agent_id,
        "snapshot.agent_id",
        max_agent_id_chars,
    )
    description = _bounded_non_empty_string(
        snapshot.description,
        "snapshot.description",
        max_description_chars,
    )
    payload: dict[str, JsonValue] = {
        "agent_id": agent_id,
        "status": snapshot.status,
        "description": description,
        "background": snapshot.background,
        "cancellation_requested": snapshot.cancellation_requested,
    }
    if snapshot.status == "completed":
        result = snapshot.result
        if result is None:  # pragma: no cover - revalidated above
            raise AgentContractError("completed snapshot must contain a result")
        payload["result"] = _bounded_string(
            result,
            "snapshot.result",
            max_result_chars,
        )
    elif snapshot.status == "failed":
        error = snapshot.error
        if error is None:  # pragma: no cover - revalidated above
            raise AgentContractError("failed snapshot must contain an error")
        error_payload: dict[str, JsonValue] = {
            "code": _bounded_non_empty_string(
                error.code,
                "snapshot.error.code",
                max_error_code_chars,
            ),
            "message": _bounded_non_empty_string(
                error.message,
                "snapshot.error.message",
                max_error_message_chars,
            ),
        }
        payload["error"] = error_payload
    return payload


def _snapshot_branch(
    status: AgentStatus,
    *,
    max_agent_id_chars: int,
    max_description_chars: int,
    max_result_chars: int,
    max_error_code_chars: int,
    max_error_message_chars: int,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "agent_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": max_agent_id_chars,
        },
        "status": {"const": status},
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": max_description_chars,
        },
        "background": {"type": "boolean"},
        "cancellation_requested": (
            {"const": status == "cancelled"}
            if status in _TERMINAL_STATUSES
            else {"type": "boolean"}
        ),
    }
    required = [
        "agent_id",
        "status",
        "description",
        "background",
        "cancellation_requested",
    ]
    if status == "completed":
        properties["result"] = {
            "type": "string",
            "maxLength": max_result_chars,
        }
        required.append("result")
    elif status == "failed":
        properties["error"] = {
            "type": "object",
            "required": ["code", "message"],
            "properties": {
                "code": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_error_code_chars,
                },
                "message": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": max_error_message_chars,
                },
            },
            "additionalProperties": False,
        }
        required.append("error")
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


def _contract_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AgentContractError(f"{label} must be an object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise AgentContractError(f"{label} keys must be strings")
    return cast(Mapping[str, object], raw)


def _validated_snapshot(value: object) -> AgentSnapshot:
    if not isinstance(value, AgentSnapshot):
        raise AgentContractError("snapshot must be an AgentSnapshot")
    try:
        return AgentSnapshot(
            agent_id=value.agent_id,
            description=value.description,
            status=value.status,
            background=value.background,
            result=value.result,
            error=value.error,
            cancellation_requested=value.cancellation_requested,
        )
    except (TypeError, ValueError) as exc:
        raise AgentContractError("snapshot fields are inconsistent") from exc


def _require_exact_keys(
    value: Mapping[str, object],
    label: str,
    *,
    required: frozenset[str],
    allowed: frozenset[str],
) -> None:
    keys = frozenset(value)
    missing = sorted(required - keys)
    if missing:
        raise AgentContractError(f"{label} is missing required fields: {', '.join(missing)}")
    unknown = sorted(keys - allowed)
    if unknown:
        raise AgentContractError(f"{label} contains unknown fields: {', '.join(unknown)}")


def _contract_positive_int(value: object, label: str) -> int:
    try:
        return positive_int(value, label)
    except ValueError as exc:
        raise AgentContractError(str(exc)) from exc


def _contract_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise AgentContractError(f"{label} must be a string")
    if not value:
        raise AgentContractError(f"{label} must not be empty")
    return value


def _bounded_non_empty_string(value: object, label: str, maximum: int) -> str:
    text = _contract_non_empty_string(value, label)
    return _bounded_string(text, label, maximum)


def _bounded_string(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise AgentContractError(f"{label} must be a string")
    if len(value) > maximum:
        raise AgentContractError(f"{label} exceeds the configured character limit")
    return value


__all__ = [
    "DEFAULT_MAX_AGENT_ID_CHARS",
    "DEFAULT_MAX_DESCRIPTION_CHARS",
    "DEFAULT_MAX_ERROR_CODE_CHARS",
    "DEFAULT_MAX_ERROR_MESSAGE_CHARS",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_MAX_RESULT_CHARS",
    "SCHEMA_VERSION",
    "AgentContractError",
    "agent_id_input_schema",
    "agent_input_schema",
    "build_wait_id",
    "normalize_agent_id",
    "normalize_agent_request",
    "positive_int",
    "snapshot_output_schema",
    "snapshot_payload",
]
