from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from agent_runtime import (
    AgentEvent,
    EventTypes,
    Message,
    ModelResponse,
    RunTrace,
)
from agent_runtime_conformance import ConformanceRunner
from agent_runtime_conformance import main as conformance_main

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "spec" / "v0"
CASES_DIR = REPO_ROOT / "conformance" / "cases"
SHARED_CONFORMANCE_RUNNER = ConformanceRunner(cases_dir=CASES_DIR, spec_dir=SPEC_DIR)


def load_json_schema(path: Path) -> dict[str, Any]:
    raw_schema = json.loads(path.read_text())
    if not isinstance(raw_schema, dict):
        raise TypeError(f"{path.name} must contain a schema object")
    schema = cast(dict[str, Any], raw_schema)
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError(f"{path.name} must define a non-empty $id")
    return schema


SCHEMAS = {path.name: load_json_schema(path) for path in sorted(SPEC_DIR.glob("*.schema.json"))}
REGISTRY_CLS: Any = Registry
RESOURCE_CLS: Any = Resource
DRAFT_2020_12_SPEC: Any = DRAFT202012
SCHEMA_REGISTRY: Any = REGISTRY_CLS().with_resources(
    [
        (
            cast(str, schema["$id"]),
            RESOURCE_CLS.from_contents(cast(Any, schema), default_specification=DRAFT_2020_12_SPEC),
        )
        for schema in SCHEMAS.values()
    ]
)
EVENT_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["events.schema.json"],
    registry=SCHEMA_REGISTRY,
)
RUN_SNAPSHOT_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["run-snapshot.schema.json"],
    registry=SCHEMA_REGISTRY,
)
RUN_TRACE_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["run-trace.schema.json"],
    registry=SCHEMA_REGISTRY,
)
RESUME_INPUT_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["resume-input.schema.json"],
    registry=SCHEMA_REGISTRY,
)
MESSAGE_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["messages.schema.json"],
    registry=SCHEMA_REGISTRY,
)
MODEL_RESPONSE_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["model-response.schema.json"],
    registry=SCHEMA_REGISTRY,
)
LIMITS_SCHEMA_VALIDATOR = Draft202012Validator(
    SCHEMAS["limits.schema.json"],
    registry=SCHEMA_REGISTRY,
)


def assert_matches_schema(
    label: str,
    validator: Any,
    instance: Mapping[str, Any],
) -> None:
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise AssertionError(f"{label} schema violation at {path}: {error.message}") from error


def test_shared_conformance_runner_loads_repository_cases() -> None:
    cases = SHARED_CONFORMANCE_RUNNER.load_cases()

    assert len(cases) == len(list(CASES_DIR.glob("*.json")))


def test_shared_conformance_runner_rejects_empty_case_dir(tmp_path: Path) -> None:
    runner = ConformanceRunner(cases_dir=tmp_path, spec_dir=SPEC_DIR)

    with pytest.raises(ValueError, match="no conformance case"):
        runner.load_cases()


def test_conformance_cli_rejects_empty_case_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = conformance_main([str(tmp_path), "--spec-dir", str(SPEC_DIR), "--quiet"])

    assert status == 1
    assert "FAIL load:" in capsys.readouterr().out


def test_conformance_cli_reports_invalid_case_json_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_case = tmp_path / "bad.json"
    bad_case.write_text("{")

    status = conformance_main([str(tmp_path), "--spec-dir", str(SPEC_DIR), "--quiet"])
    output = capsys.readouterr().out

    assert status == 1
    assert str(bad_case.resolve()) in output
    assert "invalid JSON" in output


def test_conformance_cli_reports_invalid_case_content_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_case = tmp_path / "bad.json"
    bad_case.write_text(
        json.dumps(
            {
                "name": "bad_case",
                "model_steps": [],
                "expected_status": "completed",
                "expected_tool_calls": 0,
                "expected_pendig_tool_call_ids": [],
            }
        )
    )

    status = conformance_main([str(tmp_path), "--spec-dir", str(SPEC_DIR), "--quiet"])
    output = capsys.readouterr().out

    assert status == 1
    assert str(bad_case.resolve()) in output
    assert "invalid conformance case" in output
    assert "unknown key" in output


def test_conformance_cli_reports_invalid_spec_json_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_schema = tmp_path / "events.schema.json"
    bad_schema.write_text("{")

    status = conformance_main([str(CASES_DIR), "--spec-dir", str(tmp_path), "--quiet"])
    output = capsys.readouterr().out

    assert status == 1
    assert str(bad_schema.resolve()) in output
    assert "invalid JSON" in output


def test_conformance_cli_reports_non_object_spec_json_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_schema = tmp_path / "events.schema.json"
    bad_schema.write_text("[]")

    status = conformance_main([str(CASES_DIR), "--spec-dir", str(tmp_path), "--quiet"])
    output = capsys.readouterr().out

    assert status == 1
    assert str(bad_schema.resolve()) in output
    assert "schema must contain an object" in output


def test_conformance_cli_reports_invalid_json_schema_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_schema = tmp_path / "events.schema.json"
    bad_schema.write_text(json.dumps({"$id": "https://example.test/events.schema.json", "type": 1}))

    status = conformance_main([str(CASES_DIR), "--spec-dir", str(tmp_path), "--quiet"])
    output = capsys.readouterr().out

    assert status == 1
    assert str(bad_schema.resolve()) in output
    assert "invalid JSON schema" in output


def test_conformance_case_validation_rejects_unknown_keys() -> None:
    with pytest.raises(AssertionError, match="unknown key"):
        SHARED_CONFORMANCE_RUNNER.validate_case_keys(
            "bad_case",
            {
                "name": "bad_case",
                "model_steps": [],
                "expected_status": "completed",
                "expected_tool_calls": 0,
                "expected_pendig_tool_call_ids": [],
            },
        )


def test_conformance_case_validation_rejects_incomplete_stream_events() -> None:
    with pytest.raises(KeyError, match="part_type"):
        SHARED_CONFORMANCE_RUNNER.validate_case_keys(
            "bad_stream_case",
            {
                "name": "bad_stream_case",
                "model_steps": [],
                "stream_model_steps": [
                    {
                        "events": [
                            {
                                "type": "text_delta",
                                "index": 0,
                                "text_delta": "partial",
                            }
                        ]
                    }
                ],
                "expected_status": "paused",
                "expected_tool_calls": 0,
            },
        )


def test_conformance_case_validation_rejects_resume_keys_without_resume_type() -> None:
    with pytest.raises(AssertionError, match="resume-only"):
        SHARED_CONFORMANCE_RUNNER.validate_case_keys(
            "bad_resume_case",
            {
                "name": "bad_resume_case",
                "model_steps": [],
                "expected_status": "completed",
                "expected_tool_calls": 0,
                "resume_checkpoint_status": "planning",
                "expected_resume_status": "completed",
            },
        )


def test_conformance_case_validation_rejects_mistyped_limits() -> None:
    with pytest.raises(TypeError, match="stop_on_tool_error"):
        SHARED_CONFORMANCE_RUNNER.validate_case_keys(
            "bad_limits_case",
            {
                "name": "bad_limits_case",
                "limits": {"stop_on_tool_error": "false"},
                "model_steps": [],
                "expected_status": "completed",
                "expected_tool_calls": 0,
            },
        )


def test_run_trace_schema_rejects_raw_payload_metadata() -> None:
    trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "run_started",
                "before_status": None,
                "after_status": "planning",
                "references": {},
                "payload": {
                    "status": "planning",
                    "message_count": 1,
                    "message_roles": [],
                    "pending_tool_call_ids": [],
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "final_part_count": 0,
                    "error": None,
                    "pause": None,
                },
                "schema_version": "v0",
            },
            {
                "step_id": 2,
                "kind": "state_changed",
                "before_status": "planning",
                "after_status": "completed",
                "references": {},
                "payload": {
                    "from": "planning",
                    "to": "completed",
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "error": None,
                    "pause": None,
                },
                "schema_version": "v0",
            },
            {
                "step_id": 3,
                "kind": "checkpoint",
                "before_status": "completed",
                "after_status": "completed",
                "references": {},
                "payload": {
                    "status": "completed",
                    "message_count": 2,
                    "message_roles": ["user", "assistant"],
                    "pending_tool_call_ids": [],
                    "iterations": 1,
                    "total_tool_calls": 0,
                    "final_part_count": 1,
                    "error": None,
                    "pause": None,
                    "context_sequence": 3,
                },
                "schema_version": "v0",
            },
            {
                "step_id": 4,
                "kind": "final",
                "before_status": "completed",
                "after_status": "completed",
                "references": {},
                "payload": {
                    "part_count": 1,
                    "part_types": ["text"],
                    "text_length": 4,
                    "metadata_keys": ["secret"],
                    "metadata": {"secret": "value"},
                },
                "schema_version": "v0",
            },
            {
                "step_id": 5,
                "kind": "run_completed",
                "before_status": "completed",
                "after_status": "completed",
                "references": {},
                "payload": {
                    "state": {
                        "status": "completed",
                        "message_count": 2,
                        "message_roles": [],
                        "pending_tool_call_ids": [],
                        "iterations": 1,
                        "total_tool_calls": 0,
                        "final_part_count": 1,
                        "error": None,
                        "pause": None,
                    }
                },
                "schema_version": "v0",
            },
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema("raw metadata trace", RUN_TRACE_SCHEMA_VALIDATOR, trace)


INVALID_MESSAGE_SCHEMA_PAYLOADS: list[dict[str, Any]] = [
    {"role": "user", "parts": [{"type": "text"}]},
    {"role": "user", "parts": [{"type": "file", "uri": "file://a", "ref": "artifact-a"}]},
    {"role": "user", "parts": [], "tool_call_id": "call-1"},
    {"role": "tool", "parts": []},
    {
        "role": "user",
        "parts": [],
        "tool_calls": [{"id": "call-1", "name": "tool", "mode": "execute", "arguments": {}}],
    },
    {
        "role": "assistant",
        "parts": [],
        "tool_calls": [{"id": "", "name": "tool", "mode": "execute", "arguments": {}}],
    },
    {
        "role": "assistant",
        "parts": [],
        "tool_calls": [{"id": "call-1", "name": "", "mode": "execute", "arguments": {}}],
    },
]


@pytest.mark.parametrize("payload", INVALID_MESSAGE_SCHEMA_PAYLOADS)
def test_message_schema_rejects_runtime_invalid_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema("message", MESSAGE_SCHEMA_VALIDATOR, payload)
    with pytest.raises((TypeError, ValueError, KeyError)):
        Message.from_dict(payload)


def test_tool_call_ids_are_portably_unique_beyond_json_schema() -> None:
    duplicate_calls: list[dict[str, Any]] = [
        {"id": "call-1", "name": "tool", "mode": "execute", "arguments": {}},
        {"id": "call-1", "name": "tool", "mode": "execute", "arguments": {}},
    ]
    message: dict[str, Any] = {"role": "assistant", "parts": [], "tool_calls": duplicate_calls}
    response: dict[str, Any] = {"parts": [], "tool_calls": duplicate_calls}

    assert_matches_schema("duplicate tool call message", MESSAGE_SCHEMA_VALIDATOR, message)
    assert_matches_schema(
        "duplicate tool call model response", MODEL_RESPONSE_SCHEMA_VALIDATOR, response
    )
    with pytest.raises(ValueError, match="unique"):
        Message.from_dict(message)
    with pytest.raises(ValueError, match="unique"):
        ModelResponse.from_dict(response)


def test_tool_result_pause_schemas_reject_interrupting_waits() -> None:
    pause: dict[str, Any] = {
        "reason": "external_wait",
        "source": "tool",
        "wait_id": "job-1",
        "metadata": {},
        "interrupt": True,
    }
    event = AgentEvent(
        EventTypes.TOOL_COMPLETED,
        {
            "id": "call-1",
            "name": "wait",
            "mode": "execute",
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": {
                "part_count": 1,
                "part_types": ["text"],
                "text_length": 7,
                "result_kind": "observation",
                "is_error": False,
                "metadata": {},
                "pause": pause,
            },
        },
        run_id="run-1",
        sequence=1,
    ).to_dict()
    trace = RunTrace.from_events(
        "run-1",
        [
            AgentEvent(
                EventTypes.RUN_STARTED,
                {
                    "state": {
                        "status": "executing_tools",
                        "message_count": 2,
                        "pending_tool_call_count": 1,
                        "iterations": 1,
                        "total_tool_calls": 0,
                        "has_final": False,
                        "error": None,
                        "pause": None,
                    }
                },
                run_id="run-1",
                sequence=1,
            ),
            AgentEvent(
                EventTypes.TOOL_STARTED,
                {
                    "id": "call-1",
                    "name": "wait",
                    "mode": "execute",
                    "arguments": {},
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                },
                run_id="run-1",
                sequence=2,
            ),
            AgentEvent(
                EventTypes.TOOL_COMPLETED,
                {
                    "id": "call-1",
                    "name": "wait",
                    "mode": "execute",
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 7,
                        "result_kind": "observation",
                        "is_error": False,
                        "metadata": {},
                        "pause": pause,
                    },
                },
                run_id="run-1",
                sequence=3,
            ),
            AgentEvent(
                EventTypes.STATE_CHANGED,
                {
                    "from": "executing_tools",
                    "to": "paused",
                    "iterations": 1,
                    "total_tool_calls": 1,
                    "error": None,
                    "pause": {
                        "reason": "external_wait",
                        "resume_status": "planning",
                        "source": "tool",
                        "wait_id": "job-1",
                        "metadata": {},
                    },
                },
                run_id="run-1",
                sequence=4,
            ),
            AgentEvent(
                EventTypes.CHECKPOINT,
                {
                    "state": {
                        "status": "paused",
                        "messages": [
                            {"role": "user", "parts": [{"type": "text", "text": "run"}]},
                            {
                                "role": "assistant",
                                "parts": [],
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "name": "wait",
                                        "mode": "execute",
                                        "arguments": {},
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "parts": [{"type": "text", "text": "waiting"}],
                                "tool_call_id": "call-1",
                            },
                        ],
                        "pending_tool_calls": [],
                        "iterations": 1,
                        "total_tool_calls": 1,
                        "final_parts": [],
                        "error": None,
                        "pause": {
                            "reason": "external_wait",
                            "resume_status": "planning",
                            "source": "tool",
                            "wait_id": "job-1",
                            "metadata": {},
                        },
                    },
                    "context": {
                        "run_id": "run-1",
                        "started_at": 1.0,
                        "deadline": None,
                        "metadata": {},
                        "sequence": 5,
                    },
                },
                run_id="run-1",
                sequence=5,
            ),
            AgentEvent(
                EventTypes.RUN_PAUSED,
                {
                    "pause": {
                        "reason": "external_wait",
                        "resume_status": "planning",
                        "source": "tool",
                        "wait_id": "job-1",
                        "metadata": {},
                    }
                },
                run_id="run-1",
                sequence=6,
            ),
            AgentEvent(
                EventTypes.RUN_COMPLETED,
                {
                    "state": {
                        "status": "paused",
                        "message_count": 3,
                        "pending_tool_call_count": 0,
                        "iterations": 1,
                        "total_tool_calls": 1,
                        "has_final": False,
                        "error": None,
                        "pause": {
                            "reason": "external_wait",
                            "resume_status": "planning",
                            "source": "tool",
                            "wait_id": "job-1",
                            "metadata": {},
                        },
                    }
                },
                run_id="run-1",
                sequence=7,
            ),
        ],
    ).to_dict()

    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema("interrupting tool pause event", EVENT_SCHEMA_VALIDATOR, event)
    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema("interrupting tool pause trace", RUN_TRACE_SCHEMA_VALIDATOR, trace)

    pause_requested_event = AgentEvent(
        EventTypes.PAUSE_REQUESTED,
        {"request": pause, "resume_status": "planning", "origin": "tool_result"},
        run_id="run-1",
        sequence=8,
    ).to_dict()
    pause_requested_trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "pause_requested",
                "before_status": "executing_tools",
                "after_status": "executing_tools",
                "references": {},
                "payload": {
                    "reason": "external_wait",
                    "source": "tool",
                    "wait_id": "job-1",
                    "metadata_keys": [],
                    "interrupt": True,
                    "resume_status": "planning",
                    "origin": "tool_result",
                },
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema(
            "interrupting pause_requested event", EVENT_SCHEMA_VALIDATOR, pause_requested_event
        )
    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema(
            "interrupting pause_requested trace",
            RUN_TRACE_SCHEMA_VALIDATOR,
            pause_requested_trace,
        )


def test_tool_result_schemas_reject_known_mode_result_kind_mismatch() -> None:
    event = AgentEvent(
        EventTypes.TOOL_COMPLETED,
        {
            "id": "call-1",
            "name": "tool",
            "mode": "accept",
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": {
                "part_count": 1,
                "part_types": ["text"],
                "text_length": 2,
                "result_kind": "observation",
                "is_error": False,
                "metadata": {},
                "pause": None,
            },
        },
        run_id="run-1",
        sequence=1,
    ).to_dict()
    trace_payload: dict[str, Any] = {
        "id": "call-1",
        "name": "tool",
        "mode": "accept",
        "batch_id": "tool-batch-1",
        "parallel": False,
        "index": 0,
        "result": {
            "part_count": 1,
            "part_types": ["text"],
            "text_length": 2,
            "result_kind": "observation",
            "is_error": False,
            "metadata_keys": [],
            "pause": None,
        },
    }
    trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "tool_result",
                "before_status": "executing_tools",
                "after_status": "executing_tools",
                "references": {},
                "payload": trace_payload,
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    with pytest.raises(AssertionError, match="acceptance"):
        assert_matches_schema("tool_completed event", EVENT_SCHEMA_VALIDATOR, event)
    with pytest.raises(AssertionError, match="acceptance"):
        assert_matches_schema("tool_result trace", RUN_TRACE_SCHEMA_VALIDATOR, trace)


def test_tool_result_schemas_allow_accept_mode_rejection() -> None:
    event = AgentEvent(
        EventTypes.TOOL_COMPLETED,
        {
            "id": "call-1",
            "name": "tool",
            "mode": "accept",
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": {
                "part_count": 1,
                "part_types": ["text"],
                "text_length": 8,
                "result_kind": "rejection",
                "is_error": True,
                "metadata": {},
                "pause": None,
            },
        },
        run_id="run-1",
        sequence=1,
    ).to_dict()
    trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "tool_result",
                "before_status": "executing_tools",
                "after_status": "executing_tools",
                "references": {},
                "payload": {
                    "id": "call-1",
                    "name": "tool",
                    "mode": "accept",
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 8,
                        "result_kind": "rejection",
                        "is_error": True,
                        "metadata_keys": [],
                        "pause": None,
                    },
                },
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    assert_matches_schema("accept rejection tool_completed event", EVENT_SCHEMA_VALIDATOR, event)
    assert_matches_schema("accept rejection tool_result trace", RUN_TRACE_SCHEMA_VALIDATOR, trace)


def test_tool_result_schemas_allow_extension_mode_result_kind() -> None:
    event = AgentEvent(
        EventTypes.TOOL_COMPLETED,
        {
            "id": "call-1",
            "name": "tool",
            "mode": "handoff",
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": {
                "part_count": 1,
                "part_types": ["text"],
                "text_length": 2,
                "result_kind": "handoff",
                "is_error": False,
                "metadata": {},
                "pause": None,
                "correlation_id": "job-1",
            },
        },
        run_id="run-1",
        sequence=1,
    ).to_dict()
    trace_payload: dict[str, Any] = {
        "id": "call-1",
        "name": "tool",
        "mode": "handoff",
        "batch_id": "tool-batch-1",
        "parallel": False,
        "index": 0,
        "result": {
            "part_count": 1,
            "part_types": ["text"],
            "text_length": 2,
            "result_kind": "handoff",
            "is_error": False,
            "metadata_keys": [],
            "pause": None,
            "correlation_id": "job-1",
        },
    }
    trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "tool_result",
                "before_status": "executing_tools",
                "after_status": "executing_tools",
                "references": {},
                "payload": trace_payload,
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    assert_matches_schema("extension tool_completed event", EVENT_SCHEMA_VALIDATOR, event)
    assert_matches_schema("extension tool_result trace", RUN_TRACE_SCHEMA_VALIDATOR, trace)


@pytest.mark.parametrize("reserved_kind", ["observation", "acceptance", "rejection"])
def test_tool_result_schemas_reject_extension_mode_reserved_result_kind(
    reserved_kind: str,
) -> None:
    event_result: dict[str, Any] = {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 2,
        "result_kind": reserved_kind,
        "is_error": False,
        "metadata": {},
        "pause": None,
    }
    trace_result: dict[str, Any] = {
        "part_count": 1,
        "part_types": ["text"],
        "text_length": 2,
        "result_kind": reserved_kind,
        "is_error": False,
        "metadata_keys": [],
        "pause": None,
    }
    if reserved_kind == "acceptance":
        event_result["correlation_id"] = "job-1"
        trace_result["correlation_id"] = "job-1"
    if reserved_kind == "rejection":
        event_result["is_error"] = True
        trace_result["is_error"] = True

    event = AgentEvent(
        EventTypes.TOOL_COMPLETED,
        {
            "id": "call-1",
            "name": "tool",
            "mode": "handoff",
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": event_result,
        },
        run_id="run-1",
        sequence=1,
    ).to_dict()
    trace: dict[str, Any] = {
        "run_id": "run-1",
        "steps": [
            {
                "step_id": 1,
                "kind": "tool_result",
                "before_status": "executing_tools",
                "after_status": "executing_tools",
                "references": {},
                "payload": {
                    "id": "call-1",
                    "name": "tool",
                    "mode": "handoff",
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                    "result": trace_result,
                },
                "schema_version": "v0",
            }
        ],
        "metadata": {"metadata_keys": []},
        "schema_version": "v0",
    }

    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema(
            "extension reserved tool_completed event", EVENT_SCHEMA_VALIDATOR, event
        )
    with pytest.raises(AssertionError, match="schema violation"):
        assert_matches_schema(
            "extension reserved tool_result trace", RUN_TRACE_SCHEMA_VALIDATOR, trace
        )


def test_resume_input_schema_rejects_invalid_cross_field_combinations() -> None:
    def payload(
        status: str,
        *,
        pause: dict[str, Any] | None = None,
        pending_tool_calls: list[dict[str, Any]] | None = None,
        append_messages: list[dict[str, Any]] | None = None,
        expected_pause: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "snapshot": {
                "state": {
                    "status": status,
                    "messages": [{"role": "user", "parts": [{"type": "text", "text": "run"}]}],
                    "pending_tool_calls": [] if pending_tool_calls is None else pending_tool_calls,
                    "iterations": 0,
                    "total_tool_calls": 0,
                    "final_parts": [],
                    "error": None,
                    "pause": pause,
                },
                "context": {
                    "run_id": "run-1",
                    "started_at": 1.0,
                    "deadline": None,
                    "metadata": {},
                    "sequence": 1,
                },
            },
            "append_messages": [] if append_messages is None else append_messages,
            "expected_pause": expected_pause,
            "metadata": {},
        }

    planning_append = payload(
        "planning",
        append_messages=[{"role": "user", "parts": [{"type": "text", "text": "callback"}]}],
    )
    planning_selector = payload(
        "planning",
        expected_pause={"reason": "manual_pause", "source": None, "wait_id": None, "metadata": {}},
    )
    terminal = payload("completed")
    paused_executing_append = payload(
        "paused",
        pause={
            "reason": "external_wait",
            "resume_status": "executing_tools",
            "source": "tool",
            "wait_id": "job-1",
            "metadata": {},
        },
        append_messages=[{"role": "user", "parts": [{"type": "text", "text": "callback"}]}],
    )
    pending_call: dict[str, Any] = {
        "id": "call-1",
        "name": "echo",
        "mode": "execute",
        "arguments": {},
    }
    planning_pending = payload("planning", pending_tool_calls=[pending_call])
    executing_without_pending = payload("executing_tools")
    paused_planning_pending = payload(
        "paused",
        pause={
            "reason": "manual_pause",
            "resume_status": "planning",
            "source": "host",
            "wait_id": None,
            "metadata": {},
        },
        pending_tool_calls=[pending_call],
    )
    paused_executing_without_pending = payload(
        "paused",
        pause={
            "reason": "external_wait",
            "resume_status": "executing_tools",
            "source": "tool",
            "wait_id": "job-1",
            "metadata": {},
        },
    )
    empty_selector = payload(
        "paused",
        pause={
            "reason": "manual_pause",
            "resume_status": "planning",
            "source": "host",
            "wait_id": None,
            "metadata": {},
        },
        expected_pause={"reason": None, "source": None, "wait_id": None, "metadata": {}},
    )

    for label, instance in {
        "planning append": planning_append,
        "planning selector": planning_selector,
        "terminal resume": terminal,
        "executing-tools append": paused_executing_append,
        "planning pending": planning_pending,
        "executing without pending": executing_without_pending,
        "paused planning pending": paused_planning_pending,
        "paused executing without pending": paused_executing_without_pending,
        "empty selector": empty_selector,
    }.items():
        with pytest.raises(AssertionError, match="schema violation"):
            assert_matches_schema(label, RESUME_INPUT_SCHEMA_VALIDATOR, instance)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", SHARED_CONFORMANCE_RUNNER.load_cases(), ids=lambda case: str(case["name"])
)
async def test_conformance_case(case: dict[str, Any]) -> None:
    await SHARED_CONFORMANCE_RUNNER.run_case(case)
