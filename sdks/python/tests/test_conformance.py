from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from agent_runtime import (
    AgentEvent,
    AgentLoop,
    AgentResult,
    AgentStatus,
    ContentPart,
    EventTypes,
    LoopLimits,
    Message,
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    PauseController,
    PauseRequest,
    PauseSelector,
    ResumeInput,
    RunSnapshot,
    RuntimeContext,
    RunTrace,
    Tool,
    ToolResult,
    ToolSpec,
    replay_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "spec" / "v0"
CASES_DIR = REPO_ROOT / "conformance" / "cases"


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


class ScriptedModel:
    def __init__(
        self,
        steps: Sequence[ModelResponse],
        *,
        pause_controller: PauseController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
    ) -> None:
        self._steps = list(steps)
        self._pause_controller = pause_controller
        self._pause_request_on_call = pause_request_on_call
        self._pause_request_on_stream_event = pause_request_on_stream_event
        self._pause_requested = False
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        response = self._steps[self.calls]
        self.calls += 1
        self._apply_pause_once(self._pause_request_on_call)
        return response

    def _apply_pause_once(self, request: PauseRequest | None) -> None:
        if self._pause_controller is not None and request is not None and not self._pause_requested:
            self._pause_requested = True
            apply_pause_request(self._pause_controller, request)


class StreamedCaseModel(ScriptedModel):
    def __init__(
        self,
        steps: Sequence[ModelResponse],
        stream_steps: Sequence[dict[str, Any]],
        *,
        pause_controller: PauseController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
    ) -> None:
        super().__init__(
            steps,
            pause_controller=pause_controller,
            pause_request_on_call=pause_request_on_call,
            pause_request_on_stream_event=pause_request_on_stream_event,
        )
        self._stream_steps = list(stream_steps)
        self.stream_calls = 0

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        if self.stream_calls >= len(self._stream_steps):
            raise AssertionError("scripted stream model exhausted")

        step = self._stream_steps[self.stream_calls]
        self.stream_calls += 1
        self._apply_pause_once(self._pause_request_on_call)
        for raw_event in cast(list[dict[str, Any]], step.get("events") or []):
            event_type = expect_case_str(raw_event["type"], "stream event type")
            if event_type == "text_delta":
                yield ModelContentDelta(
                    index=expect_case_int(raw_event["index"], "stream event index"),
                    text_delta=expect_case_str(raw_event["text_delta"], "stream event text_delta"),
                    part_type=expect_case_str(raw_event["part_type"], "stream event part_type"),
                )
            elif event_type == "tool_call_delta":
                yield ModelToolCallDelta(
                    index=expect_case_int(raw_event["index"], "stream event index"),
                    id=expect_case_optional_str(raw_event.get("id"), "stream event id"),
                    name=expect_case_optional_str(raw_event.get("name"), "stream event name"),
                    arguments_delta=expect_case_optional_str(
                        raw_event.get("arguments_delta"), "stream event arguments_delta"
                    ),
                )
            elif event_type == "sleep":
                await asyncio.sleep(
                    expect_case_number(raw_event["seconds"], "stream event seconds")
                )
            elif event_type == "pause_request":
                self._apply_pause_once(self._pause_request_on_stream_event)
            else:
                raise AssertionError(f"unsupported stream event type: {event_type}")


class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.text(str(arguments.get("text", "")))


class FailTool:
    spec = ToolSpec(
        name="fail",
        description="Raise an error.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = arguments, context
        raise RuntimeError("tool failed")


class DelayedEchoTool:
    spec = ToolSpec(
        name="delayed_echo",
        description="Return input text after an optional delay.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        await asyncio.sleep(float(arguments.get("delay", 0)))
        return ToolResult.text(str(arguments.get("text", "")))


class WaitTool:
    spec = ToolSpec(
        name="wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        return ToolResult.waiting(
            str(arguments.get("text", "external wait started")),
            wait_id=str(arguments["wait_id"]),
            reason=str(arguments.get("reason", "external_wait")),
        )


class ParallelWaitTool:
    spec = ToolSpec(
        name="parallel_wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(self, arguments: dict[str, Any], context: RuntimeContext) -> ToolResult:
        _ = context
        await asyncio.sleep(float(arguments.get("delay", 0)))
        return ToolResult.waiting(
            str(arguments.get("text", "external wait started")),
            wait_id=str(arguments["wait_id"]),
            reason=str(arguments.get("reason", "external_wait")),
        )


CASE_KEYS = {
    "name",
    "case_type",
    "limits",
    "pause_request",
    "pause_request_timing",
    "model_steps",
    "resume_model_steps",
    "resume_append_messages",
    "resume_expected_pause",
    "resume_checkpoint_status",
    "resume_checkpoint_total_tool_calls",
    "stream_model_steps",
    "expected_status",
    "expected_final_text",
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
    "expected_pending_tool_call_ids",
    "expected_pause",
    "expected_model_deltas",
    "forbidden_event_types",
    "forbidden_checkpoint_statuses",
    "forbidden_checkpoint_tool_counts",
    "forbidden_checkpoint_status_tool_counts",
    "forbidden_unpaused_checkpoint_tool_counts",
    "forbidden_checkpoint_message_roles",
    "model_response",
    "expected_error",
}
NEGATIVE_CASE_TYPES = {"model_response_negative"}
MODEL_STEP_REQUIRED_KEYS = {"parts", "tool_calls"}
MODEL_STEP_KEYS = {
    "parts",
    "tool_calls",
    "finish_reason",
    "usage",
    "model",
    "response_id",
    "metadata",
}
STREAM_STEP_KEYS = {"events"}
STREAM_EVENT_KEYS = {
    "type",
    "index",
    "text_delta",
    "part_type",
    "id",
    "name",
    "arguments_delta",
    "seconds",
}
LIMIT_KEYS = {
    "max_iterations",
    "max_total_tool_calls",
    "timeout_seconds",
    "stop_on_tool_error",
    "max_parallel_tool_calls",
}
STREAM_EVENT_REQUIRED_KEYS: dict[str, set[str]] = {
    "text_delta": {"index", "text_delta", "part_type"},
    "tool_call_delta": {"index"},
    "sleep": {"seconds"},
    "pause_request": set(),
}


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text())
        if not isinstance(case, dict):
            raise TypeError(f"{path.name} must contain an object")
        validate_case_keys(path.name, cast(dict[str, Any], case))
        cases.append(cast(dict[str, Any], case))
    return cases


def reject_unknown_keys(keys: set[str], allowed: set[str], label: str) -> None:
    unknown = keys - allowed
    if unknown:
        raise AssertionError(f"{label} has unknown key(s): {', '.join(sorted(unknown))}")


def expect_case_list(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return cast(list[dict[str, Any]], value)


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


def validate_case_keys(name: str, case: dict[str, Any]) -> None:
    reject_unknown_keys(set(case), CASE_KEYS, name)
    case_type = expect_case_str(case.get("case_type", "run"), f"{name}.case_type")
    if case_type not in {"run", "resume", *NEGATIVE_CASE_TYPES}:
        raise ValueError(f"{name} has invalid case_type")
    if case_type in NEGATIVE_CASE_TYPES:
        for required in ("name", "model_response", "expected_error"):
            if required not in case:
                raise KeyError(f"{name} missing required key: {required}")
        expect_case_str(case["name"], f"{name}.name")
        expect_case_str(case["expected_error"], f"{name}.expected_error")
        assert_matches_schema(
            f"{name}.model_response",
            MODEL_RESPONSE_SCHEMA_VALIDATOR,
            cast(Mapping[str, Any], case["model_response"]),
        )
        forbidden = set(case) - {"name", "case_type", "model_response", "expected_error"}
        if forbidden:
            raise AssertionError(
                f"{name} has invalid negative-case key(s): {', '.join(sorted(forbidden))}"
            )
        return
    negative_only_keys = {"model_response", "expected_error"} & set(case)
    if negative_only_keys:
        raise AssertionError(
            f"{name} has negative-only key(s): {', '.join(sorted(negative_only_keys))}"
        )
    resume_only_keys = {
        "resume_model_steps",
        "resume_append_messages",
        "resume_expected_pause",
        "resume_checkpoint_status",
        "resume_checkpoint_total_tool_calls",
        "expected_resume_status",
        "expected_resume_final_text",
        "expected_resume_tool_calls",
        "expected_resume_message_roles",
        "expected_resume_tool_texts",
        "expected_resume_error",
        "expected_resume_trace_prefix",
    }
    if case_type != "resume":
        forbidden = resume_only_keys & set(case)
        if forbidden:
            raise AssertionError(
                f"{name} has resume-only key(s) without case_type=resume: "
                f"{', '.join(sorted(forbidden))}"
            )
    for required in ("name", "model_steps", "expected_status", "expected_tool_calls"):
        if required not in case:
            raise KeyError(f"{name} missing required key: {required}")
    expect_case_str(case["name"], f"{name}.name")
    expect_case_str(case["expected_status"], f"{name}.expected_status")
    expect_case_int(case["expected_tool_calls"], f"{name}.expected_tool_calls")
    if case_type == "resume":
        for required in ("resume_checkpoint_status",):
            if required not in case:
                raise KeyError(f"{name} missing required key: {required}")
        expect_case_str(case["resume_checkpoint_status"], f"{name}.resume_checkpoint_status")
        expect_case_optional_int(
            case.get("resume_checkpoint_total_tool_calls"),
            f"{name}.resume_checkpoint_total_tool_calls",
        )
        if "expected_resume_error" not in case and "expected_resume_status" not in case:
            raise KeyError(f"{name} missing required key: expected_resume_status")
        if "expected_resume_status" in case:
            expect_case_str(case["expected_resume_status"], f"{name}.expected_resume_status")
        if "expected_resume_tool_calls" in case:
            expect_case_int(
                case["expected_resume_tool_calls"], f"{name}.expected_resume_tool_calls"
            )
        if "expected_resume_error" in case:
            expect_case_str(case["expected_resume_error"], f"{name}.expected_resume_error")
    if case.get("pause_request_timing") not in {None, "during_model_call", "stream_event"}:
        raise ValueError(f"{name} has invalid pause_request_timing")
    if "limits" in case:
        raw_limits = case["limits"]
        if not isinstance(raw_limits, dict):
            raise TypeError(f"{name}.limits must be an object")
        limits = cast(dict[str, Any], raw_limits)
        reject_unknown_keys(set(limits), LIMIT_KEYS, f"{name}.limits")
        for key in ("max_iterations", "max_total_tool_calls", "max_parallel_tool_calls"):
            if key in limits and (
                not isinstance(limits[key], int) or isinstance(limits[key], bool)
            ):
                raise TypeError(f"{name}.limits.{key} must be an integer")
        if "timeout_seconds" in limits and (
            not isinstance(limits["timeout_seconds"], int | float)
            or isinstance(limits["timeout_seconds"], bool)
        ):
            raise TypeError(f"{name}.limits.timeout_seconds must be a number")
        if "stop_on_tool_error" in limits and not isinstance(limits["stop_on_tool_error"], bool):
            raise TypeError(f"{name}.limits.stop_on_tool_error must be a boolean")
        assert_matches_schema(f"{name}.limits", LIMITS_SCHEMA_VALIDATOR, limits)
    for index, step in enumerate(expect_case_list(case["model_steps"], f"{name}.model_steps")):
        reject_unknown_keys(set(step), MODEL_STEP_KEYS, f"{name}.model_steps[{index}]")
        for required in MODEL_STEP_REQUIRED_KEYS:
            if required not in step:
                raise KeyError(f"{name}.model_steps[{index}] missing required key: {required}")
        assert_matches_schema(f"{name}.model_steps[{index}]", MODEL_RESPONSE_SCHEMA_VALIDATOR, step)
    raw_resume_steps = case.get("resume_model_steps", [])
    for index, step in enumerate(expect_case_list(raw_resume_steps, f"{name}.resume_model_steps")):
        reject_unknown_keys(set(step), MODEL_STEP_KEYS, f"{name}.resume_model_steps[{index}]")
        for required in MODEL_STEP_REQUIRED_KEYS:
            if required not in step:
                raise KeyError(
                    f"{name}.resume_model_steps[{index}] missing required key: {required}"
                )
        assert_matches_schema(
            f"{name}.resume_model_steps[{index}]", MODEL_RESPONSE_SCHEMA_VALIDATOR, step
        )
    raw_resume_messages = case.get("resume_append_messages", [])
    for message in expect_case_list(raw_resume_messages, f"{name}.resume_append_messages"):
        Message.from_dict(message)
    if "resume_expected_pause" in case:
        raw_selector = case["resume_expected_pause"]
        if not isinstance(raw_selector, dict):
            raise TypeError(f"{name}.resume_expected_pause must be an object")
        PauseSelector.from_dict(cast(Mapping[str, Any], raw_selector))
    raw_stream_steps = case.get("stream_model_steps", [])
    for index, step in enumerate(expect_case_list(raw_stream_steps, f"{name}.stream_model_steps")):
        reject_unknown_keys(set(step), STREAM_STEP_KEYS, f"{name}.stream_model_steps[{index}]")
        if "events" not in step:
            raise KeyError(f"{name}.stream_model_steps[{index}] missing required key: events")
        for event_index, event in enumerate(
            expect_case_list(step["events"], f"{name}.stream_model_steps[{index}].events")
        ):
            reject_unknown_keys(
                set(event),
                STREAM_EVENT_KEYS,
                f"{name}.stream_model_steps[{index}].events[{event_index}]",
            )
            if "type" not in event:
                raise KeyError(
                    f"{name}.stream_model_steps[{index}].events[{event_index}] "
                    "missing required key: type"
                )
            event_type = event["type"]
            event_type = expect_case_str(event_type, f"{name}.stream_model_steps event type")
            required = STREAM_EVENT_REQUIRED_KEYS.get(event_type)
            if required is None:
                raise ValueError(
                    f"{name}.stream_model_steps[{index}].events[{event_index}] "
                    f"has invalid type: {event_type}"
                )
            for key in required:
                if key not in event:
                    raise KeyError(
                        f"{name}.stream_model_steps[{index}].events[{event_index}] "
                        f"missing required key: {key}"
                    )
            if "index" in event:
                expect_case_int(
                    event["index"],
                    f"{name}.stream_model_steps[{index}].events[{event_index}].index",
                )
            if "text_delta" in event:
                expect_case_str(
                    event["text_delta"],
                    f"{name}.stream_model_steps[{index}].events[{event_index}].text_delta",
                )
            if "part_type" in event:
                expect_case_str(
                    event["part_type"],
                    f"{name}.stream_model_steps[{index}].events[{event_index}].part_type",
                )
            if "seconds" in event:
                expect_case_number(
                    event["seconds"],
                    f"{name}.stream_model_steps[{index}].events[{event_index}].seconds",
                )
            for optional_key in ("id", "name", "arguments_delta"):
                if optional_key in event:
                    expect_case_optional_str(
                        event[optional_key],
                        f"{name}.stream_model_steps[{index}].events[{event_index}].{optional_key}",
                    )
    if "forbidden_checkpoint_status_tool_counts" in case:
        for index, item in enumerate(
            expect_case_list(
                case["forbidden_checkpoint_status_tool_counts"],
                f"{name}.forbidden_checkpoint_status_tool_counts",
            )
        ):
            expect_case_str(
                item.get("status"),
                f"{name}.forbidden_checkpoint_status_tool_counts[{index}].status",
            )
            expect_case_int(
                item.get("total_tool_calls"),
                f"{name}.forbidden_checkpoint_status_tool_counts[{index}].total_tool_calls",
            )


def content_part_from_case(part: dict[str, Any]) -> ContentPart:
    return ContentPart.from_dict(part)


def model_response_from_case_step(step: dict[str, Any]) -> ModelResponse:
    return ModelResponse.from_dict(step)


def test_conformance_case_validation_rejects_unknown_keys() -> None:
    with pytest.raises(AssertionError, match="unknown key"):
        validate_case_keys(
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
        validate_case_keys(
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
        validate_case_keys(
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
        validate_case_keys(
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
        "tool_calls": [{"id": "call-1", "name": "tool", "arguments": {}}],
    },
    {
        "role": "assistant",
        "parts": [],
        "tool_calls": [{"id": "", "name": "tool", "arguments": {}}],
    },
    {
        "role": "assistant",
        "parts": [],
        "tool_calls": [{"id": "call-1", "name": "", "arguments": {}}],
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
        {"id": "call-1", "name": "tool", "arguments": {}},
        {"id": "call-1", "name": "tool", "arguments": {}},
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
            "batch_id": "tool-batch-1",
            "parallel": False,
            "index": 0,
            "result": {
                "part_count": 1,
                "part_types": ["text"],
                "text_length": 7,
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
                    "batch_id": "tool-batch-1",
                    "parallel": False,
                    "index": 0,
                    "result": {
                        "part_count": 1,
                        "part_types": ["text"],
                        "text_length": 7,
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
    pending_call: dict[str, Any] = {"id": "call-1", "name": "echo", "arguments": {}}
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


def limits_from_case(case: dict[str, Any]) -> LoopLimits:
    raw_limits = cast(dict[str, Any], case.get("limits", {}))
    return LoopLimits(
        max_iterations=cast(int, raw_limits.get("max_iterations", 8)),
        max_total_tool_calls=cast(int, raw_limits.get("max_total_tool_calls", 20)),
        timeout_seconds=cast(float | None, raw_limits.get("timeout_seconds")),
        stop_on_tool_error=cast(bool, raw_limits.get("stop_on_tool_error", False)),
        max_parallel_tool_calls=cast(int, raw_limits.get("max_parallel_tool_calls", 1)),
    )


def pause_request_from_case(case: dict[str, Any]) -> PauseRequest | None:
    raw_pause_obj = case.get("pause_request")
    if not isinstance(raw_pause_obj, dict):
        return None
    return PauseRequest.from_dict(cast(Mapping[str, Any], raw_pause_obj))


def apply_pause_request(controller: PauseController, request: PauseRequest) -> None:
    if request.interrupt:
        controller.interrupt(
            reason=request.reason,
            source=request.source,
            wait_id=request.wait_id,
            metadata=request.metadata,
        )
        return
    controller.request_pause(
        reason=request.reason,
        source=request.source,
        wait_id=request.wait_id,
        metadata=request.metadata,
    )


def pause_controller_from_case(case: dict[str, Any]) -> PauseController | None:
    request = pause_request_from_case(case)
    if request is None:
        return None
    controller = PauseController()
    if case.get("pause_request_timing") not in {"during_model_call", "stream_event"}:
        apply_pause_request(controller, request)
    return controller


def model_from_case(
    case: dict[str, Any],
    steps: Sequence[ModelResponse],
    stream_steps: Sequence[dict[str, Any]],
    pause_controller: PauseController | None,
) -> ScriptedModel:
    pause_request_on_call = (
        pause_request_from_case(case)
        if case.get("pause_request_timing") == "during_model_call"
        else None
    )
    pause_request_on_stream_event = (
        pause_request_from_case(case)
        if case.get("pause_request_timing") == "stream_event"
        else None
    )
    if stream_steps:
        return StreamedCaseModel(
            steps,
            stream_steps,
            pause_controller=pause_controller,
            pause_request_on_call=pause_request_on_call,
            pause_request_on_stream_event=pause_request_on_stream_event,
        )
    return ScriptedModel(
        steps,
        pause_controller=pause_controller,
        pause_request_on_call=pause_request_on_call,
        pause_request_on_stream_event=pause_request_on_stream_event,
    )


def case_tools() -> list[Tool]:
    return [EchoTool(), FailTool(), DelayedEchoTool(), WaitTool(), ParallelWaitTool()]


async def run_case_result(
    case: dict[str, Any],
    steps: Sequence[ModelResponse],
    stream_steps: Sequence[dict[str, Any]],
) -> AgentResult:
    pause_controller = pause_controller_from_case(case)
    model = model_from_case(case, steps, stream_steps, pause_controller)
    return await AgentLoop(
        model=model,
        tools=case_tools(),
        limits=limits_from_case(case),
    ).run(
        [Message.user_text("run conformance case")],
        stream=bool(stream_steps),
        pause_controller=pause_controller,
    )


async def collect_case_events(
    case: dict[str, Any],
    steps: Sequence[ModelResponse],
    stream_steps: Sequence[dict[str, Any]],
) -> list[AgentEvent]:
    pause_controller = pause_controller_from_case(case)
    model = model_from_case(case, steps, stream_steps, pause_controller)
    return [
        event
        async for event in AgentLoop(
            model=model,
            tools=case_tools(),
            limits=limits_from_case(case),
        ).run_events(
            [Message.user_text("run conformance case")],
            stream=bool(stream_steps),
            pause_controller=pause_controller,
        )
    ]


async def collect_resume_case_events(
    case: dict[str, Any],
    resume_input: ResumeInput,
    steps: Sequence[ModelResponse],
) -> list[AgentEvent]:
    return [
        event
        async for event in AgentLoop(
            model=model_from_case(case, steps, [], pause_controller=None),
            tools=case_tools(),
            limits=limits_from_case(case),
        ).run_snapshot_events(resume_input)
    ]


def messages_from_case(value: object) -> list[Message]:
    return [
        Message.from_dict(cast(Mapping[str, Any], message))
        for message in expect_case_list(value, "resume_append_messages")
    ]


def resume_selector_from_case(case: dict[str, Any]) -> PauseSelector | None:
    raw_selector = case.get("resume_expected_pause")
    if raw_selector is None:
        return None
    return PauseSelector.from_dict(cast(Mapping[str, Any], raw_selector))


def select_resume_snapshot(case: dict[str, Any], events: Sequence[AgentEvent]) -> RunSnapshot:
    target_status = AgentStatus(expect_case_str(case["resume_checkpoint_status"], "resume status"))
    target_tool_calls = expect_case_optional_int(
        case.get("resume_checkpoint_total_tool_calls"),
        "resume checkpoint total_tool_calls",
    )
    for event in events:
        if event.type != EventTypes.CHECKPOINT:
            continue
        snapshot = RunSnapshot.from_dict(event.data)
        if snapshot.state.status is not target_status:
            continue
        if target_tool_calls is not None and snapshot.state.total_tool_calls != target_tool_calls:
            continue
        return snapshot
    raise AssertionError(f"missing resume checkpoint with status {target_status.value}")


def assert_event_stream_invariants(events: Sequence[AgentEvent], expected: AgentStatus) -> None:
    assert events
    assert events[0].type == EventTypes.RUN_STARTED
    assert events[-1].type == EventTypes.RUN_COMPLETED
    assert events[-1].data["state"]["status"] == expected.value

    run_ids = {event.run_id for event in events}
    assert len(run_ids) == 1
    assert next(iter(run_ids))

    sequences = [event.sequence for event in events]
    assert sequences == sorted(sequences)
    assert len(sequences) == len(set(sequences))

    terminal_state_changed_index: int | None = None
    checkpoint_after_terminal_index: int | None = None
    for index, event in enumerate(events):
        envelope = event.to_dict()
        assert_matches_schema(f"{event.type} event", EVENT_SCHEMA_VALIDATOR, envelope)
        assert set(envelope) == {
            "type",
            "data",
            "run_id",
            "sequence",
            "created_at",
            "schema_version",
        }
        assert isinstance(envelope["data"], dict)

        if event.type == EventTypes.CHECKPOINT:
            snapshot = RunSnapshot.from_dict(event.data)
            assert_matches_schema(
                f"{event.type} snapshot",
                RUN_SNAPSHOT_SCHEMA_VALIDATOR,
                snapshot.to_dict(),
            )
            assert snapshot.context.run_id == event.run_id
            assert snapshot.context.sequence == event.sequence
            if terminal_state_changed_index is not None and snapshot.state.status is expected:
                checkpoint_after_terminal_index = index

        if event.type == EventTypes.STATE_CHANGED and event.data.get("to") == expected.value:
            terminal_state_changed_index = index

    assert terminal_state_changed_index is not None
    assert checkpoint_after_terminal_index is not None
    assert checkpoint_after_terminal_index > terminal_state_changed_index

    if expected is AgentStatus.COMPLETED:
        final_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.FINAL
        ]
        assert final_indexes
        assert checkpoint_after_terminal_index < final_indexes[-1] < len(events) - 1
    elif expected is AgentStatus.PAUSED:
        pause_requested_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.PAUSE_REQUESTED
        ]
        paused_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.RUN_PAUSED
        ]
        assert pause_requested_indexes
        assert paused_indexes
        assert pause_requested_indexes[-1] < terminal_state_changed_index
        assert checkpoint_after_terminal_index < paused_indexes[-1] < len(events) - 1
        assert [event.type for event in events[pause_requested_indexes[-1] :]] == [
            EventTypes.PAUSE_REQUESTED,
            EventTypes.STATE_CHANGED,
            EventTypes.CHECKPOINT,
            EventTypes.RUN_PAUSED,
            EventTypes.RUN_COMPLETED,
        ]
        assert (
            events[paused_indexes[-1]].data["pause"]
            == events[checkpoint_after_terminal_index].data["state"]["pause"]
        )
        assert not [event for event in events if event.type == EventTypes.ERROR]
    else:
        error_indexes = [
            index for index, event in enumerate(events) if event.type == EventTypes.ERROR
        ]
        assert error_indexes
        assert checkpoint_after_terminal_index < error_indexes[-1] < len(events) - 1


def assert_run_case_expectations(
    case: dict[str, Any],
    result: AgentResult,
    events: Sequence[AgentEvent],
    expected_status: AgentStatus,
) -> None:
    assert result.status is expected_status
    assert result.total_tool_calls == case["expected_tool_calls"]
    assert_event_stream_invariants(events, expected_status)
    assert result.trace is not None
    assert_matches_schema("run trace", RUN_TRACE_SCHEMA_VALIDATOR, result.trace.to_dict())
    assert replay_trace(result.trace).final_status is expected_status
    event_trace = RunTrace.from_events(events[0].run_id, events)
    assert_matches_schema("event trace", RUN_TRACE_SCHEMA_VALIDATOR, event_trace.to_dict())
    assert replay_trace(event_trace).final_status is expected_status
    if result.snapshot is not None:
        assert_matches_schema(
            "result snapshot",
            RUN_SNAPSHOT_SCHEMA_VALIDATOR,
            result.snapshot.to_dict(),
        )
    if "expected_message_roles" in case:
        assert result.snapshot is not None
        assert [message.role for message in result.snapshot.state.messages] == case[
            "expected_message_roles"
        ]
    if "expected_final_text" in case:
        assert (
            "".join(part.text or "" for part in result.final_parts) == case["expected_final_text"]
        )
    if "expected_tool_texts" in case:
        assert [message.text for message in result.messages if message.role == "tool"] == case[
            "expected_tool_texts"
        ]
    if "expected_pending_tool_call_ids" in case:
        assert result.snapshot is not None
        assert [call.id for call in result.snapshot.state.pending_tool_calls] == case[
            "expected_pending_tool_call_ids"
        ]
    if "expected_pause" in case:
        assert result.snapshot is not None
        assert result.snapshot.state.pause is not None
        assert result.snapshot.state.pause.to_dict() == case["expected_pause"]
        paused_events = [event for event in events if event.type == EventTypes.RUN_PAUSED]
        assert paused_events
        assert paused_events[-1].data["pause"] == case["expected_pause"]
        pause_requested_events = [
            event for event in events if event.type == EventTypes.PAUSE_REQUESTED
        ]
        assert pause_requested_events
        pause_request = pause_requested_events[-1].data["request"]
        assert (
            pause_requested_events[-1].data["resume_status"]
            == case["expected_pause"]["resume_status"]
        )
        assert pause_request["reason"] == case["expected_pause"]["reason"]
        assert pause_request["source"] == case["expected_pause"]["source"]
        assert pause_request["wait_id"] == case["expected_pause"]["wait_id"]
        assert pause_request["metadata"] == case["expected_pause"]["metadata"]
        raw_pause_request = case.get("pause_request")
        expected_interrupt = (
            bool(cast(dict[str, Any], raw_pause_request)["interrupt"])
            if isinstance(raw_pause_request, dict)
            else False
        )
        assert pause_request["interrupt"] is expected_interrupt
    if "expected_model_deltas" in case:
        assert [
            dict(event.data) for event in events if event.type == EventTypes.MODEL_DELTA
        ] == case["expected_model_deltas"]
    if "forbidden_event_types" in case:
        forbidden_events = set(cast(list[str], case["forbidden_event_types"]))
        assert not [event for event in events if event.type in forbidden_events]
    if "forbidden_checkpoint_statuses" in case:
        forbidden_statuses = set(cast(list[str], case["forbidden_checkpoint_statuses"]))
        checkpoint_statuses = [
            RunSnapshot.from_dict(event.data).state.status.value
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (forbidden_statuses & set(checkpoint_statuses))
    if "forbidden_checkpoint_tool_counts" in case:
        forbidden = set(cast(list[int], case["forbidden_checkpoint_tool_counts"]))
        checkpoint_counts = [
            RunSnapshot.from_dict(event.data).state.total_tool_calls
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (forbidden & set(checkpoint_counts))
    if "forbidden_checkpoint_status_tool_counts" in case:
        forbidden_pairs = {
            (
                expect_case_str(item["status"], "forbidden checkpoint status"),
                expect_case_int(item["total_tool_calls"], "forbidden checkpoint tool count"),
            )
            for item in cast(list[dict[str, Any]], case["forbidden_checkpoint_status_tool_counts"])
        }
        checkpoint_pairs = [
            (
                RunSnapshot.from_dict(event.data).state.status.value,
                RunSnapshot.from_dict(event.data).state.total_tool_calls,
            )
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (forbidden_pairs & set(checkpoint_pairs))
    if "forbidden_unpaused_checkpoint_tool_counts" in case:
        forbidden = set(cast(list[int], case["forbidden_unpaused_checkpoint_tool_counts"]))
        checkpoint_counts = [
            RunSnapshot.from_dict(event.data).state.total_tool_calls
            for event in events
            if event.type == EventTypes.CHECKPOINT
            and RunSnapshot.from_dict(event.data).state.status is not AgentStatus.PAUSED
        ]
        assert not (forbidden & set(checkpoint_counts))
    if "forbidden_checkpoint_message_roles" in case:
        forbidden_roles = [
            tuple(item)
            for item in cast(list[list[str]], case["forbidden_checkpoint_message_roles"])
        ]
        checkpoint_roles = [
            tuple(message.role for message in RunSnapshot.from_dict(event.data).state.messages)
            for event in events
            if event.type == EventTypes.CHECKPOINT
        ]
        assert not (set(forbidden_roles) & set(checkpoint_roles))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_cases(), ids=lambda case: str(case["name"]))
async def test_conformance_case(case: dict[str, Any]) -> None:
    if case.get("case_type") == "model_response_negative":
        with pytest.raises(
            (TypeError, ValueError, KeyError),
            match=expect_case_str(case["expected_error"], "expected_error"),
        ):
            ModelResponse.from_dict(cast(Mapping[str, Any], case["model_response"]))
        return

    if case.get("case_type") == "resume":
        await assert_resume_conformance_case(case)
        return

    steps = [model_response_from_case_step(step) for step in case["model_steps"]]
    stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
    expected_status = AgentStatus(case["expected_status"])
    result = await run_case_result(case, steps, stream_steps)
    events = await collect_case_events(case, steps, stream_steps)

    assert_run_case_expectations(case, result, events, expected_status)


async def assert_resume_conformance_case(case: dict[str, Any]) -> None:
    initial_result_steps = [model_response_from_case_step(step) for step in case["model_steps"]]
    initial_event_steps = [model_response_from_case_step(step) for step in case["model_steps"]]
    stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
    initial_result = await run_case_result(case, initial_result_steps, stream_steps)
    initial_events = await collect_case_events(case, initial_event_steps, stream_steps)
    initial_status = AgentStatus(case["expected_status"])

    assert_run_case_expectations(case, initial_result, initial_events, initial_status)
    snapshot = select_resume_snapshot(case, initial_events)
    assert_matches_schema(
        "selected resume snapshot",
        RUN_SNAPSHOT_SCHEMA_VALIDATOR,
        snapshot.to_dict(),
    )
    append_messages = messages_from_case(case.get("resume_append_messages", []))

    if "expected_resume_error" in case:
        selector = resume_selector_from_case(case)
        if selector is not None and str(case["expected_resume_error"]) == "does not match":
            assert_matches_schema(
                "schema-valid mismatched resume input",
                RESUME_INPUT_SCHEMA_VALIDATOR,
                {
                    "snapshot": snapshot.to_dict(),
                    "append_messages": [message.to_dict() for message in append_messages],
                    "expected_pause": selector.to_dict(),
                    "metadata": {},
                },
            )
        with pytest.raises(
            ValueError,
            match=expect_case_str(case["expected_resume_error"], "expected_resume_error"),
        ):
            ResumeInput(
                snapshot=snapshot,
                append_messages=append_messages,
                expected_pause=selector,
            )
        return

    resume_input = ResumeInput(
        snapshot=snapshot,
        append_messages=append_messages,
        expected_pause=resume_selector_from_case(case),
    )
    assert_matches_schema(
        "resume input",
        RESUME_INPUT_SCHEMA_VALIDATOR,
        resume_input.to_dict(),
    )

    resume_steps = [
        model_response_from_case_step(step)
        for step in cast(list[dict[str, Any]], case.get("resume_model_steps") or [])
    ]
    resume_event_steps = [
        model_response_from_case_step(step)
        for step in cast(list[dict[str, Any]], case.get("resume_model_steps") or [])
    ]
    result = await AgentLoop(
        model=model_from_case(case, resume_steps, [], pause_controller=None),
        tools=case_tools(),
        limits=limits_from_case(case),
    ).run_snapshot(resume_input)
    resume_events = await collect_resume_case_events(case, resume_input, resume_event_steps)

    expected_status = AgentStatus(case["expected_resume_status"])
    expected_tool_calls = expect_case_int(
        case.get("expected_resume_tool_calls", 0), "expected_resume_tool_calls"
    )
    assert result.status is expected_status
    assert result.total_tool_calls == expected_tool_calls
    assert resume_events[-1].data["state"]["total_tool_calls"] == expected_tool_calls
    assert_event_stream_invariants(resume_events, expected_status)
    assert result.trace is not None
    assert_matches_schema("resume run trace", RUN_TRACE_SCHEMA_VALIDATOR, result.trace.to_dict())
    assert replay_trace(result.trace).final_status is expected_status
    resume_event_trace = RunTrace.from_events(resume_events[0].run_id, resume_events)
    assert_matches_schema(
        "resume event trace", RUN_TRACE_SCHEMA_VALIDATOR, resume_event_trace.to_dict()
    )
    assert replay_trace(resume_event_trace).final_status is expected_status
    if result.snapshot is not None:
        assert_matches_schema(
            "resume result snapshot",
            RUN_SNAPSHOT_SCHEMA_VALIDATOR,
            result.snapshot.to_dict(),
        )
    if "expected_resume_trace_prefix" in case:
        assert [step.kind for step in result.trace.steps][
            : len(case["expected_resume_trace_prefix"])
        ] == case["expected_resume_trace_prefix"]
    if "expected_resume_final_text" in case:
        assert (
            "".join(part.text or "" for part in result.final_parts)
            == case["expected_resume_final_text"]
        )
    if "expected_resume_message_roles" in case:
        assert [message.role for message in result.messages] == case[
            "expected_resume_message_roles"
        ]
    if "expected_resume_tool_texts" in case:
        assert [message.text for message in result.messages if message.role == "tool"] == case[
            "expected_resume_tool_texts"
        ]
