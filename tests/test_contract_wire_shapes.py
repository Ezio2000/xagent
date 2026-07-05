from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from kernel import (
    AgentEvent,
    AgentState,
    AgentStatus,
    ApprovalDecision,
    ApprovalRequest,
    CheckpointSummary,
    ContentPart,
    EventTypes,
    JournalRecord,
    LoopLimits,
    Message,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PauseSelector,
    PauseState,
    ResponseFormat,
    ResumeInput,
    RunSnapshot,
    RuntimeContext,
    StoredCheckpoint,
    ToolAcceptance,
    ToolCall,
    ToolChoice,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "contracts" / "v0"
SPEC_BASE_URI = "https://agent-runtime.local/spec/v0"


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


def schema_validator(schema_name: str) -> Any:
    return Draft202012Validator(SCHEMAS[schema_name], registry=SCHEMA_REGISTRY)


def ref_validator(ref: str) -> Any:
    return Draft202012Validator({"$ref": ref}, registry=SCHEMA_REGISTRY)


def assert_matches_schema(label: str, validator: Any, instance: Mapping[str, Any]) -> None:
    errors = sorted(
        validator.iter_errors(instance),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "$"
    raise AssertionError(f"{label} schema violation at {path}: {error.message}") from error


def tool_spec() -> ToolSpec:
    return ToolSpec(
        name="search",
        description="Search indexed content.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        modes=("execute", "accept", "custom"),
        output_schema={"type": "object"},
        annotations={
            "parallel_safe": True,
            "read_only": True,
            "idempotent": True,
            "risk": {"network": "read", "requires_approval": False},
        },
        metadata={"owner": "tests"},
    )


def tool_call() -> ToolCall:
    return ToolCall(
        id="call-1",
        name="search",
        mode="execute",
        arguments={"query": "runtime"},
        metadata={"provider": "test"},
    )


def user_message() -> Message:
    return Message.user([ContentPart.text_part("hello")], metadata={"tenant": "acme"})


def runtime_context() -> RuntimeContext:
    return RuntimeContext(
        run_id="run-1",
        started_at=1.0,
        deadline=10.0,
        metadata={"tenant": "acme"},
        parent_run_id="parent-run",
        parent_tool_call_id="parent-call",
        run_kind="subagent",
    )


def planning_state() -> AgentState:
    return AgentState(
        status=AgentStatus.PLANNING,
        messages=[user_message()],
        total_usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
    )


def snapshot() -> RunSnapshot:
    context = runtime_context()
    context.sequence = 1
    return RunSnapshot(state=planning_state(), context=context)


def paused_snapshot() -> RunSnapshot:
    context = runtime_context()
    context.sequence = 2
    return RunSnapshot(
        state=AgentState(
            status=AgentStatus.PAUSED,
            messages=[user_message()],
            pause=PauseState(
                reason="external_wait",
                resume_status=AgentStatus.PLANNING,
                source="tool",
                wait_id="job-1",
                metadata={"job": "job-1"},
            ),
        ),
        context=context,
    )


def model_request() -> ModelRequest:
    return ModelRequest(
        messages=(user_message(),),
        tools=(tool_spec(),),
        options=ModelOptions(
            model="test-model",
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=128,
            stop_sequences=("END",),
            seed=7,
            metadata={"adapter": "test"},
        ),
        tool_choice=ToolChoice(mode="tool", name="search", allow_parallel_tool_calls=True),
        response_format=ResponseFormat(
            type="json_schema",
            json_schema={"type": "object"},
            strict=True,
        ),
        metadata={"request": "metadata"},
    )


def model_response() -> ModelResponse:
    return ModelResponse(
        parts=[ContentPart.text_part("done")],
        tool_calls=[tool_call()],
        finish_reason="tool_calls",
        usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        model="test-model",
        response_id="resp-1",
        metadata={"provider": "test"},
    )


def resume_input() -> ResumeInput:
    return ResumeInput(
        snapshot=paused_snapshot(),
        append_messages=[
            Message.external(
                [ContentPart.text_part("callback complete")],
                insert_id="insert-1",
                source="callback",
            )
        ],
        expected_pause=PauseSelector(reason="external_wait", wait_id="job-1"),
        metadata={"resumed_by": "test"},
    )


def stored_checkpoint() -> StoredCheckpoint:
    checkpoint = snapshot()
    return StoredCheckpoint(
        run_id=checkpoint.context.run_id,
        checkpoint_id="checkpoint-1",
        parent_checkpoint_id=None,
        sequence=checkpoint.context.sequence,
        status=checkpoint.state.status,
        snapshot=checkpoint,
        created_at=2.0,
        metadata={"store": "test"},
    )


def journal_record() -> JournalRecord:
    return JournalRecord(
        event=AgentEvent(
            EventTypes.MODEL_STARTED,
            {"iteration": 1},
            run_id="run-1",
            sequence=1,
            created_at=1.5,
        ),
        checkpoint_id=None,
        trace_step_id=1,
        payload_ref="trace://step-1",
        payload_hash="sha256:abc",
        metadata={"journal": "test"},
    )


def wire_shape_cases() -> list[tuple[str, Any, Mapping[str, Any]]]:
    return [
        ("message", schema_validator("messages.schema.json"), user_message().to_dict()),
        (
            "content_part",
            ref_validator(f"{SPEC_BASE_URI}/messages.schema.json#/$defs/content_part"),
            ContentPart.text_part("hello").to_dict(),
        ),
        (
            "tool_call",
            ref_validator(f"{SPEC_BASE_URI}/messages.schema.json#/$defs/tool_call"),
            tool_call().to_dict(),
        ),
        ("tool_spec", schema_validator("tools.schema.json"), tool_spec().to_dict()),
        (
            "tool_observation",
            schema_validator("tool-result.schema.json"),
            ToolObservation.text("result").to_dict(),
        ),
        (
            "tool_acceptance",
            schema_validator("tool-result.schema.json"),
            ToolAcceptance.text("accepted", correlation_id="job-1").to_dict(),
        ),
        (
            "tool_rejection",
            schema_validator("tool-result.schema.json"),
            ToolRejection.text("rejected", correlation_id="job-1").to_dict(),
        ),
        (
            "extension_tool_output",
            schema_validator("tool-result.schema.json"),
            ToolOutput(kind="custom_result", parts=[ContentPart.text_part("custom")]).to_dict(),
        ),
        ("model_request", schema_validator("model-request.schema.json"), model_request().to_dict()),
        (
            "model_response",
            schema_validator("model-response.schema.json"),
            model_response().to_dict(),
        ),
        (
            "model_error",
            schema_validator("model-error.schema.json"),
            {
                "message": "provider failed",
                "provider": "test",
                "code": "rate_limit",
                "status_code": 429,
                "retryable": True,
                "request_id": "req-1",
                "metadata": {"region": "test"},
            },
        ),
        ("limits", schema_validator("limits.schema.json"), LoopLimits().to_dict()),
        ("state", schema_validator("state.schema.json"), planning_state().to_dict()),
        (
            "runtime_context",
            schema_validator("runtime-context.schema.json"),
            runtime_context().to_dict(),
        ),
        ("run_snapshot", schema_validator("run-snapshot.schema.json"), snapshot().to_dict()),
        ("resume_input", schema_validator("resume-input.schema.json"), resume_input().to_dict()),
        (
            "event",
            schema_validator("events.schema.json"),
            AgentEvent(
                EventTypes.MODEL_STARTED,
                {"iteration": 1},
                run_id="run-1",
                sequence=1,
                created_at=1.5,
            ).to_dict(),
        ),
        (
            "approval_request",
            ref_validator(
                f"{SPEC_BASE_URI}/runtime-extensions.schema.json#/$defs/approval_request"
            ),
            ApprovalRequest(
                tool_call=tool_call(),
                context=runtime_context(),
                tool_spec=tool_spec(),
                risk={"network": "read"},
                metadata={"policy": "test"},
            ).to_dict(),
        ),
        (
            "approval_decision",
            ref_validator(
                f"{SPEC_BASE_URI}/runtime-extensions.schema.json#/$defs/approval_decision"
            ),
            ApprovalDecision.allow(metadata={"policy": "test"}).to_dict(),
        ),
        (
            "checkpoint_summary",
            ref_validator(
                f"{SPEC_BASE_URI}/runtime-extensions.schema.json#/$defs/checkpoint_summary"
            ),
            CheckpointSummary(
                run_id="run-1",
                checkpoint_id="checkpoint-1",
                parent_checkpoint_id=None,
                sequence=1,
                status=AgentStatus.PLANNING,
                created_at=2.0,
                metadata={"store": "test"},
            ).to_dict(),
        ),
        (
            "stored_checkpoint",
            ref_validator(
                f"{SPEC_BASE_URI}/runtime-extensions.schema.json#/$defs/stored_checkpoint"
            ),
            stored_checkpoint().to_dict(),
        ),
        (
            "journal_record",
            ref_validator(f"{SPEC_BASE_URI}/runtime-extensions.schema.json#/$defs/journal_record"),
            journal_record().to_dict(),
        ),
    ]


def test_public_dto_wire_shapes_match_contract_schemas() -> None:
    for label, validator, instance in wire_shape_cases():
        assert_matches_schema(label, validator, instance)
