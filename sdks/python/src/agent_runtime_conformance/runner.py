"""Shared conformance runner for the Python reference SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from agent_runtime.control import ConversationInsert, PauseRequest, RunController
from agent_runtime.events import AgentEvent, EventTypes
from agent_runtime.limits import LoopLimits
from agent_runtime.loop import AgentLoop, AgentResult
from agent_runtime.messages import ContentPart, Message
from agent_runtime.models import (
    ModelContentDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
)
from agent_runtime.resume import PauseSelector, ResumeInput
from agent_runtime.runtime import RuntimeContext
from agent_runtime.snapshot import RunSnapshot
from agent_runtime.state import AgentStatus
from agent_runtime.tools import (
    Tool,
    ToolAcceptance,
    ToolExecutionContext,
    ToolInvocation,
    ToolObservation,
    ToolRejection,
    ToolSpec,
)
from agent_runtime.trace import RunTrace, replay_trace

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
    "expected_event_types",
    "expected_trace_kinds",
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
    "mode",
    "seconds",
}
LIMIT_KEYS = {
    "max_iterations",
    "max_total_tool_calls",
    "timeout_seconds",
    "stop_on_tool_error",
    "max_parallel_tool_calls",
}
REQUIRED_SCHEMA_FILES = {
    "events.schema.json",
    "run-snapshot.schema.json",
    "run-trace.schema.json",
    "resume-input.schema.json",
    "messages.schema.json",
    "model-response.schema.json",
    "tools.schema.json",
    "tool-result.schema.json",
    "limits.schema.json",
}
STREAM_EVENT_REQUIRED_KEYS: dict[str, set[str]] = {
    "text_delta": {"index", "text_delta", "part_type"},
    "tool_call_delta": {"index"},
    "sleep": {"seconds"},
    "pause_request": set(),
}
REGISTRY_CLS: Any = Registry
RESOURCE_CLS: Any = Resource
DRAFT_2020_12_SPEC: Any = DRAFT202012


@dataclass(slots=True, frozen=True)
class ConformanceCaseResult:
    """Result for a single conformance case."""

    name: str
    case_type: str


@dataclass(slots=True)
class ConformanceValidators:
    event: Any
    run_snapshot: Any
    run_trace: Any
    resume_input: Any
    message: Any
    model_response: Any
    limits: Any


class ScriptedModel:
    def __init__(
        self,
        steps: Sequence[ModelResponse],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
    ) -> None:
        self._steps = list(steps)
        self._controller = controller
        self._pause_request_on_call = pause_request_on_call
        self._pause_request_on_stream_event = pause_request_on_stream_event
        self._conversation_insert_on_call = conversation_insert_on_call
        self._pause_requested = False
        self._conversation_inserted = False
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = request, context
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        response = self._steps[self.calls]
        self.calls += 1
        self._apply_conversation_insert_once(self._conversation_insert_on_call)
        self._apply_pause_once(self._pause_request_on_call)
        return response

    def _apply_pause_once(self, request: PauseRequest | None) -> None:
        if self._controller is not None and request is not None and not self._pause_requested:
            self._pause_requested = True
            apply_pause_request(self._controller, request)

    def _apply_conversation_insert_once(self, insert: ConversationInsert | None) -> None:
        if self._controller is not None and insert is not None and not self._conversation_inserted:
            self._conversation_inserted = True
            self._controller.insert(insert)


class StreamedCaseModel(ScriptedModel):
    def __init__(
        self,
        steps: Sequence[ModelResponse],
        stream_steps: Sequence[dict[str, Any]],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
    ) -> None:
        super().__init__(
            steps,
            controller=controller,
            pause_request_on_call=pause_request_on_call,
            pause_request_on_stream_event=pause_request_on_stream_event,
            conversation_insert_on_call=conversation_insert_on_call,
        )
        self._stream_steps = list(stream_steps)
        self.stream_calls = 0

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = request, context
        if self.stream_calls >= len(self._stream_steps):
            raise AssertionError("scripted stream model exhausted")

        step = self._stream_steps[self.stream_calls]
        self.stream_calls += 1
        self._apply_conversation_insert_once(self._conversation_insert_on_call)
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
                    mode=expect_case_optional_str(raw_event.get("mode"), "stream event mode"),
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

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.text(str(invocation.arguments.get("text", "")))


class AcceptTool:
    spec = ToolSpec(
        name="accept",
        description="Accept an external operation.",
        input_schema={"type": "object", "properties": {}},
        modes=("accept",),
    )

    async def accept(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolAcceptance | ToolRejection:
        _ = context
        if invocation.arguments.get("reject") is True:
            return ToolRejection.text(str(invocation.arguments.get("text", "rejected")))
        return ToolAcceptance.text(
            str(invocation.arguments.get("text", "accepted")),
            correlation_id=str(invocation.arguments.get("correlation_id", invocation.id)),
        )


class FailTool:
    spec = ToolSpec(
        name="fail",
        description="Raise an error.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = invocation, context
        raise RuntimeError("tool failed")


class DelayedEchoTool:
    spec = ToolSpec(
        name="delayed_echo",
        description="Return input text after an optional delay.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        await asyncio.sleep(float(invocation.arguments.get("delay", 0)))
        return ToolObservation.text(str(invocation.arguments.get("text", "")))


class WaitTool:
    spec = ToolSpec(
        name="wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        return ToolObservation.waiting(
            str(invocation.arguments.get("text", "external wait started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason=str(invocation.arguments.get("reason", "external_wait")),
        )


class ParallelWaitTool:
    spec = ToolSpec(
        name="parallel_wait",
        description="Start external work and pause the run.",
        input_schema={"type": "object", "properties": {}},
        annotations={"parallel_safe": True, "read_only": True, "idempotent": True},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        await asyncio.sleep(float(invocation.arguments.get("delay", 0)))
        return ToolObservation.waiting(
            str(invocation.arguments.get("text", "external wait started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason=str(invocation.arguments.get("reason", "external_wait")),
        )


class ConformanceRunner:
    """Load and execute shared conformance cases."""

    def __init__(self, *, cases_dir: Path, spec_dir: Path) -> None:
        self.cases_dir = cases_dir
        self.spec_dir = spec_dir
        self.validators = build_validators(spec_dir)

    def load_cases(self) -> list[dict[str, Any]]:
        if not self.cases_dir.is_dir():
            raise FileNotFoundError(f"conformance cases directory not found: {self.cases_dir}")
        cases: list[dict[str, Any]] = []
        paths = sorted(self.cases_dir.glob("*.json"))
        if not paths:
            raise ValueError(f"no conformance case JSON files found in {self.cases_dir}")
        for path in paths:
            case = load_json_object(path, "conformance case")
            try:
                self.validate_case_keys(str(path), case)
            except Exception as exc:
                raise ValueError(f"{path}: invalid conformance case: {exc}") from exc
            cases.append(case)
        return cases

    def validate_case_keys(self, name: str, case: dict[str, Any]) -> None:
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
            self.assert_matches_schema(
                f"{name}.model_response",
                self.validators.model_response,
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
            if "resume_checkpoint_status" not in case:
                raise KeyError(f"{name} missing required key: resume_checkpoint_status")
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
        if case.get("conversation_insert_timing") not in {None, "during_model_call"}:
            raise ValueError(f"{name} has invalid conversation_insert_timing")
        if "conversation_insert" in case:
            raw_insert = case["conversation_insert"]
            if not isinstance(raw_insert, dict):
                raise TypeError(f"{name}.conversation_insert must be an object")
            ConversationInsert.from_dict(cast(Mapping[str, Any], raw_insert))
            if case.get("conversation_insert_timing") != "during_model_call":
                raise ValueError(f"{name}.conversation_insert requires conversation_insert_timing")
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
            if "stop_on_tool_error" in limits and not isinstance(
                limits["stop_on_tool_error"], bool
            ):
                raise TypeError(f"{name}.limits.stop_on_tool_error must be a boolean")
            self.assert_matches_schema(f"{name}.limits", self.validators.limits, limits)
        for index, step in enumerate(expect_case_list(case["model_steps"], f"{name}.model_steps")):
            reject_unknown_keys(set(step), MODEL_STEP_KEYS, f"{name}.model_steps[{index}]")
            for required in MODEL_STEP_REQUIRED_KEYS:
                if required not in step:
                    raise KeyError(f"{name}.model_steps[{index}] missing required key: {required}")
            self.assert_matches_schema(
                f"{name}.model_steps[{index}]", self.validators.model_response, step
            )
        raw_resume_steps = case.get("resume_model_steps", [])
        for index, step in enumerate(
            expect_case_list(raw_resume_steps, f"{name}.resume_model_steps")
        ):
            reject_unknown_keys(set(step), MODEL_STEP_KEYS, f"{name}.resume_model_steps[{index}]")
            for required in MODEL_STEP_REQUIRED_KEYS:
                if required not in step:
                    raise KeyError(
                        f"{name}.resume_model_steps[{index}] missing required key: {required}"
                    )
            self.assert_matches_schema(
                f"{name}.resume_model_steps[{index}]", self.validators.model_response, step
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
        for index, step in enumerate(
            expect_case_list(raw_stream_steps, f"{name}.stream_model_steps")
        ):
            reject_unknown_keys(set(step), STREAM_STEP_KEYS, f"{name}.stream_model_steps[{index}]")
            if "events" not in step:
                raise KeyError(f"{name}.stream_model_steps[{index}] missing required key: events")
            for event_index, event in enumerate(
                expect_case_list(step["events"], f"{name}.stream_model_steps[{index}].events")
            ):
                self._validate_stream_event(name, index, event_index, event)
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
        for key in ("expected_event_types", "expected_trace_kinds", "forbidden_event_types"):
            if key in case:
                for item in expect_case_list_of_strings(case[key], f"{name}.{key}"):
                    if not item:
                        raise ValueError(f"{name}.{key} items must not be empty")

    def _validate_stream_event(
        self, name: str, step_index: int, event_index: int, event: dict[str, Any]
    ) -> None:
        label = f"{name}.stream_model_steps[{step_index}].events[{event_index}]"
        reject_unknown_keys(set(event), STREAM_EVENT_KEYS, label)
        if "type" not in event:
            raise KeyError(f"{label} missing required key: type")
        event_type = expect_case_str(event["type"], f"{name}.stream_model_steps event type")
        required = STREAM_EVENT_REQUIRED_KEYS.get(event_type)
        if required is None:
            raise ValueError(f"{label} has invalid type: {event_type}")
        for key in required:
            if key not in event:
                raise KeyError(f"{label} missing required key: {key}")
        if "index" in event:
            expect_case_int(event["index"], f"{label}.index")
        if "text_delta" in event:
            expect_case_str(event["text_delta"], f"{label}.text_delta")
        if "part_type" in event:
            expect_case_str(event["part_type"], f"{label}.part_type")
        if "seconds" in event:
            expect_case_number(event["seconds"], f"{label}.seconds")
        for optional_key in ("id", "name", "mode", "arguments_delta"):
            if optional_key in event:
                expect_case_optional_str(event[optional_key], f"{label}.{optional_key}")

    def assert_matches_schema(
        self,
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

    async def run_case(self, case: dict[str, Any]) -> ConformanceCaseResult:
        case_type = expect_case_str(case.get("case_type", "run"), "case_type")
        name = expect_case_str(case["name"], "case name")
        if case_type == "model_response_negative":
            try:
                ModelResponse.from_dict(cast(Mapping[str, Any], case["model_response"]))
            except (TypeError, ValueError, KeyError) as exc:
                expected = expect_case_str(case["expected_error"], "expected_error")
                if expected not in str(exc):
                    raise AssertionError(
                        f"{name} expected error containing {expected!r}, got {exc!r}"
                    ) from exc
                return ConformanceCaseResult(name=name, case_type=case_type)
            raise AssertionError(f"{name} expected model response rejection")

        if case_type == "resume":
            await self.assert_resume_conformance_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        steps = [model_response_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        expected_status = AgentStatus(case["expected_status"])
        result = await self.run_case_result(case, steps, stream_steps)
        events = await self.collect_case_events(case, steps, stream_steps)
        self.assert_run_case_expectations(case, result, events, expected_status)
        return ConformanceCaseResult(name=name, case_type=case_type)

    async def run_all(self) -> list[ConformanceCaseResult]:
        results: list[ConformanceCaseResult] = []
        for case in self.load_cases():
            results.append(await self.run_case(case))
        return results

    async def run_case_result(
        self,
        case: dict[str, Any],
        steps: Sequence[ModelResponse],
        stream_steps: Sequence[dict[str, Any]],
    ) -> AgentResult:
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller)
        return await AgentLoop(
            model=model,
            tools=case_tools(),
            limits=limits_from_case(case),
        ).run(
            [Message.user_text("run conformance case")],
            stream=bool(stream_steps),
            controller=controller,
        )

    async def collect_case_events(
        self,
        case: dict[str, Any],
        steps: Sequence[ModelResponse],
        stream_steps: Sequence[dict[str, Any]],
    ) -> list[AgentEvent]:
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller)
        return [
            event
            async for event in AgentLoop(
                model=model,
                tools=case_tools(),
                limits=limits_from_case(case),
            ).run_events(
                [Message.user_text("run conformance case")],
                stream=bool(stream_steps),
                controller=controller,
            )
        ]

    async def collect_resume_case_events(
        self,
        case: dict[str, Any],
        resume_input: ResumeInput,
        steps: Sequence[ModelResponse],
    ) -> list[AgentEvent]:
        return [
            event
            async for event in AgentLoop(
                model=model_from_case(case, steps, [], controller=None),
                tools=case_tools(),
                limits=limits_from_case(case),
            ).run_snapshot_events(resume_input)
        ]

    def assert_event_stream_invariants(
        self, events: Sequence[AgentEvent], expected: AgentStatus
    ) -> None:
        check(bool(events), "event stream is empty")
        check(events[0].type == EventTypes.RUN_STARTED, "first event must be run_started")
        check(events[-1].type == EventTypes.RUN_COMPLETED, "last event must be run_completed")
        check(
            events[-1].data["state"]["status"] == expected.value,
            f"run_completed status must be {expected.value}",
        )

        run_ids = {event.run_id for event in events}
        check(len(run_ids) == 1, "event stream must use one run_id")
        check(bool(next(iter(run_ids))), "event run_id must not be empty")

        sequences = [event.sequence for event in events]
        check(sequences == sorted(sequences), "event sequences must be sorted")
        check(len(sequences) == len(set(sequences)), "event sequences must be unique")

        terminal_state_changed_index: int | None = None
        checkpoint_after_terminal_index: int | None = None
        for index, event in enumerate(events):
            envelope = event.to_dict()
            self.assert_matches_schema(f"{event.type} event", self.validators.event, envelope)
            check(
                set(envelope)
                == {
                    "type",
                    "data",
                    "run_id",
                    "sequence",
                    "created_at",
                    "schema_version",
                },
                f"{event.type} event envelope keys mismatch",
            )
            check(isinstance(envelope["data"], dict), f"{event.type} data must be an object")

            if event.type == EventTypes.CHECKPOINT:
                snapshot = RunSnapshot.from_dict(event.data)
                self.assert_matches_schema(
                    f"{event.type} snapshot",
                    self.validators.run_snapshot,
                    snapshot.to_dict(),
                )
                check(
                    snapshot.context.run_id == event.run_id,
                    "checkpoint context run_id must match event run_id",
                )
                check(
                    snapshot.context.sequence == event.sequence,
                    "checkpoint context sequence must match event sequence",
                )
                if terminal_state_changed_index is not None and snapshot.state.status is expected:
                    checkpoint_after_terminal_index = index

            if event.type == EventTypes.STATE_CHANGED and event.data.get("to") == expected.value:
                terminal_state_changed_index = index

        if terminal_state_changed_index is None:
            raise AssertionError("missing terminal state_changed event")
        if checkpoint_after_terminal_index is None:
            raise AssertionError("missing terminal checkpoint event")
        terminal_index = terminal_state_changed_index
        checkpoint_index = checkpoint_after_terminal_index
        check(
            checkpoint_index > terminal_index,
            "terminal checkpoint must follow terminal state_changed",
        )

        if expected is AgentStatus.COMPLETED:
            final_indexes = [
                index for index, event in enumerate(events) if event.type == EventTypes.FINAL
            ]
            check(bool(final_indexes), "completed run must emit final")
            check(
                checkpoint_index < final_indexes[-1] < len(events) - 1,
                "final must appear between terminal checkpoint and run_completed",
            )
        elif expected is AgentStatus.PAUSED:
            pause_requested_indexes = [
                index
                for index, event in enumerate(events)
                if event.type == EventTypes.PAUSE_REQUESTED
            ]
            paused_indexes = [
                index for index, event in enumerate(events) if event.type == EventTypes.RUN_PAUSED
            ]
            check(bool(pause_requested_indexes), "paused run must emit pause_requested")
            check(bool(paused_indexes), "paused run must emit run_paused")
            check(
                pause_requested_indexes[-1] < terminal_index,
                "pause_requested must precede paused transition",
            )
            check(
                checkpoint_index < paused_indexes[-1] < len(events) - 1,
                "run_paused must appear between paused checkpoint and run_completed",
            )
            check(
                [event.type for event in events[pause_requested_indexes[-1] :]]
                == [
                    EventTypes.PAUSE_REQUESTED,
                    EventTypes.STATE_CHANGED,
                    EventTypes.CHECKPOINT,
                    EventTypes.RUN_PAUSED,
                    EventTypes.RUN_COMPLETED,
                ],
                "paused terminal tail mismatch",
            )
            check(
                events[paused_indexes[-1]].data["pause"]
                == events[checkpoint_index].data["state"]["pause"],
                "run_paused pause must match paused checkpoint",
            )
            check(
                not [event for event in events if event.type == EventTypes.ERROR],
                "paused run must not emit error",
            )
        else:
            error_indexes = [
                index for index, event in enumerate(events) if event.type == EventTypes.ERROR
            ]
            check(bool(error_indexes), "failed/limit run must emit error")
            check(
                checkpoint_index < error_indexes[-1] < len(events) - 1,
                "error must appear between terminal checkpoint and run_completed",
            )

    def assert_run_case_expectations(
        self,
        case: dict[str, Any],
        result: AgentResult,
        events: Sequence[AgentEvent],
        expected_status: AgentStatus,
    ) -> None:
        check(result.status is expected_status, f"result status must be {expected_status.value}")
        check(
            result.total_tool_calls == case["expected_tool_calls"],
            f"expected {case['expected_tool_calls']} tool calls, got {result.total_tool_calls}",
        )
        self.assert_event_stream_invariants(events, expected_status)
        trace = result.trace
        if trace is None:
            raise AssertionError("result trace is missing")
        self.assert_matches_schema("run trace", self.validators.run_trace, trace.to_dict())
        check(
            replay_trace(trace).final_status is expected_status,
            "result trace final status mismatch",
        )
        event_trace = RunTrace.from_events(events[0].run_id, events)
        self.assert_matches_schema("event trace", self.validators.run_trace, event_trace.to_dict())
        check(
            replay_trace(event_trace).final_status is expected_status,
            "event trace final status mismatch",
        )
        if "expected_event_types" in case:
            expected_event_types = expect_case_list_of_strings(
                case["expected_event_types"], "expected_event_types"
            )
            actual_event_types = [event.type for event in events]
            missing = [
                event_type
                for event_type in expected_event_types
                if event_type not in actual_event_types
            ]
            check(not missing, f"missing expected event type(s): {missing}")
        if "expected_trace_kinds" in case:
            expected_trace_kinds = expect_case_list_of_strings(
                case["expected_trace_kinds"], "expected_trace_kinds"
            )
            result_kinds = [step.kind for step in trace.steps]
            event_kinds = [step.kind for step in event_trace.steps]
            missing_result = [kind for kind in expected_trace_kinds if kind not in result_kinds]
            missing_event = [kind for kind in expected_trace_kinds if kind not in event_kinds]
            check(not missing_result, f"result trace missing expected kind(s): {missing_result}")
            check(not missing_event, f"event trace missing expected kind(s): {missing_event}")
        if result.snapshot is not None:
            self.assert_matches_schema(
                "result snapshot",
                self.validators.run_snapshot,
                result.snapshot.to_dict(),
            )
        if "expected_message_roles" in case:
            snapshot = result.snapshot
            if snapshot is None:
                raise AssertionError("expected message roles require a result snapshot")
            actual_roles = [message.role for message in snapshot.state.messages]
            check(
                actual_roles == case["expected_message_roles"],
                f"expected message roles {case['expected_message_roles']}, got {actual_roles}",
            )
        if "expected_final_text" in case:
            actual_final_text = "".join(part.text or "" for part in result.final_parts)
            check(
                actual_final_text == case["expected_final_text"],
                f"expected final text {case['expected_final_text']!r}, got {actual_final_text!r}",
            )
        if "expected_tool_texts" in case:
            actual_tool_texts = [
                message.text for message in result.messages if message.role == "tool"
            ]
            check(
                actual_tool_texts == case["expected_tool_texts"],
                f"expected tool texts {case['expected_tool_texts']}, got {actual_tool_texts}",
            )
        if "expected_pending_tool_call_ids" in case:
            snapshot = result.snapshot
            if snapshot is None:
                raise AssertionError("expected pending tool calls require a result snapshot")
            actual_pending_ids = [call.id for call in snapshot.state.pending_tool_calls]
            check(
                actual_pending_ids == case["expected_pending_tool_call_ids"],
                f"expected pending tool call ids {case['expected_pending_tool_call_ids']}, "
                f"got {actual_pending_ids}",
            )
        if "expected_pause" in case:
            self._assert_expected_pause(case, result, events)
        if "expected_model_deltas" in case:
            actual_deltas = [
                dict(event.data) for event in events if event.type == EventTypes.MODEL_DELTA
            ]
            check(
                actual_deltas == case["expected_model_deltas"],
                f"expected model deltas {case['expected_model_deltas']}, got {actual_deltas}",
            )
        self._assert_forbidden_expectations(case, events)

    def _assert_expected_pause(
        self, case: dict[str, Any], result: AgentResult, events: Sequence[AgentEvent]
    ) -> None:
        snapshot = result.snapshot
        if snapshot is None:
            raise AssertionError("expected pause requires a result snapshot")
        if snapshot.state.pause is None:
            raise AssertionError("expected pause but result snapshot has no pause")
        check(
            snapshot.state.pause.to_dict() == case["expected_pause"],
            f"expected pause {case['expected_pause']}, got {snapshot.state.pause.to_dict()}",
        )
        paused_events = [event for event in events if event.type == EventTypes.RUN_PAUSED]
        check(bool(paused_events), "expected pause but no run_paused event was emitted")
        check(
            paused_events[-1].data["pause"] == case["expected_pause"],
            "run_paused payload does not match expected pause",
        )
        pause_requested_events = [
            event for event in events if event.type == EventTypes.PAUSE_REQUESTED
        ]
        check(
            bool(pause_requested_events),
            "expected pause but no pause_requested event was emitted",
        )
        pause_request = pause_requested_events[-1].data["request"]
        check(
            pause_requested_events[-1].data["resume_status"]
            == case["expected_pause"]["resume_status"],
            "pause_requested resume_status does not match expected pause",
        )
        for key in ("reason", "source", "wait_id", "metadata"):
            check(
                pause_request[key] == case["expected_pause"][key],
                f"pause_requested {key} expected {case['expected_pause'][key]!r}, "
                f"got {pause_request[key]!r}",
            )
        raw_pause_request = case.get("pause_request")
        expected_interrupt = (
            bool(cast(dict[str, Any], raw_pause_request)["interrupt"])
            if isinstance(raw_pause_request, dict)
            else False
        )
        check(
            pause_request["interrupt"] is expected_interrupt,
            f"pause_requested interrupt expected {expected_interrupt}, "
            f"got {pause_request['interrupt']}",
        )

    def _assert_forbidden_expectations(
        self, case: dict[str, Any], events: Sequence[AgentEvent]
    ) -> None:
        if "forbidden_event_types" in case:
            forbidden_events = set(cast(list[str], case["forbidden_event_types"]))
            actual = [event.type for event in events if event.type in forbidden_events]
            check(not actual, f"forbidden event type(s) emitted: {actual}")
        if "forbidden_checkpoint_statuses" in case:
            forbidden_statuses = set(cast(list[str], case["forbidden_checkpoint_statuses"]))
            checkpoint_statuses = [
                RunSnapshot.from_dict(event.data).state.status.value
                for event in events
                if event.type == EventTypes.CHECKPOINT
            ]
            actual = forbidden_statuses & set(checkpoint_statuses)
            check(not actual, f"forbidden checkpoint status(es) emitted: {sorted(actual)}")
        if "forbidden_checkpoint_tool_counts" in case:
            forbidden = set(cast(list[int], case["forbidden_checkpoint_tool_counts"]))
            checkpoint_counts = [
                RunSnapshot.from_dict(event.data).state.total_tool_calls
                for event in events
                if event.type == EventTypes.CHECKPOINT
            ]
            actual = forbidden & set(checkpoint_counts)
            check(not actual, f"forbidden checkpoint tool count(s) emitted: {sorted(actual)}")
        if "forbidden_checkpoint_status_tool_counts" in case:
            forbidden_pairs = {
                (
                    expect_case_str(item["status"], "forbidden checkpoint status"),
                    expect_case_int(item["total_tool_calls"], "forbidden checkpoint tool count"),
                )
                for item in cast(
                    list[dict[str, Any]], case["forbidden_checkpoint_status_tool_counts"]
                )
            }
            checkpoint_pairs = [
                (
                    RunSnapshot.from_dict(event.data).state.status.value,
                    RunSnapshot.from_dict(event.data).state.total_tool_calls,
                )
                for event in events
                if event.type == EventTypes.CHECKPOINT
            ]
            actual = forbidden_pairs & set(checkpoint_pairs)
            check(not actual, f"forbidden checkpoint status/tool pairs emitted: {sorted(actual)}")
        if "forbidden_unpaused_checkpoint_tool_counts" in case:
            forbidden = set(cast(list[int], case["forbidden_unpaused_checkpoint_tool_counts"]))
            checkpoint_counts = [
                RunSnapshot.from_dict(event.data).state.total_tool_calls
                for event in events
                if event.type == EventTypes.CHECKPOINT
                and RunSnapshot.from_dict(event.data).state.status is not AgentStatus.PAUSED
            ]
            actual = forbidden & set(checkpoint_counts)
            check(
                not actual,
                f"forbidden unpaused checkpoint tool count(s) emitted: {sorted(actual)}",
            )
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
            actual = set(forbidden_roles) & set(checkpoint_roles)
            check(not actual, f"forbidden checkpoint message roles emitted: {sorted(actual)}")

    async def assert_resume_conformance_case(self, case: dict[str, Any]) -> None:
        initial_result_steps = [model_response_from_case_step(step) for step in case["model_steps"]]
        initial_event_steps = [model_response_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        initial_result = await self.run_case_result(case, initial_result_steps, stream_steps)
        initial_events = await self.collect_case_events(case, initial_event_steps, stream_steps)
        initial_status = AgentStatus(case["expected_status"])

        self.assert_run_case_expectations(case, initial_result, initial_events, initial_status)
        snapshot = select_resume_snapshot(case, initial_events)
        self.assert_matches_schema(
            "selected resume snapshot",
            self.validators.run_snapshot,
            snapshot.to_dict(),
        )
        append_messages = messages_from_case(case.get("resume_append_messages", []))

        if "expected_resume_error" in case:
            selector = resume_selector_from_case(case)
            if selector is not None and str(case["expected_resume_error"]) == "does not match":
                self.assert_matches_schema(
                    "schema-valid mismatched resume input",
                    self.validators.resume_input,
                    {
                        "snapshot": snapshot.to_dict(),
                        "append_messages": [message.to_dict() for message in append_messages],
                        "expected_pause": selector.to_dict(),
                        "metadata": {},
                    },
                )
            try:
                ResumeInput(
                    snapshot=snapshot,
                    append_messages=append_messages,
                    expected_pause=selector,
                )
            except ValueError as exc:
                expected_error = expect_case_str(
                    case["expected_resume_error"], "expected_resume_error"
                )
                if expected_error not in str(exc):
                    raise AssertionError(
                        f"expected resume error containing {expected_error!r}, got {exc!r}"
                    ) from exc
                return
            raise AssertionError("expected resume input rejection")

        resume_input = ResumeInput(
            snapshot=snapshot,
            append_messages=append_messages,
            expected_pause=resume_selector_from_case(case),
        )
        self.assert_matches_schema(
            "resume input",
            self.validators.resume_input,
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
            model=model_from_case(case, resume_steps, [], controller=None),
            tools=case_tools(),
            limits=limits_from_case(case),
        ).run_snapshot(resume_input)
        resume_events = await self.collect_resume_case_events(
            case, resume_input, resume_event_steps
        )

        expected_status = AgentStatus(case["expected_resume_status"])
        expected_tool_calls = expect_case_int(
            case.get("expected_resume_tool_calls", 0), "expected_resume_tool_calls"
        )
        check(result.status is expected_status, f"resume status must be {expected_status.value}")
        check(
            result.total_tool_calls == expected_tool_calls,
            f"expected {expected_tool_calls} resume tool calls, got {result.total_tool_calls}",
        )
        check(
            resume_events[-1].data["state"]["total_tool_calls"] == expected_tool_calls,
            "resume run_completed total_tool_calls mismatch",
        )
        self.assert_event_stream_invariants(resume_events, expected_status)
        trace = result.trace
        if trace is None:
            raise AssertionError("resume result trace is missing")
        self.assert_matches_schema("resume run trace", self.validators.run_trace, trace.to_dict())
        check(
            replay_trace(trace).final_status is expected_status,
            "resume result trace final status mismatch",
        )
        resume_event_trace = RunTrace.from_events(resume_events[0].run_id, resume_events)
        self.assert_matches_schema(
            "resume event trace", self.validators.run_trace, resume_event_trace.to_dict()
        )
        check(
            replay_trace(resume_event_trace).final_status is expected_status,
            "resume event trace final status mismatch",
        )
        if result.snapshot is not None:
            self.assert_matches_schema(
                "resume result snapshot",
                self.validators.run_snapshot,
                result.snapshot.to_dict(),
            )
        if "expected_resume_trace_prefix" in case:
            actual_prefix = [step.kind for step in trace.steps][
                : len(case["expected_resume_trace_prefix"])
            ]
            check(
                actual_prefix == case["expected_resume_trace_prefix"],
                f"expected resume trace prefix {case['expected_resume_trace_prefix']}, "
                f"got {actual_prefix}",
            )
        if "expected_resume_final_text" in case:
            actual_final_text = "".join(part.text or "" for part in result.final_parts)
            check(
                actual_final_text == case["expected_resume_final_text"],
                f"expected resume final text {case['expected_resume_final_text']!r}, "
                f"got {actual_final_text!r}",
            )
        if "expected_resume_message_roles" in case:
            actual_roles = [message.role for message in result.messages]
            check(
                actual_roles == case["expected_resume_message_roles"],
                f"expected resume message roles {case['expected_resume_message_roles']}, "
                f"got {actual_roles}",
            )
        if "expected_resume_tool_texts" in case:
            actual_tool_texts = [
                message.text for message in result.messages if message.role == "tool"
            ]
            check(
                actual_tool_texts == case["expected_resume_tool_texts"],
                f"expected resume tool texts {case['expected_resume_tool_texts']}, "
                f"got {actual_tool_texts}",
            )


def load_json_schema(path: Path) -> dict[str, Any]:
    schema = load_json_object(path, "schema")
    schema_id = schema.get("$id")
    if not isinstance(schema_id, str) or not schema_id:
        raise ValueError(f"{path}: schema must define a non-empty $id")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{path}: invalid JSON schema: {exc.message}") from exc
    return schema


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text()
    except OSError as exc:
        raise OSError(f"failed to read {label} {path}: {exc}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path}: {label} must contain an object")
    return cast(dict[str, Any], value)


def build_validators(spec_dir: Path) -> ConformanceValidators:
    if not spec_dir.is_dir():
        raise FileNotFoundError(f"spec directory not found: {spec_dir}")
    schemas = {path.name: load_json_schema(path) for path in sorted(spec_dir.glob("*.schema.json"))}
    missing = REQUIRED_SCHEMA_FILES - set(schemas)
    if missing:
        raise ValueError(f"spec directory missing schema file(s): {', '.join(sorted(missing))}")
    registry: Any = REGISTRY_CLS().with_resources(
        [
            (
                cast(str, schema["$id"]),
                RESOURCE_CLS.from_contents(
                    cast(Any, schema), default_specification=DRAFT_2020_12_SPEC
                ),
            )
            for schema in schemas.values()
        ]
    )
    return ConformanceValidators(
        event=Draft202012Validator(schemas["events.schema.json"], registry=registry),
        run_snapshot=Draft202012Validator(schemas["run-snapshot.schema.json"], registry=registry),
        run_trace=Draft202012Validator(schemas["run-trace.schema.json"], registry=registry),
        resume_input=Draft202012Validator(schemas["resume-input.schema.json"], registry=registry),
        message=Draft202012Validator(schemas["messages.schema.json"], registry=registry),
        model_response=Draft202012Validator(
            schemas["model-response.schema.json"], registry=registry
        ),
        limits=Draft202012Validator(schemas["limits.schema.json"], registry=registry),
    )


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


def content_part_from_case(part: dict[str, Any]) -> ContentPart:
    return ContentPart.from_dict(part)


def model_response_from_case_step(step: dict[str, Any]) -> ModelResponse:
    return ModelResponse.from_dict(step)


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


def conversation_insert_from_case(case: dict[str, Any]) -> ConversationInsert | None:
    raw_insert_obj = case.get("conversation_insert")
    if not isinstance(raw_insert_obj, dict):
        return None
    return ConversationInsert.from_dict(cast(Mapping[str, Any], raw_insert_obj))


def apply_pause_request(controller: RunController, request: PauseRequest) -> None:
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


def controller_from_case(case: dict[str, Any]) -> RunController | None:
    request = pause_request_from_case(case)
    insert = conversation_insert_from_case(case)
    if request is None and insert is None:
        return None
    controller = RunController()
    if request is not None and case.get("pause_request_timing") not in {
        "during_model_call",
        "stream_event",
    }:
        apply_pause_request(controller, request)
    return controller


def model_from_case(
    case: dict[str, Any],
    steps: Sequence[ModelResponse],
    stream_steps: Sequence[dict[str, Any]],
    controller: RunController | None,
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
    conversation_insert_on_call = (
        conversation_insert_from_case(case)
        if case.get("conversation_insert_timing") == "during_model_call"
        else None
    )
    if stream_steps:
        return StreamedCaseModel(
            steps,
            stream_steps,
            controller=controller,
            pause_request_on_call=pause_request_on_call,
            pause_request_on_stream_event=pause_request_on_stream_event,
            conversation_insert_on_call=conversation_insert_on_call,
        )
    return ScriptedModel(
        steps,
        controller=controller,
        pause_request_on_call=pause_request_on_call,
        pause_request_on_stream_event=pause_request_on_stream_event,
        conversation_insert_on_call=conversation_insert_on_call,
    )


def case_tools() -> list[Tool]:
    return [EchoTool(), AcceptTool(), FailTool(), DelayedEchoTool(), WaitTool(), ParallelWaitTool()]


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


def infer_spec_dir(cases_dir: Path) -> Path:
    if cases_dir.name == "cases" and cases_dir.parent.name == "conformance":
        return cases_dir.parent.parent / "spec" / "v0"
    return Path.cwd() / "spec" / "v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run agent-runtime conformance cases.")
    parser.add_argument("cases_dir", type=Path, help="Directory containing conformance case JSON")
    parser.add_argument(
        "--spec-dir",
        type=Path,
        help="Directory containing spec/v0 schema JSON. Defaults to the repository layout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print failures and the final summary.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print Python tracebacks for failing cases.",
    )
    return parser


async def _run_cli_cases(runner: ConformanceRunner, *, quiet: bool, show_tracebacks: bool) -> int:
    cases = runner.load_cases()
    passed = 0
    failed = 0
    for case in cases:
        name = expect_case_str(case["name"], "case name")
        try:
            result = await runner.run_case(case)
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            if show_tracebacks:
                traceback.print_exception(exc)
            continue
        passed += 1
        if not quiet:
            print(f"PASS {result.name} [{result.case_type}]")
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cases_dir = args.cases_dir.resolve()
    spec_dir = args.spec_dir.resolve() if args.spec_dir is not None else infer_spec_dir(cases_dir)
    try:
        runner = ConformanceRunner(cases_dir=cases_dir, spec_dir=spec_dir)
        return asyncio.run(
            _run_cli_cases(
                runner,
                quiet=cast(bool, args.quiet),
                show_tracebacks=cast(bool, args.traceback),
            )
        )
    except Exception as exc:
        print(f"FAIL load: {exc}")
        if cast(bool, args.traceback):
            traceback.print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
