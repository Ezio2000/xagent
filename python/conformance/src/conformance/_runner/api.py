"""Shared conformance runner for the Python reference SDK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from diagnostics import RunTrace, replay_trace
from kernel import (
    AgentEvent,
    AgentLoop,
    AgentResult,
    AgentStatus,
    ApprovalDecision,
    ContentPart,
    ConversationInsert,
    EventTypes,
    JournalRecord,
    Message,
    ModelResponse,
    PauseSelector,
    ResumeInput,
    RunSnapshot,
    RuntimeContext,
    StoredCheckpoint,
)
from prompting import user_text
from support import (
    FailingCheckpointJournal,
    FailingRunStore,
    MemoryRunJournal,
    MemoryRunStore,
    ModelStep,
)

from conformance._case import (
    APPROVAL_DECISION_KEYS,
    APPROVAL_REQUEST_EXPECTATION_KEYS,
    CASE_KEYS,
    LIMIT_KEYS,
    MODEL_STEP_KEYS,
    MODEL_STEP_RESPONSE_KEYS,
    NEGATIVE_CASE_TYPES,
    STREAM_EVENT_KEYS_BY_TYPE,
    STREAM_EVENT_REQUIRED_KEYS,
    STREAM_STEP_KEYS,
    check,
    expect_case_int,
    expect_case_list,
    expect_case_list_of_strings,
    expect_case_mapping,
    expect_case_number,
    expect_case_optional_int,
    expect_case_optional_str,
    expect_case_str,
    reject_unknown_keys,
)
from conformance._fixtures import (
    approval_metadata_from_case,
    approval_policy_from_case,
    case_tools,
    controller_from_case,
    hooks_from_case,
    limits_from_case,
    messages_from_case,
    model_from_case,
    model_step_from_case_step,
    resume_selector_from_case,
    runtime_context_from_case,
    select_resume_snapshot,
)
from conformance._schemas import (
    assert_validator_matches,
    build_case_validator,
    build_validators,
    load_json_object,
)


@dataclass(slots=True, frozen=True)
class ConformanceCaseResult:
    """Result for a single conformance case."""

    name: str
    case_type: str


class ConformanceRunner:
    """Load and execute shared conformance cases."""

    def __init__(
        self,
        *,
        cases_dir: Path,
        spec_dir: Path,
        case_schema_path: Path | None = None,
    ) -> None:
        self.cases_dir = cases_dir
        self.spec_dir = spec_dir
        self.case_schema_path = case_schema_path or self._infer_case_schema_path(
            cases_dir, spec_dir
        )
        self.validators = build_validators(spec_dir)
        self.case_validator: Any = build_case_validator(spec_dir, self.case_schema_path)

    @staticmethod
    def _infer_case_schema_path(cases_dir: Path, spec_dir: Path) -> Path:
        if cases_dir.name == "cases" and cases_dir.parent.name == "conformance":
            return cases_dir.parent / "case.schema.json"
        return spec_dir.parent.parent / "conformance" / "case.schema.json"

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
                self.assert_matches_schema(str(path), self.case_validator, case)
                self.validate_case_keys(str(path), case)
            except Exception as exc:
                raise ValueError(f"{path}: invalid conformance case: {exc}") from exc
            cases.append(case)
        return cases

    def validate_case_keys(self, name: str, case: dict[str, Any]) -> None:
        reject_unknown_keys(set(case), CASE_KEYS, name)
        case_type = expect_case_str(case.get("case_type", "run"), f"{name}.case_type")
        if case_type not in {
            "run",
            "resume",
            "run_store_failure",
            "run_journal_failure",
            "run_store_journal",
            "run_store_resume_journal",
            *NEGATIVE_CASE_TYPES,
        }:
            raise ValueError(f"{name} has invalid case_type")
        if case_type == "model_response_negative":
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
        if case_type == "message_negative":
            for required in ("name", "message", "expected_error"):
                if required not in case:
                    raise KeyError(f"{name} missing required key: {required}")
            expect_case_str(case["name"], f"{name}.name")
            expect_case_str(case["expected_error"], f"{name}.expected_error")
            self.assert_matches_schema(
                f"{name}.message",
                self.validators.message,
                cast(Mapping[str, Any], case["message"]),
            )
            forbidden = set(case) - {"name", "case_type", "message", "expected_error"}
            if forbidden:
                raise AssertionError(
                    f"{name} has invalid negative-case key(s): {', '.join(sorted(forbidden))}"
                )
            return
        negative_only_keys = {"message", "model_response"} & set(case)
        if (
            case_type not in {"run_store_failure", "run_journal_failure"}
            and "expected_error" in case
        ):
            negative_only_keys.add("expected_error")
        if negative_only_keys:
            raise AssertionError(
                f"{name} has negative-only key(s): {', '.join(sorted(negative_only_keys))}"
            )
        unsupported_expectations = self._unsupported_expectation_keys(case_type) & set(case)
        if unsupported_expectations:
            raise AssertionError(
                f"{name} has unsupported expectation key(s) for {case_type}: "
                f"{', '.join(sorted(unsupported_expectations))}"
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
        if case_type not in {"resume", "run_store_resume_journal"}:
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
        if "retry_model_errors" in case and not isinstance(case["retry_model_errors"], bool):
            raise TypeError(f"{name}.retry_model_errors must be a boolean")
        if "approval_decisions" in case:
            raw_decisions_obj = case["approval_decisions"]
            raw_decisions = cast(Mapping[str, object], raw_decisions_obj)
            if not isinstance(raw_decisions_obj, dict):
                raise TypeError(f"{name}.approval_decisions must be an object")
            for call_id, raw_decision in raw_decisions.items():
                expect_case_str(call_id, f"{name}.approval_decisions key")
                if not isinstance(raw_decision, dict):
                    raise TypeError(f"{name}.approval_decisions.{call_id} must be an object")
                decision = cast(Mapping[str, Any], raw_decision)
                reject_unknown_keys(
                    set(decision),
                    APPROVAL_DECISION_KEYS,
                    f"{name}.approval_decisions.{call_id}",
                )
                ApprovalDecision.from_dict(decision)
                self.assert_matches_schema(
                    f"{name}.approval_decisions.{call_id}",
                    self.validators.approval_decision,
                    decision,
                )
        if "approval_metadata" in case:
            expect_case_mapping(case["approval_metadata"], f"{name}.approval_metadata")
        if "expected_approval_requests" in case:
            if "approval_decisions" not in case:
                raise ValueError(f"{name}.expected_approval_requests requires approval_decisions")
            raw_requests_obj = case["expected_approval_requests"]
            raw_requests = cast(Mapping[str, object], raw_requests_obj)
            if not isinstance(raw_requests_obj, dict):
                raise TypeError(f"{name}.expected_approval_requests must be an object")
            for call_id, raw_request in raw_requests.items():
                expect_case_str(call_id, f"{name}.expected_approval_requests key")
                if not isinstance(raw_request, dict):
                    raise TypeError(
                        f"{name}.expected_approval_requests.{call_id} must be an object"
                    )
                expected_request = cast(Mapping[str, Any], raw_request)
                reject_unknown_keys(
                    set(expected_request),
                    APPROVAL_REQUEST_EXPECTATION_KEYS,
                    f"{name}.expected_approval_requests.{call_id}",
                )
                if "risk" in expected_request:
                    expect_case_mapping(
                        expected_request["risk"],
                        f"{name}.expected_approval_requests.{call_id}.risk",
                    )
                if "metadata" in expected_request:
                    expect_case_mapping(
                        expected_request["metadata"],
                        f"{name}.expected_approval_requests.{call_id}.metadata",
                    )
        if "runtime_context" in case:
            raw_context = expect_case_mapping(case["runtime_context"], f"{name}.runtime_context")
            self.assert_matches_schema(
                f"{name}.runtime_context",
                self.validators.runtime_context,
                raw_context,
            )
            RuntimeContext.from_dict(raw_context)
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
        if case_type == "run_store_failure":
            if "expected_error" not in case:
                raise KeyError(f"{name} missing required key: expected_error")
            expect_case_str(case["expected_error"], f"{name}.expected_error")
        if case_type == "run_journal_failure":
            if "expected_error" not in case:
                raise KeyError(f"{name} missing required key: expected_error")
            expect_case_str(case["expected_error"], f"{name}.expected_error")
        if case_type == "run_store_resume_journal":
            if "resume_checkpoint_status" not in case:
                raise KeyError(f"{name} missing required key: resume_checkpoint_status")
            expect_case_str(case["resume_checkpoint_status"], f"{name}.resume_checkpoint_status")
            expect_case_optional_int(
                case.get("resume_checkpoint_total_tool_calls"),
                f"{name}.resume_checkpoint_total_tool_calls",
            )
            if "expected_resume_status" not in case:
                raise KeyError(f"{name} missing required key: expected_resume_status")
            expect_case_str(case["expected_resume_status"], f"{name}.expected_resume_status")
            if "expected_resume_tool_calls" in case:
                expect_case_int(
                    case["expected_resume_tool_calls"], f"{name}.expected_resume_tool_calls"
                )
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
            if "timeout_seconds" in limits:
                timeout_seconds = limits["timeout_seconds"]
                if timeout_seconds is not None and (
                    not isinstance(timeout_seconds, int | float)
                    or isinstance(timeout_seconds, bool)
                ):
                    raise TypeError(f"{name}.limits.timeout_seconds must be a number or null")
            if "stop_on_tool_error" in limits and not isinstance(
                limits["stop_on_tool_error"], bool
            ):
                raise TypeError(f"{name}.limits.stop_on_tool_error must be a boolean")
            self.assert_matches_schema(f"{name}.limits", self.validators.limits, limits)
        for index, step in enumerate(expect_case_list(case["model_steps"], f"{name}.model_steps")):
            self._validate_model_step(name, index, step, "model_steps")
        raw_resume_steps = case.get("resume_model_steps", [])
        for index, step in enumerate(
            expect_case_list(raw_resume_steps, f"{name}.resume_model_steps")
        ):
            self._validate_model_step(name, index, step, "resume_model_steps")
        raw_resume_messages = case.get("resume_append_messages", [])
        for index, message in enumerate(
            expect_case_list(raw_resume_messages, f"{name}.resume_append_messages")
        ):
            self.assert_matches_schema(
                f"{name}.resume_append_messages[{index}]",
                self.validators.message,
                message,
            )
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
        for key in (
            "expected_event_types",
            "expected_trace_kinds",
            "expected_tool_text_contains",
            "expected_final_part_types",
            "forbidden_event_types",
            "forbidden_journal_event_types",
        ):
            if key in case:
                for item in expect_case_list_of_strings(case[key], f"{name}.{key}"):
                    if not item:
                        raise ValueError(f"{name}.{key} items must not be empty")
        if "expected_tool_progress" in case:
            for index, item in enumerate(
                expect_case_list(case["expected_tool_progress"], f"{name}.expected_tool_progress")
            ):
                expect_case_mapping(item, f"{name}.expected_tool_progress[{index}]")
        if "expected_final_parts" in case:
            for index, item in enumerate(
                expect_case_list(case["expected_final_parts"], f"{name}.expected_final_parts")
            ):
                part = expect_case_mapping(item, f"{name}.expected_final_parts[{index}]")
                ContentPart.from_dict(part)
        if "expected_child_run" in case:
            raw_child_run = expect_case_mapping(
                case["expected_child_run"], f"{name}.expected_child_run"
            )
            allowed = {"parent_run_id", "parent_tool_call_id", "run_kind"}
            reject_unknown_keys(set(raw_child_run), allowed, f"{name}.expected_child_run")
            expect_case_str(
                raw_child_run["parent_run_id"],
                f"{name}.expected_child_run.parent_run_id",
            )
            if "parent_tool_call_id" in raw_child_run:
                expect_case_str(
                    raw_child_run["parent_tool_call_id"],
                    f"{name}.expected_child_run.parent_tool_call_id",
                )
            if "run_kind" in raw_child_run:
                expect_case_str(raw_child_run["run_kind"], f"{name}.expected_child_run.run_kind")

    @staticmethod
    def _unsupported_expectation_keys(case_type: str) -> set[str]:
        run_expectations = {
            "approval_decisions",
            "expected_approval_requests",
            "expected_final_text",
            "expected_final_part_types",
            "expected_final_parts",
            "expected_message_roles",
            "expected_tool_texts",
            "expected_tool_text_contains",
            "expected_pending_tool_call_ids",
            "expected_pause",
            "expected_tool_progress",
            "expected_child_run",
            "expected_model_deltas",
            "expected_trace_kinds",
        }
        resume_expectations = {
            "expected_resume_error",
            "expected_resume_message_roles",
            "expected_resume_tool_texts",
            "expected_resume_trace_prefix",
        }
        if case_type in {"run_store_failure", "run_journal_failure"}:
            return run_expectations | resume_expectations
        if case_type in {"run", "resume"}:
            return {"forbidden_journal_event_types"}
        if case_type == "run_store_journal":
            return {
                "expected_message_roles",
                "expected_tool_texts",
                "expected_tool_text_contains",
                "expected_pending_tool_call_ids",
                "expected_pause",
                "expected_model_deltas",
            } | resume_expectations
        if case_type == "run_store_resume_journal":
            return {
                "approval_decisions",
                "expected_event_types",
                "expected_message_roles",
                "expected_tool_texts",
                "expected_tool_text_contains",
                "expected_pending_tool_call_ids",
                "expected_pause",
                "expected_model_deltas",
                "expected_trace_kinds",
                "forbidden_event_types",
            } | resume_expectations
        return set()

    def _validate_stream_event(
        self, name: str, step_index: int, event_index: int, event: dict[str, Any]
    ) -> None:
        label = f"{name}.stream_model_steps[{step_index}].events[{event_index}]"
        if "type" not in event:
            raise KeyError(f"{label} missing required key: type")
        event_type = expect_case_str(event["type"], f"{name}.stream_model_steps event type")
        allowed = STREAM_EVENT_KEYS_BY_TYPE.get(event_type)
        if allowed is None:
            raise ValueError(f"{label} has invalid type: {event_type}")
        reject_unknown_keys(set(event), allowed, label)
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
        if "metadata" in event:
            expect_case_mapping(event["metadata"], f"{label}.metadata")
        if "usage" in event:
            expect_case_mapping(event["usage"], f"{label}.usage")
        for optional_key in ("id", "name", "mode", "arguments_delta"):
            if optional_key in event:
                expect_case_optional_str(event[optional_key], f"{label}.{optional_key}")

    def _validate_model_step(
        self, name: str, step_index: int, step: dict[str, Any], key: str
    ) -> None:
        label = f"{name}.{key}[{step_index}]"
        reject_unknown_keys(set(step), MODEL_STEP_KEYS, label)
        has_error = "error" in step
        has_response = bool(MODEL_STEP_RESPONSE_KEYS & set(step))
        if has_error == has_response:
            raise ValueError(f"{label} must contain either error or a model response")
        if has_error:
            error = step["error"]
            if not isinstance(error, dict):
                raise TypeError(f"{label}.error must be an object")
            self.assert_matches_schema(
                f"{label}.error",
                self.validators.model_error,
                cast(Mapping[str, Any], error),
            )
            return
        for required in ("parts", "tool_calls"):
            if required not in step:
                raise KeyError(f"{label} missing required key: {required}")
        self.assert_matches_schema(label, self.validators.model_response, step)

    def assert_matches_schema(
        self,
        label: str,
        validator: Any,
        instance: Mapping[str, Any],
    ) -> None:
        assert_validator_matches(label, validator, instance)

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

        if case_type == "message_negative":
            try:
                Message.from_dict(cast(Mapping[str, Any], case["message"]))
            except (TypeError, ValueError, KeyError) as exc:
                expected = expect_case_str(case["expected_error"], "expected_error")
                if expected not in str(exc):
                    raise AssertionError(
                        f"{name} expected error containing {expected!r}, got {exc!r}"
                    ) from exc
                return ConformanceCaseResult(name=name, case_type=case_type)
            raise AssertionError(f"{name} expected message rejection")

        if case_type == "resume":
            await self.assert_resume_conformance_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        if case_type == "run_store_failure":
            await self.assert_run_store_failure_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        if case_type == "run_journal_failure":
            await self.assert_run_journal_failure_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        if case_type == "run_store_journal":
            await self.assert_run_store_journal_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        if case_type == "run_store_resume_journal":
            await self.assert_run_store_resume_journal_case(case)
            return ConformanceCaseResult(name=name, case_type=case_type)

        steps = [model_step_from_case_step(step) for step in case["model_steps"]]
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
        steps: Sequence[ModelStep],
        stream_steps: Sequence[dict[str, Any]],
    ) -> AgentResult:
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller, self.validators)
        return await AgentLoop(
            model=model,
            tools=case_tools(self.validators),
            limits=limits_from_case(case),
            hooks=hooks_from_case(case),
            approval_policy=approval_policy_from_case(case, self.validators),
            approval_metadata=approval_metadata_from_case(case),
        ).run(
            [user_text("run conformance case")],
            context=runtime_context_from_case(case),
            stream=bool(stream_steps),
            controller=controller,
        )

    async def assert_run_store_failure_case(self, case: dict[str, Any]) -> None:
        steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller, self.validators)
        journal = MemoryRunJournal()
        events: list[AgentEvent] = []
        try:
            async for event in AgentLoop(
                model=model,
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
                run_store=FailingRunStore(),
                run_journal=journal,
            ).run_events(
                [user_text("run conformance case")],
                stream=bool(stream_steps),
                controller=controller,
            ):
                events.append(event)
        except RuntimeError as exc:
            expected_error = expect_case_str(case["expected_error"], "expected_error")
            if expected_error not in str(exc):
                raise AssertionError(
                    f"expected store failure containing {expected_error!r}, got {exc!r}"
                ) from exc
        else:
            raise AssertionError("expected run store save failure")

        expected_status = AgentStatus(case["expected_status"])
        check(events, "store failure case must emit events before failing")
        self.assert_event_schema_contracts(events)
        state_changes = [event for event in events if event.type == EventTypes.STATE_CHANGED]
        check(state_changes, "store failure case must emit a failed state_changed event")
        check(
            state_changes[-1].data["to"] == expected_status.value,
            f"expected failed state {expected_status.value}, got {state_changes[-1].data['to']}",
        )
        check(
            state_changes[-1].data["total_tool_calls"] == case["expected_tool_calls"],
            f"expected {case['expected_tool_calls']} tool calls before store failure, "
            f"got {state_changes[-1].data['total_tool_calls']}",
        )
        self._assert_expected_event_types(case, events)
        self._assert_forbidden_expectations(case, events)
        self._assert_forbidden_journal_expectations(case, journal.records)
        check(
            [record.event_type for record in journal.records] == [event.type for event in events],
            "journal must contain exactly the emitted events before store failure",
        )
        self._assert_journal_record_schema_contracts(journal.records)

    async def assert_run_journal_failure_case(self, case: dict[str, Any]) -> None:
        steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller, self.validators)
        store = MemoryRunStore()
        journal = FailingCheckpointJournal()
        events: list[AgentEvent] = []
        try:
            async for event in AgentLoop(
                model=model,
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
                run_store=store,
                run_journal=journal,
            ).run_events(
                [user_text("run conformance case")],
                stream=bool(stream_steps),
                controller=controller,
            ):
                events.append(event)
        except RuntimeError as exc:
            expected_error = expect_case_str(case["expected_error"], "expected_error")
            if expected_error not in str(exc):
                raise AssertionError(
                    f"expected journal failure containing {expected_error!r}, got {exc!r}"
                ) from exc
        else:
            raise AssertionError("expected run journal append failure")

        check(events, "journal failure case must emit events before failing")
        self.assert_event_schema_contracts(events)
        expected_status = AgentStatus(case["expected_status"])
        state_changes = [event for event in events if event.type == EventTypes.STATE_CHANGED]
        check(state_changes, "journal failure case must emit state_changed before failing")
        check(
            state_changes[-1].data["to"] == expected_status.value,
            f"expected state {expected_status.value}, got {state_changes[-1].data['to']}",
        )
        check(
            state_changes[-1].data["total_tool_calls"] == case["expected_tool_calls"],
            f"expected {case['expected_tool_calls']} tool calls before journal failure, "
            f"got {state_changes[-1].data['total_tool_calls']}",
        )
        self._assert_expected_event_types(case, events)
        self._assert_forbidden_expectations(case, events)
        self._assert_forbidden_journal_expectations(case, journal.records)
        check(store.checkpoints, "journal failure must happen after checkpoint save")
        self._assert_stored_checkpoint_schema_contracts(store.checkpoints)
        self._assert_journal_record_schema_contracts(journal.records)
        check(
            EventTypes.CHECKPOINT not in [event.type for event in events],
            "failed journal checkpoint event must not be emitted",
        )
        check(
            EventTypes.CHECKPOINT not in [record.event_type for record in journal.records],
            "failed journal checkpoint event must not be recorded",
        )

    async def assert_run_store_journal_case(self, case: dict[str, Any]) -> None:
        steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller, self.validators)
        store = MemoryRunStore()
        journal = MemoryRunJournal()
        events = [
            event
            async for event in AgentLoop(
                model=model,
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
                run_store=store,
                run_journal=journal,
            ).run_events(
                [user_text("run conformance case")],
                context=runtime_context_from_case(case),
                stream=bool(stream_steps),
                controller=controller,
            )
        ]

        check(events, "run store journal case emitted no events")
        self.assert_event_schema_contracts(events)
        self._assert_tool_result_schema_contracts(events, self._latest_checkpoint_messages(events))
        expected_status = AgentStatus(case["expected_status"])
        completed = [event for event in events if event.type == EventTypes.RUN_COMPLETED]
        check(completed, "run store journal case did not emit run_completed")
        completed_state = cast(Mapping[str, Any], completed[-1].data["state"])
        check(
            completed_state["status"] == expected_status.value,
            f"expected status {expected_status.value}, got {completed_state['status']}",
        )
        check(
            completed_state["total_tool_calls"] == case["expected_tool_calls"],
            f"expected {case['expected_tool_calls']} tool calls, "
            f"got {completed_state['total_tool_calls']}",
        )
        if "expected_final_text" in case:
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected final text but no final event was emitted")
            parts = cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            actual_final_text = "".join(str(part.get("text") or "") for part in parts)
            check(
                actual_final_text == case["expected_final_text"],
                f"expected final text {case['expected_final_text']!r}, got {actual_final_text!r}",
            )
        if "expected_final_part_types" in case:
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected final part types but no final event was emitted")
            parts = cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            expected_types = expect_case_list_of_strings(
                case["expected_final_part_types"], "expected_final_part_types"
            )
            actual_types = [
                expect_case_str(part.get("type"), "final event part type") for part in parts
            ]
            check(
                actual_types == expected_types,
                f"expected final event part types {expected_types}, got {actual_types}",
            )
        if "expected_final_parts" in case:
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected final parts but no final event was emitted")
            expected_parts = [
                dict(part)
                for part in expect_case_list(case["expected_final_parts"], "expected_final_parts")
            ]
            parts = [
                dict(part) for part in cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            ]
            check(
                parts == expected_parts,
                f"expected final event parts {expected_parts}, got {parts}",
            )

        self._assert_expected_event_types(case, events)
        if "expected_tool_progress" in case:
            self._assert_expected_tool_progress(case, events)
        if "expected_child_run" in case:
            self._assert_expected_child_run(case, events, expected_status)
        self._assert_forbidden_expectations(case, events)
        self._assert_forbidden_journal_expectations(case, journal.records)
        self._assert_approval_expectations(case, events)
        event_trace = RunTrace.from_events(events[0].run_id, events)
        self.assert_matches_schema(
            "run store journal event trace", self.validators.run_trace, event_trace.to_dict()
        )
        check(
            replay_trace(event_trace).final_status is expected_status,
            "run store journal event trace final status mismatch",
        )
        self._assert_expected_trace_kinds(case, event_trace, event_trace)
        await self._assert_store_journal_segment(
            events,
            store,
            journal,
            initial_parent_checkpoint_id=None,
        )

    async def assert_run_store_resume_journal_case(self, case: dict[str, Any]) -> None:
        initial_result_steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        initial_event_steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        stream_steps = cast(list[dict[str, Any]], case.get("stream_model_steps") or [])
        initial_result = await self.run_case_result(case, initial_result_steps, stream_steps)
        initial_events = await self.collect_case_events(case, initial_event_steps, stream_steps)
        initial_status = AgentStatus(case["expected_status"])
        self.assert_run_case_expectations(case, initial_result, initial_events, initial_status)

        snapshot = select_resume_snapshot(case, initial_events)
        resume_input = ResumeInput(
            snapshot=snapshot,
            append_messages=messages_from_case(case.get("resume_append_messages", [])),
            expected_pause=resume_selector_from_case(case),
        )
        self.assert_matches_schema(
            "run store resume journal resume input",
            self.validators.resume_input,
            resume_input.to_dict(),
        )
        resume_steps = [
            model_step_from_case_step(step)
            for step in cast(list[dict[str, Any]], case.get("resume_model_steps") or [])
        ]
        store = MemoryRunStore()
        journal = MemoryRunJournal()
        events = [
            event
            async for event in AgentLoop(
                model=model_from_case(
                    case,
                    resume_steps,
                    [],
                    controller=None,
                    validators=self.validators,
                ),
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
                run_store=store,
                run_journal=journal,
            ).run_snapshot_events(resume_input)
        ]
        self._assert_tool_result_schema_contracts(events, self._latest_checkpoint_messages(events))

        expected_status = AgentStatus(case["expected_resume_status"])
        self.assert_event_stream_invariants(events, expected_status)
        completed_state = cast(Mapping[str, Any], events[-1].data["state"])
        expected_tool_calls = expect_case_int(
            case.get("expected_resume_tool_calls", 0), "expected_resume_tool_calls"
        )
        check(
            completed_state["total_tool_calls"] == expected_tool_calls,
            f"expected {expected_tool_calls} resume tool calls, "
            f"got {completed_state['total_tool_calls']}",
        )
        if "expected_resume_final_text" in case:
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected resume final text but no final event was emitted")
            parts = cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            actual_final_text = "".join(str(part.get("text") or "") for part in parts)
            check(
                actual_final_text == case["expected_resume_final_text"],
                f"expected resume final text {case['expected_resume_final_text']!r}, "
                f"got {actual_final_text!r}",
            )
        event_trace = RunTrace.from_events(events[0].run_id, events)
        self._assert_forbidden_journal_expectations(case, journal.records)
        self.assert_matches_schema(
            "run store resume journal event trace",
            self.validators.run_trace,
            event_trace.to_dict(),
        )
        check(
            replay_trace(event_trace).final_status is expected_status,
            "run store resume journal event trace final status mismatch",
        )
        await self._assert_store_journal_segment(
            events,
            store,
            journal,
            initial_parent_checkpoint_id=self._expected_checkpoint_id(snapshot.context.sequence),
        )

    async def _assert_store_journal_segment(
        self,
        events: Sequence[AgentEvent],
        store: MemoryRunStore,
        journal: MemoryRunJournal,
        *,
        initial_parent_checkpoint_id: str | None,
    ) -> None:
        checkpoint_events = [event for event in events if event.type == EventTypes.CHECKPOINT]
        check(checkpoint_events, "run store journal case emitted no checkpoint")
        check(
            len(store.checkpoints) == len(checkpoint_events),
            f"expected {len(checkpoint_events)} stored checkpoints, got {len(store.checkpoints)}",
        )
        previous_checkpoint_id = initial_parent_checkpoint_id
        for event, checkpoint in zip(checkpoint_events, store.checkpoints, strict=True):
            self.assert_matches_schema(
                "stored checkpoint",
                self.validators.stored_checkpoint,
                checkpoint.to_dict(),
            )
            self.assert_matches_schema(
                "checkpoint summary",
                self.validators.checkpoint_summary,
                checkpoint.summary().to_dict(),
            )
            snapshot = RunSnapshot.from_dict(event.data)
            checkpoint_id = self._expected_checkpoint_id(event.sequence)
            check(
                checkpoint.checkpoint_id == checkpoint_id,
                f"expected checkpoint id {checkpoint_id}, got {checkpoint.checkpoint_id}",
            )
            check(
                checkpoint.parent_checkpoint_id == previous_checkpoint_id,
                f"expected checkpoint parent {previous_checkpoint_id}, "
                f"got {checkpoint.parent_checkpoint_id}",
            )
            check(checkpoint.sequence == event.sequence, "checkpoint sequence mismatch")
            check(checkpoint.status is snapshot.state.status, "checkpoint status mismatch")
            check(
                checkpoint.snapshot.to_dict() == snapshot.to_dict(),
                "stored checkpoint snapshot does not match checkpoint event",
            )
            previous_checkpoint_id = checkpoint_id

        latest = await store.load_checkpoint(events[0].run_id)
        check(
            latest.to_dict() == store.checkpoints[-1].snapshot.to_dict(),
            "load_checkpoint did not return latest stored snapshot",
        )
        summaries = list(await store.list_checkpoints(events[0].run_id))
        check(
            [summary.to_dict() for summary in summaries]
            == [checkpoint.summary().to_dict() for checkpoint in store.checkpoints],
            "list_checkpoints summaries do not match stored checkpoints",
        )

        check(
            [record.event.to_dict() for record in journal.records]
            == [event.to_dict() for event in events],
            "journal records must match emitted events",
        )
        for record, event in zip(journal.records, events, strict=True):
            self.assert_matches_schema(
                "journal record",
                self.validators.journal_record,
                record.to_dict(),
            )
            expected_checkpoint_id = (
                self._expected_checkpoint_id(event.sequence)
                if event.type == EventTypes.CHECKPOINT
                else None
            )
            check(
                record.checkpoint_id == expected_checkpoint_id,
                f"expected journal checkpoint_id {expected_checkpoint_id}, "
                f"got {record.checkpoint_id}",
            )

    @staticmethod
    def _expected_checkpoint_id(sequence: int) -> str:
        return f"checkpoint-{sequence}"

    def assert_event_schema_contracts(self, events: Sequence[AgentEvent]) -> None:
        for event in events:
            self.assert_matches_schema(
                f"{event.type} event", self.validators.event, event.to_dict()
            )

    def _assert_stored_checkpoint_schema_contracts(
        self, checkpoints: Sequence[StoredCheckpoint]
    ) -> None:
        for checkpoint in checkpoints:
            self.assert_matches_schema(
                "stored checkpoint",
                self.validators.stored_checkpoint,
                checkpoint.to_dict(),
            )
            self.assert_matches_schema(
                "checkpoint summary",
                self.validators.checkpoint_summary,
                checkpoint.summary().to_dict(),
            )

    def _assert_journal_record_schema_contracts(self, records: Sequence[JournalRecord]) -> None:
        for record in records:
            self.assert_matches_schema(
                "journal record",
                self.validators.journal_record,
                record.to_dict(),
            )

    def _assert_tool_result_schema_contracts(
        self, events: Sequence[AgentEvent], messages: Sequence[Message]
    ) -> None:
        tool_messages = {
            message.tool_call_id: message
            for message in messages
            if message.role == "tool" and message.tool_call_id is not None
        }
        for event in events:
            if event.type != EventTypes.TOOL_COMPLETED:
                continue
            call_id = expect_case_str(event.data["id"], "tool_completed id")
            message = tool_messages.get(call_id)
            if message is None:
                continue
            summary = expect_case_mapping(event.data["result"], "tool_completed result")
            result_kind = expect_case_str(summary["result_kind"], "tool result kind")
            payload: dict[str, Any] = {
                "kind": result_kind,
                "parts": [part.to_dict() for part in message.parts],
            }
            is_error = bool(summary.get("is_error", False))
            if result_kind == "observation" or is_error:
                payload["is_error"] = is_error
            metadata = expect_case_mapping(summary.get("metadata", {}), "tool result metadata")
            if metadata:
                payload["metadata"] = dict(metadata)
            message_metadata = message.metadata
            check(
                message_metadata.get("result_kind") == result_kind,
                f"tool message {call_id} result_kind metadata mismatch",
            )
            if is_error:
                check(
                    message_metadata.get("is_error") is True,
                    f"tool message {call_id} is_error metadata mismatch",
                )
            else:
                check(
                    "is_error" not in message_metadata,
                    f"tool message {call_id} unexpectedly marks is_error",
                )
            pause = summary.get("pause")
            if pause is not None:
                payload["pause"] = expect_case_mapping(pause, "tool result pause")
            if "correlation_id" in summary:
                correlation_id = expect_case_str(
                    summary["correlation_id"], "tool result correlation_id"
                )
                payload["correlation_id"] = correlation_id
                check(
                    message_metadata.get("correlation_id") == correlation_id,
                    f"tool message {call_id} correlation_id metadata mismatch",
                )
            if "background_task" in summary:
                background_task = expect_case_mapping(
                    summary["background_task"], "tool result background_task"
                )
                payload["background_task"] = background_task
                check(
                    message_metadata.get("background_task") == background_task,
                    f"tool message {call_id} background_task metadata mismatch",
                )
            self.assert_matches_schema(
                f"tool result {call_id}", self.validators.tool_result, payload
            )

    def _latest_checkpoint_messages(self, events: Sequence[AgentEvent]) -> Sequence[Message]:
        checkpoint_events = [event for event in events if event.type == EventTypes.CHECKPOINT]
        check(checkpoint_events, "expected checkpoint event with messages")
        snapshot = RunSnapshot.from_dict(checkpoint_events[-1].data)
        return snapshot.state.messages

    async def collect_case_events(
        self,
        case: dict[str, Any],
        steps: Sequence[ModelStep],
        stream_steps: Sequence[dict[str, Any]],
    ) -> list[AgentEvent]:
        controller = controller_from_case(case)
        model = model_from_case(case, steps, stream_steps, controller, self.validators)
        return [
            event
            async for event in AgentLoop(
                model=model,
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
            ).run_events(
                [user_text("run conformance case")],
                context=runtime_context_from_case(case),
                stream=bool(stream_steps),
                controller=controller,
            )
        ]

    async def collect_resume_case_events(
        self,
        case: dict[str, Any],
        resume_input: ResumeInput,
        steps: Sequence[ModelStep],
    ) -> list[AgentEvent]:
        return [
            event
            async for event in AgentLoop(
                model=model_from_case(
                    case,
                    steps,
                    [],
                    controller=None,
                    validators=self.validators,
                ),
                tools=case_tools(self.validators),
                limits=limits_from_case(case),
                hooks=hooks_from_case(case),
                approval_policy=approval_policy_from_case(case, self.validators),
                approval_metadata=approval_metadata_from_case(case),
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

    def assert_result_schema_contracts(self, label: str, result: AgentResult) -> None:
        for index, message in enumerate(result.messages):
            self.assert_matches_schema(
                f"{label} message[{index}]",
                self.validators.message,
                message.to_dict(),
            )
        if result.snapshot is not None:
            self.assert_matches_schema(
                f"{label} snapshot",
                self.validators.run_snapshot,
                result.snapshot.to_dict(),
            )
        if result.trace is not None:
            self.assert_matches_schema(
                f"{label} trace",
                self.validators.run_trace,
                result.trace,
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
        self.assert_result_schema_contracts("result", result)
        self.assert_event_stream_invariants(events, expected_status)
        self._assert_tool_result_schema_contracts(events, result.messages)
        trace_payload = result.trace
        if trace_payload is None:
            raise AssertionError("result trace is missing")
        result_trace = RunTrace.from_dict(trace_payload)
        check(
            replay_trace(trace_payload).final_status is expected_status,
            "result trace final status mismatch",
        )
        event_trace = RunTrace.from_events(events[0].run_id, events)
        self.assert_matches_schema("event trace", self.validators.run_trace, event_trace.to_dict())
        check(
            replay_trace(event_trace).final_status is expected_status,
            "event trace final status mismatch",
        )
        self._assert_expected_event_types(case, events)
        if "expected_trace_kinds" in case:
            expected_trace_kinds = expect_case_list_of_strings(
                case["expected_trace_kinds"], "expected_trace_kinds"
            )
            result_kinds = [step.kind for step in result_trace.steps]
            event_kinds = [step.kind for step in event_trace.steps]
            missing_result = [kind for kind in expected_trace_kinds if kind not in result_kinds]
            missing_event = [kind for kind in expected_trace_kinds if kind not in event_kinds]
            check(not missing_result, f"result trace missing expected kind(s): {missing_result}")
            check(not missing_event, f"event trace missing expected kind(s): {missing_event}")
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
        if "expected_final_part_types" in case:
            expected_types = expect_case_list_of_strings(
                case["expected_final_part_types"], "expected_final_part_types"
            )
            actual_types = [part.type for part in result.final_parts]
            check(
                actual_types == expected_types,
                f"expected final part types {expected_types}, got {actual_types}",
            )
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected final part types but no final event was emitted")
            event_parts = cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            event_types = [
                expect_case_str(part.get("type"), "final event part type") for part in event_parts
            ]
            check(
                event_types == expected_types,
                f"expected final event part types {expected_types}, got {event_types}",
            )
        if "expected_final_parts" in case:
            expected_parts = [
                dict(part)
                for part in expect_case_list(case["expected_final_parts"], "expected_final_parts")
            ]
            actual_parts = [part.to_dict() for part in result.final_parts]
            check(
                actual_parts == expected_parts,
                f"expected final parts {expected_parts}, got {actual_parts}",
            )
            final_events = [event for event in events if event.type == EventTypes.FINAL]
            check(final_events, "expected final parts but no final event was emitted")
            raw_event_parts = cast(list[Mapping[str, Any]], final_events[-1].data["parts"])
            event_parts = [dict(part) for part in raw_event_parts]
            check(
                event_parts == expected_parts,
                f"expected final event parts {expected_parts}, got {event_parts}",
            )
        if "expected_tool_texts" in case:
            actual_tool_texts = [
                message.text for message in result.messages if message.role == "tool"
            ]
            check(
                actual_tool_texts == case["expected_tool_texts"],
                f"expected tool texts {case['expected_tool_texts']}, got {actual_tool_texts}",
            )
        if "expected_tool_text_contains" in case:
            actual_tool_texts = [
                message.text for message in result.messages if message.role == "tool"
            ]
            for expected in expect_case_list_of_strings(
                case["expected_tool_text_contains"], "expected_tool_text_contains"
            ):
                check(
                    any(expected in text for text in actual_tool_texts),
                    f"expected a tool text containing {expected!r}, got {actual_tool_texts}",
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
        if "expected_tool_progress" in case:
            self._assert_expected_tool_progress(case, events)
        if "expected_child_run" in case:
            self._assert_expected_child_run(case, events, expected_status)
        if "expected_model_deltas" in case:
            actual_deltas = [
                dict(event.data) for event in events if event.type == EventTypes.MODEL_DELTA
            ]
            check(
                actual_deltas == case["expected_model_deltas"],
                f"expected model deltas {case['expected_model_deltas']}, got {actual_deltas}",
            )
        self._assert_approval_expectations(case, events)
        self._assert_forbidden_expectations(case, events)

    def _assert_expected_tool_progress(
        self, case: dict[str, Any], events: Sequence[AgentEvent]
    ) -> None:
        expected = [
            dict(item)
            for item in expect_case_list(case["expected_tool_progress"], "expected_tool_progress")
        ]
        actual = [
            dict(cast(Mapping[str, Any], event.data["progress"]))
            for event in events
            if event.type == EventTypes.TOOL_PROGRESS
        ]
        check(actual == expected, f"expected tool progress {expected}, got {actual}")

    def _assert_expected_child_run(
        self,
        case: dict[str, Any],
        events: Sequence[AgentEvent],
        expected_status: AgentStatus,
    ) -> None:
        expected = dict(expect_case_mapping(case["expected_child_run"], "expected_child_run"))
        started = [event for event in events if event.type == EventTypes.CHILD_RUN_STARTED]
        completed = [event for event in events if event.type == EventTypes.CHILD_RUN_COMPLETED]
        check(len(started) == 1, f"expected one child_run_started, got {len(started)}")
        check(len(completed) == 1, f"expected one child_run_completed, got {len(completed)}")
        check(
            started[0].data == expected,
            f"expected child_run_started {expected}, got {started[0].data}",
        )
        expected_completed = expected | {"status": expected_status.value}
        check(
            completed[0].data == expected_completed,
            f"expected child_run_completed {expected_completed}, got {completed[0].data}",
        )
        check(
            events.index(started[0]) < events.index(completed[0]),
            "child_run_completed must follow child_run_started",
        )

    def _assert_approval_expectations(
        self, case: dict[str, Any], events: Sequence[AgentEvent]
    ) -> None:
        raw_decisions_obj = case.get("approval_decisions")
        if not isinstance(raw_decisions_obj, dict):
            if "expected_approval_requests" in case:
                raise AssertionError("expected_approval_requests require approval_decisions")
            return
        raw_decisions = cast(Mapping[str, object], raw_decisions_obj)
        raw_request_expectations_obj = case.get("expected_approval_requests", {})
        request_expectations: Mapping[str, object]
        if isinstance(raw_request_expectations_obj, dict):
            request_expectations = cast(Mapping[str, object], raw_request_expectations_obj)
        else:
            request_expectations = {}
        for call_id, raw_decision in raw_decisions.items():
            decision = ApprovalDecision.from_dict(cast(Mapping[str, Any], raw_decision))
            requested_events = [
                event
                for event in events
                if event.type == EventTypes.APPROVAL_REQUESTED and event.data["id"] == call_id
            ]
            completed_events = [
                event
                for event in events
                if event.type == EventTypes.APPROVAL_COMPLETED and event.data["id"] == call_id
            ]
            check(requested_events, f"approval call {call_id} has no approval_requested event")
            check(completed_events, f"approval call {call_id} has no approval_completed event")
            check(
                events.index(requested_events[-1]) < events.index(completed_events[-1]),
                f"approval call {call_id} completed before it was requested",
            )
            check(
                completed_events[-1].data["action"] == decision.action,
                f"approval call {call_id} expected action {decision.action}, "
                f"got {completed_events[-1].data['action']}",
            )
            lifecycle_events = [
                event
                for event in events
                if event.type in {EventTypes.TOOL_STARTED, EventTypes.TOOL_COMPLETED}
                and event.data["id"] == call_id
            ]
            if decision.action == "pause":
                check(
                    not lifecycle_events,
                    f"paused approval call {call_id} must not start tool lifecycle",
                )
                pause_events = [
                    event
                    for event in events
                    if event.type == EventTypes.PAUSE_REQUESTED
                    and event.data["request"]["wait_id"] == call_id
                ]
                check(pause_events, f"paused approval call {call_id} has no pause_requested")
                check(
                    events.index(completed_events[-1]) < events.index(pause_events[-1]),
                    f"approval call {call_id} paused before approval_completed",
                )
                continue
            check(lifecycle_events, f"approval call {call_id} has no tool lifecycle event")
            check(
                events.index(completed_events[-1]) < events.index(lifecycle_events[0]),
                f"approval call {call_id} started lifecycle before approval_completed",
            )
            for event in lifecycle_events:
                expected_invoked = decision.action == "allow"
                check(
                    event.data.get("implementation_invoked") is expected_invoked,
                    f"{event.type} for approval call {call_id} expected "
                    f"implementation_invoked={expected_invoked}",
                )
        for call_id, raw_expected_request in request_expectations.items():
            expected_request = cast(Mapping[str, Any], raw_expected_request)
            requested_events = [
                event
                for event in events
                if event.type == EventTypes.APPROVAL_REQUESTED and event.data["id"] == call_id
            ]
            check(requested_events, f"approval call {call_id} has no approval_requested event")
            request_data = requested_events[-1].data
            if "risk" in expected_request:
                expected_risk = dict(
                    expect_case_mapping(
                        expected_request["risk"],
                        f"expected approval request {call_id}.risk",
                    )
                )
                actual_risk = dict(cast(Mapping[str, Any], request_data.get("risk", {})))
                check(
                    actual_risk == expected_risk,
                    f"approval call {call_id} expected risk {expected_risk}, got {actual_risk}",
                )
            if "metadata" in expected_request:
                expected_metadata = dict(
                    expect_case_mapping(
                        expected_request["metadata"],
                        f"expected approval request {call_id}.metadata",
                    )
                )
                actual_metadata = dict(cast(Mapping[str, Any], request_data.get("metadata", {})))
                check(
                    actual_metadata == expected_metadata,
                    f"approval call {call_id} expected metadata {expected_metadata}, "
                    f"got {actual_metadata}",
                )

    def _assert_expected_trace_kinds(
        self,
        case: dict[str, Any],
        result_trace: RunTrace,
        event_trace: RunTrace,
    ) -> None:
        if "expected_trace_kinds" not in case:
            return
        expected_trace_kinds = expect_case_list_of_strings(
            case["expected_trace_kinds"], "expected_trace_kinds"
        )
        result_kinds = [step.kind for step in result_trace.steps]
        event_kinds = [step.kind for step in event_trace.steps]
        missing_result = [kind for kind in expected_trace_kinds if kind not in result_kinds]
        missing_event = [kind for kind in expected_trace_kinds if kind not in event_kinds]
        check(not missing_result, f"result trace missing expected kind(s): {missing_result}")
        check(not missing_event, f"event trace missing expected kind(s): {missing_event}")

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

    def _assert_expected_event_types(
        self, case: dict[str, Any], events: Sequence[AgentEvent]
    ) -> None:
        if "expected_event_types" not in case:
            return
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

    def _assert_forbidden_journal_expectations(
        self, case: dict[str, Any], records: Sequence[JournalRecord]
    ) -> None:
        if "forbidden_journal_event_types" not in case:
            return
        forbidden_events = set(
            expect_case_list_of_strings(
                case["forbidden_journal_event_types"], "forbidden_journal_event_types"
            )
        )
        actual = [record.event_type for record in records if record.event_type in forbidden_events]
        check(not actual, f"forbidden journal event type(s) recorded: {actual}")

    async def assert_resume_conformance_case(self, case: dict[str, Any]) -> None:
        initial_result_steps = [model_step_from_case_step(step) for step in case["model_steps"]]
        initial_event_steps = [model_step_from_case_step(step) for step in case["model_steps"]]
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
            model_step_from_case_step(step)
            for step in cast(list[dict[str, Any]], case.get("resume_model_steps") or [])
        ]
        resume_event_steps = [
            model_step_from_case_step(step)
            for step in cast(list[dict[str, Any]], case.get("resume_model_steps") or [])
        ]
        result = await AgentLoop(
            model=model_from_case(
                case,
                resume_steps,
                [],
                controller=None,
                validators=self.validators,
            ),
            tools=case_tools(self.validators),
            limits=limits_from_case(case),
            hooks=hooks_from_case(case),
            approval_policy=approval_policy_from_case(case, self.validators),
            approval_metadata=approval_metadata_from_case(case),
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
        self.assert_result_schema_contracts("resume result", result)
        self.assert_event_stream_invariants(resume_events, expected_status)
        self._assert_tool_result_schema_contracts(resume_events, result.messages)
        trace_payload = result.trace
        if trace_payload is None:
            raise AssertionError("resume result trace is missing")
        result_trace = RunTrace.from_dict(trace_payload)
        check(
            replay_trace(trace_payload).final_status is expected_status,
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
        if "expected_resume_trace_prefix" in case:
            actual_prefix = [step.kind for step in result_trace.steps][
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
