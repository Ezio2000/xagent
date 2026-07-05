"""Scripted models, tools, and factories used by conformance cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from kernel import (
    AgentEvent,
    AgentStatus,
    ApprovalDecision,
    ApprovalRequest,
    ContentPart,
    ConversationInsert,
    EventTypes,
    LoopLimits,
    Message,
    ModelContentDelta,
    ModelErrorInfo,
    ModelProviderError,
    ModelReasoningDelta,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    PauseRequest,
    PauseSelector,
    RunController,
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    ToolCall,
    ToolOutput,
)
from support import (
    ControlledModelDriver,
    ControlledStreamingModelDriver,
    ModelStep,
    ModelStreamAction,
    ModelStreamPause,
    ModelStreamSleep,
    RetryModelErrorHook,
    apply_pause_request,
)
from toolkit import (
    RuntimeContextSnapshot,
    Tool,
    ToolCancelChecker,
    ToolProgressEmitter,
    ToolRegistry,
)

from conformance._case import (
    expect_case_int,
    expect_case_list,
    expect_case_mapping,
    expect_case_number,
    expect_case_optional_int,
    expect_case_optional_str,
    expect_case_str,
)
from conformance._schemas import ConformanceValidators, assert_validator_matches
from conformance._standard_tools import standard_case_tools


class CaseApprovalPolicy:
    def __init__(
        self,
        decisions: Mapping[str, ApprovalDecision],
        *,
        validate_request: Callable[[ApprovalRequest], None] | None = None,
        validate_decision: Callable[[ApprovalDecision], None] | None = None,
    ) -> None:
        self._decisions = dict(decisions)
        self._validate_request = validate_request
        self._validate_decision = validate_decision

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        if self._validate_request is not None:
            self._validate_request(request)
        decision = self._decisions.get(request.tool_call.id, ApprovalDecision.allow())
        if self._validate_decision is not None:
            self._validate_decision(decision)
        return decision


class ValidatingToolRegistry(ToolRegistry):
    __slots__ = ("_validators",)

    def __init__(self, tools: Sequence[Tool], validators: ConformanceValidators) -> None:
        super().__init__(tools)
        self._validators = validators

    async def invoke(
        self,
        call: ToolCall,
        context: RuntimeContextSnapshot,
        *,
        progress_emitter: ToolProgressEmitter | None = None,
        cancel_checker: ToolCancelChecker | None = None,
    ) -> ToolOutput:
        output = await super().invoke(
            call,
            context,
            progress_emitter=progress_emitter,
            cancel_checker=cancel_checker,
        )
        assert_validator_matches("tool result", self._validators.tool_result, output.to_dict())
        return output


def content_part_from_case(part: dict[str, Any]) -> ContentPart:
    return ContentPart.from_dict(part)


def model_step_from_case_step(step: dict[str, Any]) -> ModelStep:
    if "error" in step:
        error = ModelErrorInfo.from_dict(cast(Mapping[str, Any], step["error"]))
        return ModelProviderError(error)
    return ModelResponse.from_dict(step)


def limits_from_case(case: dict[str, Any]) -> LoopLimits:
    raw_limits = cast(dict[str, Any], case.get("limits", {}))
    return LoopLimits(
        max_iterations=cast(int, raw_limits.get("max_iterations", 8)),
        max_total_tool_calls=cast(int, raw_limits.get("max_total_tool_calls", 20)),
        timeout_seconds=cast(float | None, raw_limits.get("timeout_seconds")),
        stop_on_tool_error=cast(bool, raw_limits.get("stop_on_tool_error", False)),
        max_parallel_tool_calls=cast(int, raw_limits.get("max_parallel_tool_calls", 1)),
        max_total_tokens=cast(int | None, raw_limits.get("max_total_tokens")),
        max_model_retries=cast(int, raw_limits.get("max_model_retries", 0)),
    )


def approval_policy_from_case(
    case: dict[str, Any],
    validators: ConformanceValidators,
) -> CaseApprovalPolicy | None:
    raw_decisions_obj = case.get("approval_decisions")
    if not isinstance(raw_decisions_obj, dict):
        return None
    raw_decisions = cast(Mapping[str, object], raw_decisions_obj)
    decisions: dict[str, ApprovalDecision] = {}
    for call_id, raw_decision in raw_decisions.items():
        decisions[expect_case_str(call_id, "approval decision call_id")] = (
            ApprovalDecision.from_dict(cast(Mapping[str, Any], raw_decision))
        )
    return CaseApprovalPolicy(
        decisions,
        validate_request=lambda request: assert_validator_matches(
            "approval request",
            validators.approval_request,
            request.to_dict(),
        ),
        validate_decision=lambda decision: assert_validator_matches(
            "approval decision",
            validators.approval_decision,
            decision.to_dict(),
        ),
    )


def approval_metadata_from_case(case: dict[str, Any]) -> Mapping[str, Any] | None:
    raw_metadata = case.get("approval_metadata")
    if raw_metadata is None:
        return None
    return expect_case_mapping(raw_metadata, "approval_metadata")


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


def stream_actions_from_case_steps(
    stream_steps: Sequence[dict[str, Any]],
    *,
    pause_request_on_stream_event: PauseRequest | None,
) -> list[list[ModelStreamAction]]:
    actions_by_step: list[list[ModelStreamAction]] = []
    for step in stream_steps:
        actions: list[ModelStreamAction] = []
        for raw_event in cast(list[dict[str, Any]], step.get("events") or []):
            event_type = expect_case_str(raw_event["type"], "stream event type")
            if event_type == "text_delta":
                actions.append(
                    ModelContentDelta(
                        index=expect_case_int(raw_event["index"], "stream event index"),
                        text_delta=expect_case_str(
                            raw_event["text_delta"], "stream event text_delta"
                        ),
                        part_type=expect_case_str(raw_event["part_type"], "stream event part_type"),
                        metadata=expect_case_mapping(
                            raw_event.get("metadata", {}), "stream event metadata"
                        ),
                    )
                )
            elif event_type == "reasoning_delta":
                actions.append(
                    ModelReasoningDelta(
                        index=expect_case_int(raw_event["index"], "stream event index"),
                        text_delta=expect_case_str(
                            raw_event["text_delta"], "stream event text_delta"
                        ),
                        metadata=expect_case_mapping(
                            raw_event.get("metadata", {}), "stream event metadata"
                        ),
                    )
                )
            elif event_type == "tool_call_delta":
                actions.append(
                    ModelToolCallDelta(
                        index=expect_case_int(raw_event["index"], "stream event index"),
                        id=expect_case_optional_str(raw_event.get("id"), "stream event id"),
                        name=expect_case_optional_str(raw_event.get("name"), "stream event name"),
                        mode=expect_case_optional_str(raw_event.get("mode"), "stream event mode"),
                        arguments_delta=expect_case_optional_str(
                            raw_event.get("arguments_delta"), "stream event arguments_delta"
                        ),
                        metadata=expect_case_mapping(
                            raw_event.get("metadata", {}), "stream event metadata"
                        ),
                    )
                )
            elif event_type == "usage_delta":
                actions.append(
                    ModelUsageDelta(
                        usage=ModelUsage.from_dict(
                            expect_case_mapping(raw_event["usage"], "stream event usage")
                        ),
                        metadata=expect_case_mapping(
                            raw_event.get("metadata", {}), "stream event metadata"
                        ),
                    )
                )
            elif event_type == "sleep":
                actions.append(
                    ModelStreamSleep(
                        expect_case_number(raw_event["seconds"], "stream event seconds")
                    )
                )
            elif event_type == "pause_request":
                if pause_request_on_stream_event is not None:
                    actions.append(ModelStreamPause(pause_request_on_stream_event))
            else:
                raise AssertionError(f"unsupported stream event type: {event_type}")
        actions_by_step.append(actions)
    return actions_by_step


def runtime_context_from_case(case: dict[str, Any]) -> RuntimeContext | None:
    raw_context = case.get("runtime_context")
    if raw_context is None:
        return None
    return RuntimeContext.from_dict(expect_case_mapping(raw_context, "runtime_context"))


def model_from_case(
    case: dict[str, Any],
    steps: Sequence[ModelStep],
    stream_steps: Sequence[dict[str, Any]],
    controller: RunController | None,
    validators: ConformanceValidators,
) -> ControlledModelDriver:
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
        return ControlledStreamingModelDriver(
            steps,
            stream_actions_from_case_steps(
                stream_steps,
                pause_request_on_stream_event=pause_request_on_stream_event,
            ),
            controller=controller,
            pause_request_on_call=pause_request_on_call,
            conversation_insert_on_call=conversation_insert_on_call,
            validate_request=lambda request: assert_validator_matches(
                "model request",
                validators.model_request,
                request.to_dict(),
            ),
        )
    return ControlledModelDriver(
        steps,
        controller=controller,
        pause_request_on_call=pause_request_on_call,
        conversation_insert_on_call=conversation_insert_on_call,
        validate_request=lambda request: assert_validator_matches(
            "model request",
            validators.model_request,
            request.to_dict(),
        ),
    )


def case_tools(validators: ConformanceValidators) -> ToolRegistry:
    return ValidatingToolRegistry(
        cast(Sequence[Tool], standard_case_tools()),
        validators,
    )


def hooks_from_case(case: dict[str, Any]) -> list[RuntimeHook]:
    if case.get("retry_model_errors") is True:
        return [RetryModelErrorHook()]
    return []


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
