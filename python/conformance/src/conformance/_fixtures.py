"""Scripted models, tools, and factories used by conformance cases."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, cast

from kernel import (
    AgentEvent,
    AgentStatus,
    ApprovalDecision,
    ApprovalRequest,
    BackgroundTask,
    CheckpointSummary,
    ContentPart,
    ConversationInsert,
    EventTypes,
    JournalRecord,
    LoopLimits,
    Message,
    ModelCapabilities,
    ModelContentDelta,
    ModelErrorDecision,
    ModelErrorInfo,
    ModelProviderError,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    PauseRequest,
    PauseSelector,
    RunController,
    RunSnapshot,
    RuntimeContext,
    RuntimeHook,
    StoredCheckpoint,
    ToolAcceptance,
    ToolCall,
    ToolObservation,
    ToolOutput,
    ToolRejection,
    ToolSpec,
)
from toolkit import (
    RuntimeContextSnapshot,
    Tool,
    ToolCancelChecker,
    ToolExecutionContext,
    ToolInvocation,
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
    expect_case_sequence,
    expect_case_str,
)
from conformance._schemas import ConformanceValidators, assert_validator_matches

ModelStep = ModelResponse | ModelProviderError


class ScriptedModel:
    def __init__(
        self,
        steps: Sequence[ModelStep],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
        validate_request: Callable[[ModelRequest], None] | None = None,
    ) -> None:
        self._steps = list(steps)
        self._controller = controller
        self._pause_request_on_call = pause_request_on_call
        self._pause_request_on_stream_event = pause_request_on_stream_event
        self._conversation_insert_on_call = conversation_insert_on_call
        self._validate_request = validate_request
        self._pause_requested = False
        self._conversation_inserted = False
        self.calls = 0

    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        self._assert_model_request_contract(request)
        if self.calls >= len(self._steps):
            raise AssertionError("scripted model exhausted")
        step = self._steps[self.calls]
        self.calls += 1
        self._apply_conversation_insert_once(self._conversation_insert_on_call)
        self._apply_pause_once(self._pause_request_on_call)
        if isinstance(step, ModelProviderError):
            raise step
        return step

    def _apply_pause_once(self, request: PauseRequest | None) -> None:
        if self._controller is not None and request is not None and not self._pause_requested:
            self._pause_requested = True
            apply_pause_request(self._controller, request)

    def _apply_conversation_insert_once(self, insert: ConversationInsert | None) -> None:
        if self._controller is not None and insert is not None and not self._conversation_inserted:
            self._conversation_inserted = True
            self._controller.insert(insert)

    def _assert_model_request_contract(self, request: ModelRequest) -> None:
        if self._validate_request is not None:
            self._validate_request(request)


class StreamedCaseModel(ScriptedModel):
    capabilities = ModelCapabilities(streaming=True)

    def __init__(
        self,
        steps: Sequence[ModelStep],
        stream_steps: Sequence[dict[str, Any]],
        *,
        controller: RunController | None = None,
        pause_request_on_call: PauseRequest | None = None,
        pause_request_on_stream_event: PauseRequest | None = None,
        conversation_insert_on_call: ConversationInsert | None = None,
        validate_request: Callable[[ModelRequest], None] | None = None,
    ) -> None:
        super().__init__(
            steps,
            controller=controller,
            pause_request_on_call=pause_request_on_call,
            pause_request_on_stream_event=pause_request_on_stream_event,
            conversation_insert_on_call=conversation_insert_on_call,
            validate_request=validate_request,
        )
        self._stream_steps = list(stream_steps)
        self.stream_calls = 0

    async def stream(self, request: ModelRequest, context: RuntimeContext) -> AsyncIterator[object]:
        _ = context
        self._assert_model_request_contract(request)
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
                    metadata=expect_case_mapping(
                        raw_event.get("metadata", {}), "stream event metadata"
                    ),
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
                    metadata=expect_case_mapping(
                        raw_event.get("metadata", {}), "stream event metadata"
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


class HandoffTool:
    spec = ToolSpec(
        name="handoff",
        description="Return generic custom-mode tool output.",
        input_schema={"type": "object", "properties": {}},
        modes=("handoff",),
    )

    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolOutput:
        _ = context
        return ToolOutput(
            kind=str(invocation.arguments.get("kind", "handoff")),
            parts=[ContentPart.text_part(str(invocation.arguments.get("text", "handoff")))],
            is_error=bool(invocation.arguments.get("is_error", False)),
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
        raw_background_task = invocation.arguments.get("background_task")
        background_task = (
            None
            if raw_background_task is None
            else BackgroundTask.from_dict(
                expect_case_mapping(raw_background_task, "wait background_task")
            )
        )
        return ToolObservation.waiting(
            str(invocation.arguments.get("text", "external wait started")),
            wait_id=str(invocation.arguments["wait_id"]),
            reason=str(invocation.arguments.get("reason", "external_wait")),
            background_task=background_task,
        )


class ProgressTool:
    spec = ToolSpec(
        name="progress",
        description="Emit live progress records.",
        input_schema={"type": "object", "properties": {}},
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        raw_steps = invocation.arguments.get("steps", [])
        for step in expect_case_sequence(raw_steps, "progress steps"):
            context.emit_progress({"step": step})
        return ToolObservation.text(str(invocation.arguments.get("text", "progress complete")))


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


class StrictCountTool:
    def __init__(self) -> None:
        self.calls = 0

    spec = ToolSpec(
        name="strict_count",
        description="Require an integer count.",
        input_schema={
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
            "additionalProperties": False,
        },
    )

    async def execute(
        self, invocation: ToolInvocation, context: ToolExecutionContext
    ) -> ToolObservation:
        _ = context
        self.calls += 1
        return ToolObservation.text(str(invocation.arguments["count"]))


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


class FailingCheckpointStore:
    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        _ = checkpoint
        raise RuntimeError("store unavailable")

    async def load_checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> RunSnapshot:
        _ = checkpoint_id
        raise KeyError(run_id)

    async def list_checkpoints(self, run_id: str) -> Sequence[CheckpointSummary]:
        _ = run_id
        return ()


class CapturingRunStore:
    def __init__(self) -> None:
        self.checkpoints: list[StoredCheckpoint] = []

    async def save_checkpoint(self, checkpoint: StoredCheckpoint) -> None:
        self.checkpoints.append(StoredCheckpoint.from_dict(checkpoint.to_dict()))

    async def load_checkpoint(self, run_id: str, checkpoint_id: str | None = None) -> RunSnapshot:
        matches = [checkpoint for checkpoint in self.checkpoints if checkpoint.run_id == run_id]
        if checkpoint_id is not None:
            matches = [
                checkpoint for checkpoint in matches if checkpoint.checkpoint_id == checkpoint_id
            ]
        if not matches:
            raise KeyError(run_id)
        return RunSnapshot.from_dict(matches[-1].snapshot.to_dict())

    async def list_checkpoints(self, run_id: str) -> Sequence[CheckpointSummary]:
        return [
            checkpoint.summary() for checkpoint in self.checkpoints if checkpoint.run_id == run_id
        ]


class CapturingRunJournal:
    def __init__(self) -> None:
        self.records: list[JournalRecord] = []

    async def append(self, record: JournalRecord) -> None:
        self.records.append(
            JournalRecord(
                event=AgentEvent(**record.event.to_dict()),
                checkpoint_id=record.checkpoint_id,
                trace_step_id=record.trace_step_id,
                payload_ref=record.payload_ref,
                payload_hash=record.payload_hash,
                metadata=record.metadata,
            )
        )

    async def read(
        self, run_id: str, *, after_sequence: int | None = None
    ) -> AsyncIterator[JournalRecord]:
        for record in self.records:
            if record.run_id != run_id:
                continue
            if after_sequence is not None and record.sequence <= after_sequence:
                continue
            yield record


class FailingCheckpointJournal(CapturingRunJournal):
    async def append(self, record: JournalRecord) -> None:
        if record.event_type == EventTypes.CHECKPOINT:
            raise RuntimeError("journal unavailable")
        await super().append(record)


class RetryModelErrorHook(RuntimeHook):
    def on_model_error(
        self,
        error: ModelErrorInfo,
        request: ModelRequest,
        context: RuntimeContext,
    ) -> ModelErrorDecision | None:
        _ = request, context
        return ModelErrorDecision(retry=error.retryable)


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
            validate_request=lambda request: assert_validator_matches(
                "model request",
                validators.model_request,
                request.to_dict(),
            ),
        )
    return ScriptedModel(
        steps,
        controller=controller,
        pause_request_on_call=pause_request_on_call,
        pause_request_on_stream_event=pause_request_on_stream_event,
        conversation_insert_on_call=conversation_insert_on_call,
        validate_request=lambda request: assert_validator_matches(
            "model request",
            validators.model_request,
            request.to_dict(),
        ),
    )


def case_tools(validators: ConformanceValidators) -> ToolRegistry:
    return ValidatingToolRegistry(
        [
            EchoTool(),
            AcceptTool(),
            HandoffTool(),
            FailTool(),
            DelayedEchoTool(),
            WaitTool(),
            ProgressTool(),
            ParallelWaitTool(),
            StrictCountTool(),
        ],
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
