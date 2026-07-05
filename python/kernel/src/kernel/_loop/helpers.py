"""Pure helper functions for AgentLoop."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kernel._loop.types import RunControlState
from kernel.context import RuntimeContext
from kernel.control import ToolCancelRequest
from kernel.errors import AgentError
from kernel.events import EventTypes
from kernel.messages import ContentPart, Message, ToolCall
from kernel.models import ModelUsage
from kernel.scheduler import ToolBatch, ToolCompleted, ToolStarted
from kernel.snapshot import RunSnapshot
from kernel.state import AgentState
from kernel.tools import (
    BackgroundTask,
    ToolAcceptance,
    ToolObservation,
    ToolOutput,
    ToolRejection,
)

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def tool_call_snapshots(calls: tuple[ToolCall, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(ToolCall.from_dict(call.to_dict()).to_dict() for call in calls)


def tool_calls_from_snapshots(
    snapshots: tuple[Mapping[str, Any], ...],
) -> tuple[ToolCall, ...]:
    return tuple(ToolCall.from_dict(snapshot) for snapshot in snapshots)


def normalize_tool_output(output: ToolOutput) -> ToolOutput:
    if isinstance(output, ToolObservation):
        return ToolObservation.from_dict(output.to_dict())
    if isinstance(output, ToolAcceptance):
        return ToolAcceptance.from_dict(output.to_dict())
    if isinstance(output, ToolRejection):
        return ToolRejection.from_dict(output.to_dict())
    return ToolOutput.from_dict(output.to_dict())


def tool_output_snapshot(output: ToolOutput) -> dict[str, Any]:
    return normalize_tool_output(output).to_dict()


def tool_output_from_snapshot(snapshot: Mapping[str, Any]) -> ToolOutput:
    kind = snapshot.get("kind")
    if kind == "observation":
        return ToolObservation.from_dict(snapshot)
    if kind == "acceptance":
        return ToolAcceptance.from_dict(snapshot)
    if kind == "rejection":
        return ToolRejection.from_dict(snapshot)
    return ToolOutput.from_dict(snapshot)


def is_prefix_tool_batch(batch: ToolBatch, snapshots: tuple[Mapping[str, Any], ...]) -> bool:
    if not batch.calls or len(batch.calls) > len(snapshots):
        return False
    expected = snapshots[: len(batch.calls)]
    return all(
        actual.to_dict() == dict(pending)
        for actual, pending in zip(batch.calls, expected, strict=True)
    )


def batch_call_index(snapshots: tuple[Mapping[str, Any], ...], call: ToolCall) -> int:
    call_data = call.to_dict()
    for index, snapshot in enumerate(snapshots):
        if call_data == dict(snapshot):
            return index
    raise AgentError("tool scheduler attempted to execute a call outside the selected batch")


def validate_scheduler_progress(
    batch: ToolBatch,
    snapshots: tuple[Mapping[str, Any], ...],
    progress: ToolStarted | ToolCompleted,
) -> None:
    if progress.batch.id != batch.id or progress.batch.parallel != batch.parallel:
        raise AgentError("tool scheduler progress batch does not match selected batch")
    if len(progress.batch.calls) != len(snapshots):
        raise AgentError("tool scheduler progress batch calls do not match selected batch")
    if any(
        actual.to_dict() != dict(expected)
        for actual, expected in zip(progress.batch.calls, snapshots, strict=True)
    ):
        raise AgentError("tool scheduler progress batch calls do not match selected batch")
    if progress.index < 0 or progress.index >= len(snapshots):
        raise AgentError("tool scheduler progress index is outside the selected batch")
    if progress.call.to_dict() != dict(snapshots[progress.index]):
        raise AgentError("tool scheduler progress call does not match batch index")


def approval_denial_output(call: ToolCall, reason: str, metadata: Mapping[str, Any]) -> ToolOutput:
    text = f"tool call denied by approval policy: {reason}"
    output_metadata = {"approval": "denied", **dict(metadata)}
    if call.mode == "execute":
        return ToolObservation(
            parts=[ContentPart.text_part(text)],
            metadata=output_metadata,
            is_error=True,
        )
    if call.mode == "accept":
        return ToolRejection.text(text, metadata=output_metadata)
    return ToolOutput(
        kind="tool_error",
        parts=[ContentPart.text_part(text)],
        metadata=output_metadata,
        is_error=True,
    )


def tool_error_output(call: ToolCall, exc: Exception) -> ToolOutput:
    text = str(exc) or exc.__class__.__name__
    metadata = {"error_type": exc.__class__.__name__}
    if call.mode == "execute":
        return ToolObservation(
            parts=[ContentPart.text_part(text)],
            metadata=metadata,
            is_error=True,
        )
    if call.mode == "accept":
        return ToolRejection.text(text, metadata=metadata)
    return ToolOutput(
        kind="tool_error",
        parts=[ContentPart.text_part(text)],
        metadata=metadata,
        is_error=True,
    )


def validate_tool_output_mode(call: ToolCall, output: ToolOutput) -> None:
    if call.mode == "execute" and output.kind != "observation":
        raise AgentError("execute tool call must produce ToolObservation")
    if call.mode == "accept" and output.kind not in {"acceptance", "rejection"}:
        raise AgentError("accept tool call must produce ToolAcceptance or ToolRejection")
    if call.mode not in {"execute", "accept"} and output.kind in {
        "observation",
        "acceptance",
        "rejection",
    }:
        raise AgentError("custom tool call must produce an extension ToolOutput kind")


def validate_non_invoked_tool_output(call: ToolCall, output: ToolOutput) -> None:
    if not output.is_error:
        raise AgentError("non-invoked tool result must remain an error")
    if output.pause is not None:
        raise AgentError("non-invoked tool result must not request pause")
    if call.mode == "accept" and output.kind != "rejection":
        raise AgentError("non-invoked accept tool result must remain a rejection")


def replace_tool_call_in_history(
    messages: list[Message], original_call_id: str, replacement: ToolCall
) -> None:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for index, call in enumerate(message.tool_calls):
            if call.id == original_call_id:
                message.tool_calls[index] = ToolCall.from_dict(replacement.to_dict())
                return


def record_model_usage(state: AgentState, usage: ModelUsage | None) -> None:
    existing = state.total_usage
    if usage is None:
        return
    values: dict[str, int] = {}
    for usage_field in _USAGE_FIELDS:
        current = None if existing is None else getattr(existing, usage_field)
        increment = getattr(usage, usage_field)
        if existing is None:
            if increment is not None:
                values[usage_field] = increment
        elif current is not None and increment is not None:
            values[usage_field] = current + increment
        elif current is not None:
            values[usage_field] = current
        elif increment is not None:
            values[usage_field] = increment
    if not values:
        return
    state.total_usage = ModelUsage(
        input_tokens=values.get("input_tokens"),
        output_tokens=values.get("output_tokens"),
        total_tokens=values.get("total_tokens"),
        reasoning_tokens=values.get("reasoning_tokens"),
        cache_read_tokens=values.get("cache_read_tokens"),
        cache_write_tokens=values.get("cache_write_tokens"),
    )


def clear_pause_request(control: RunControlState) -> None:
    controller = control.run_controller
    if controller is not None and controller.pause_request is not None:
        controller.clear_pause()


def begin_tool_execution(control: RunControlState, tool_call_id: str) -> None:
    controller = control.run_controller
    if controller is not None:
        controller.clear_tool_cancel(tool_call_id)
    control.active_tool_call_ids.add(tool_call_id)


def finish_tool_execution(control: RunControlState, tool_call_id: str) -> None:
    control.active_tool_call_ids.discard(tool_call_id)
    controller = control.run_controller
    if controller is not None:
        controller.clear_tool_cancel(tool_call_id)


def is_active_tool_cancel(control: RunControlState, request: ToolCancelRequest) -> bool:
    controller = control.run_controller
    return (
        controller is not None
        and request.tool_call_id in control.active_tool_call_ids
        and controller.is_tool_cancelled(request.tool_call_id)
    )


def clear_tool_cancel(control: RunControlState, tool_call_id: str) -> None:
    controller = control.run_controller
    if controller is not None:
        controller.clear_tool_cancel(tool_call_id)


def rollback_trace_to_durable(control: RunControlState) -> None:
    if control.trace is not None:
        control.trace.rollback_to_durable()


def restore_state_from_snapshot(state: AgentState, snapshot: RunSnapshot) -> None:
    restored = AgentState.from_dict(snapshot.state.to_dict())
    state.status = restored.status
    state.messages = restored.messages
    state.pending_tool_calls = restored.pending_tool_calls
    state.iterations = restored.iterations
    state.total_tool_calls = restored.total_tool_calls
    state.total_usage = restored.total_usage
    state.final_parts = restored.final_parts
    state.error = restored.error
    state.pause = restored.pause


def build_snapshot(
    state: AgentState,
    context: RuntimeContext,
    control: RunControlState,
    *,
    sequence: int | None = None,
) -> RunSnapshot:
    context_data = context.to_dict()
    context_data["run_id"] = control.run_id
    context_data["started_at"] = control.started_at
    context_data["deadline"] = control.deadline
    context_data["sequence"] = control.sequence if sequence is None else sequence
    return RunSnapshot(
        state=AgentState.from_dict(state.to_dict()),
        context=RuntimeContext.from_dict(context_data),
    )


def child_run_event_data(context: RuntimeContext) -> dict[str, Any]:
    if context.parent_run_id is None:
        raise RuntimeError("child run event requires parent_run_id")
    data: dict[str, Any] = {"parent_run_id": context.parent_run_id}
    if context.parent_tool_call_id is not None:
        data["parent_tool_call_id"] = context.parent_tool_call_id
    if context.run_kind is not None:
        data["run_kind"] = context.run_kind
    return data


def background_task_event_type(task: BackgroundTask) -> str:
    if task.lifecycle == "started":
        return EventTypes.BACKGROUND_TASK_STARTED
    if task.lifecycle == "completed":
        return EventTypes.BACKGROUND_TASK_COMPLETED
    return EventTypes.BACKGROUND_TASK_UPDATED
