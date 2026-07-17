"""Host-side extraction and resume helpers for Agent completion waits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from jharness.kernel import (
    Checkpoint,
    Invocation,
    Message,
    Planning,
    Runtime,
    Suspended,
    SuspensionSelector,
    ToolBatchFact,
    ToolCall,
    ToolOutcomeKind,
    ToolWaiting,
    thaw_json_value,
)
from jharness.tools.agent._schema import SCHEMA_VERSION, build_wait_id, snapshot_payload
from jharness.tools.agent.models import AgentSnapshot, AgentStatus

AgentWaitSource: TypeAlias = Literal["Agent", "AgentWait"]
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_CONTRACT_FIELDS = (
    "max_agent_id_chars",
    "max_description_chars",
    "max_result_chars",
    "max_error_code_chars",
    "max_error_message_chars",
)


@dataclass(frozen=True, slots=True)
class AgentWaitRequest:
    """One immutable Agent completion wait extracted from a checkpoint."""

    wait_id: str
    source: AgentWaitSource
    tool_call_id: str
    snapshot: AgentSnapshot
    max_agent_id_chars: int
    max_description_chars: int
    max_result_chars: int
    max_error_code_chars: int
    max_error_message_chars: int

    def __post_init__(self) -> None:
        _non_empty_string(self.wait_id, "wait_id")
        if self.source not in {"Agent", "AgentWait"}:
            raise ValueError("source must be 'Agent' or 'AgentWait'")
        _non_empty_string(self.tool_call_id, "tool_call_id")
        if not isinstance(cast(object, self.snapshot), AgentSnapshot):
            raise TypeError("snapshot must be an AgentSnapshot")
        if self.snapshot.status in _TERMINAL_STATUSES:
            raise ValueError("an Agent wait request requires a non-terminal snapshot")
        for field_name in _CONTRACT_FIELDS:
            _positive_int(getattr(self, field_name), field_name)

    @property
    def agent_id(self) -> str:
        return self.snapshot.agent_id


def extract_agent_wait(checkpoint: Checkpoint) -> AgentWaitRequest:
    """Extract and validate a foreground ``Agent`` or ``AgentWait`` suspension."""

    if not isinstance(cast(object, checkpoint), Checkpoint):
        raise TypeError("checkpoint must be a Checkpoint")
    state, source = _agent_suspension(checkpoint)
    wait_id, tool_call_id, agent_id, limits = _suspension_contract(checkpoint, state)
    waiting, assistant_call = _current_waiting_outcome(
        checkpoint,
        tool_call_id,
        source,
    )
    payload = _mapping(
        thaw_json_value(
            waiting.structured_content,
            label="Agent waiting structured_content",
        ),
        "Agent waiting structured_content",
    )
    snapshot = _waiting_snapshot(payload, limits)
    _validate_waiting_identity(waiting, assistant_call, snapshot, agent_id)

    return AgentWaitRequest(
        wait_id=wait_id,
        source=source,
        tool_call_id=tool_call_id,
        snapshot=snapshot,
        max_agent_id_chars=limits["max_agent_id_chars"],
        max_description_chars=limits["max_description_chars"],
        max_result_chars=limits["max_result_chars"],
        max_error_code_chars=limits["max_error_code_chars"],
        max_error_message_chars=limits["max_error_message_chars"],
    )


def agent_completion_message(
    request: AgentWaitRequest,
    snapshot: AgentSnapshot,
) -> Message:
    """Create the deterministic external message used to resume a waiting parent."""

    if not isinstance(cast(object, request), AgentWaitRequest):
        raise TypeError("request must be an AgentWaitRequest")
    if not isinstance(cast(object, snapshot), AgentSnapshot):
        raise TypeError("snapshot must be an AgentSnapshot")
    if snapshot.status not in _TERMINAL_STATUSES:
        raise ValueError("Agent completion requires a terminal snapshot")
    if snapshot.agent_id != request.agent_id:
        raise ValueError("completion agent_id does not match the Agent wait request")
    if snapshot.description != request.snapshot.description:
        raise ValueError("completion description does not match the Agent wait request")
    if snapshot.background != request.snapshot.background:
        raise ValueError("completion background mode does not match the Agent wait request")

    payload = snapshot_payload(
        snapshot,
        max_agent_id_chars=request.max_agent_id_chars,
        max_description_chars=request.max_description_chars,
        max_result_chars=request.max_result_chars,
        max_error_code_chars=request.max_error_code_chars,
        max_error_message_chars=request.max_error_message_chars,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return Message.external(
        f"Agent completion:\n{encoded}",
        metadata=_completion_metadata(request, snapshot),
    )


def resume_agent(
    runtime: Runtime,
    checkpoint: Checkpoint,
    snapshot: AgentSnapshot,
    *,
    stream: bool = False,
) -> Invocation:
    """Validate a terminal Agent snapshot and resume its waiting parent Run."""

    if not isinstance(cast(object, runtime), Runtime):
        raise TypeError("runtime must be a Runtime")
    if not isinstance(cast(object, stream), bool):
        raise TypeError("stream must be bool")
    request = extract_agent_wait(checkpoint)
    message = agent_completion_message(request, snapshot)
    state = cast(Suspended, checkpoint.snapshot.state)
    suspension = state.suspension
    return runtime.resume(
        checkpoint,
        selector=SuspensionSelector(
            reason=suspension.reason,
            source=suspension.source,
            wait_id=suspension.wait_id,
            metadata=suspension.metadata,
        ),
        append_messages=(message,),
        metadata=_completion_metadata(request, snapshot),
        stream=stream,
    )


def _current_waiting_outcome(
    checkpoint: Checkpoint,
    tool_call_id: str,
    source: AgentWaitSource,
) -> tuple[ToolWaiting, ToolCall]:
    fact = checkpoint.fact
    if not isinstance(fact, ToolBatchFact):
        raise ValueError("Agent checkpoint must end at its waiting tool batch")
    if fact.call_ids != (tool_call_id,) or fact.outcome_kinds != (ToolOutcomeKind.WAITING,):
        raise ValueError("Agent suspension does not match the current waiting tool call")
    history = checkpoint.snapshot.history
    if len(history) < 2:
        raise ValueError("Agent checkpoint is missing its assistant and tool messages")
    tool_message = history[-1]
    assistant_message = history[-2]
    if tool_message.role != "tool" or tool_message.tool_call_id != tool_call_id:
        raise ValueError("Agent checkpoint has no matching trailing tool message")
    if assistant_message.role != "assistant" or len(assistant_message.tool_calls) != 1:
        raise ValueError("Agent completion tools must be the only call in their assistant turn")
    assistant_call = assistant_message.tool_calls[0]
    if assistant_call.id != tool_call_id or assistant_call.name != source:
        raise ValueError("Agent checkpoint assistant call has the wrong identity")
    outcome = tool_message.outcome
    if not isinstance(outcome, ToolWaiting):
        raise ValueError("current Agent tool message must contain ToolWaiting")
    return outcome, assistant_call


def _agent_suspension(checkpoint: Checkpoint) -> tuple[Suspended, AgentWaitSource]:
    state = checkpoint.snapshot.state
    if not isinstance(state, Suspended):
        raise ValueError("checkpoint must be suspended for an Agent completion")
    suspension = state.suspension
    if suspension.reason != "agent_completion" or suspension.source not in {
        "Agent",
        "AgentWait",
    }:
        raise ValueError("checkpoint is not an Agent completion suspension")
    if not isinstance(state.resume_to, Planning):
        raise ValueError("Agent completion waits must resume to Planning")
    return state, cast(AgentWaitSource, suspension.source)


def _suspension_contract(
    checkpoint: Checkpoint,
    state: Suspended,
) -> tuple[str, str, str, dict[str, int]]:
    suspension = state.suspension
    metadata = suspension.metadata
    expected_metadata = frozenset(
        {
            "agent_id",
            "schema_version",
            "tool_call_id",
            *_CONTRACT_FIELDS,
        }
    )
    if frozenset(metadata) != expected_metadata:
        raise ValueError("Agent suspension metadata has unexpected fields")
    if _exact_int(metadata["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Agent suspension schema_version")
    tool_call_id = _non_empty_string(metadata["tool_call_id"], "tool_call_id")
    agent_id = _non_empty_string(metadata["agent_id"], "agent_id")
    wait_id = _non_empty_string(suspension.wait_id, "wait_id")
    if wait_id != build_wait_id(checkpoint.snapshot.context.run_id, tool_call_id):
        raise ValueError("Agent suspension wait_id does not match its run and tool call")
    limits = {name: _positive_int(metadata[name], name) for name in _CONTRACT_FIELDS}
    return wait_id, tool_call_id, agent_id, limits


def _validate_waiting_identity(
    waiting: ToolWaiting,
    assistant_call: ToolCall,
    snapshot: AgentSnapshot,
    suspension_agent_id: str,
) -> None:
    if snapshot.agent_id != suspension_agent_id:
        raise ValueError("Agent waiting payload does not match suspension agent_id")
    if waiting.task is None:
        raise ValueError("Agent waiting outcome requires a task reference")
    if waiting.task.id != snapshot.agent_id or waiting.task.status != snapshot.status:
        raise ValueError("Agent task reference does not match its waiting snapshot")
    if assistant_call.name == "Agent":
        _validate_foreground_agent_call(assistant_call, snapshot)
        return
    _validate_agent_wait_call(assistant_call, snapshot)


def _validate_foreground_agent_call(
    assistant_call: ToolCall,
    snapshot: AgentSnapshot,
) -> None:
    if frozenset(assistant_call.arguments) - {"description", "prompt", "background"}:
        raise ValueError("foreground Agent arguments contain unexpected fields")
    if assistant_call.arguments.get("description") != snapshot.description:
        raise ValueError("foreground Agent description does not match its waiting snapshot")
    prompt = assistant_call.arguments.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("foreground Agent prompt must be a non-empty string")
    if assistant_call.arguments.get("background", False) is not False:
        raise ValueError("a foreground Agent suspension cannot be a background request")


def _validate_agent_wait_call(
    assistant_call: ToolCall,
    snapshot: AgentSnapshot,
) -> None:
    if frozenset(assistant_call.arguments) != frozenset({"agent_id"}):
        raise ValueError("AgentWait arguments must contain only agent_id")
    if assistant_call.arguments.get("agent_id") != snapshot.agent_id:
        raise ValueError("AgentWait arguments do not match its waiting snapshot")


def _waiting_snapshot(
    payload: Mapping[str, Any],
    limits: Mapping[str, int],
) -> AgentSnapshot:
    status = payload.get("status")
    if status not in {"queued", "running"}:
        raise ValueError("Agent waiting payload must contain a non-terminal status")
    try:
        snapshot = AgentSnapshot(
            agent_id=cast(str, payload.get("agent_id")),
            description=cast(str, payload.get("description")),
            status=cast(AgentStatus, status),
            background=cast(bool, payload.get("background")),
            cancellation_requested=cast(bool, payload.get("cancellation_requested")),
        )
        canonical = snapshot_payload(
            snapshot,
            max_agent_id_chars=limits["max_agent_id_chars"],
            max_description_chars=limits["max_description_chars"],
            max_result_chars=limits["max_result_chars"],
            max_error_code_chars=limits["max_error_code_chars"],
            max_error_message_chars=limits["max_error_message_chars"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Agent waiting payload is invalid") from exc
    if canonical != dict(payload):
        raise ValueError("Agent waiting payload is not canonical")
    return snapshot


def _completion_metadata(
    request: AgentWaitRequest,
    snapshot: AgentSnapshot,
) -> Mapping[str, Any]:
    return {
        "agent_id": snapshot.agent_id,
        "kind": "agent_completion",
        "status": snapshot.status,
        "wait_id": request.wait_id,
    }


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} keys must be strings")
    return cast(Mapping[str, Any], raw)


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be a positive integer")
    return result
