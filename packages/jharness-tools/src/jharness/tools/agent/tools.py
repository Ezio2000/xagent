"""Model-visible Agent preset tools backed by a Host-owned supervisor."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

from jharness.kernel import (
    ContentPart,
    SettledResult,
    Suspension,
    TaskRef,
    ToolAccepted,
    ToolCall,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolRisk,
    ToolSpec,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
    thaw_json_value,
)
from jharness.tools.agent._schema import (
    DEFAULT_MAX_AGENT_ID_CHARS,
    DEFAULT_MAX_DESCRIPTION_CHARS,
    DEFAULT_MAX_ERROR_CODE_CHARS,
    DEFAULT_MAX_ERROR_MESSAGE_CHARS,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_MAX_RESULT_CHARS,
    SCHEMA_VERSION,
    AgentContractError,
    agent_id_input_schema,
    agent_input_schema,
    build_wait_id,
    normalize_agent_id,
    normalize_agent_request,
    positive_int,
    snapshot_output_schema,
    snapshot_payload,
)
from jharness.tools.agent.backend import AgentBackend
from jharness.tools.agent.models import AgentBackendError, AgentRequest, AgentSnapshot

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _OutputLimits:
    max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS
    max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS
    max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS

    def __post_init__(self) -> None:
        for name in (
            "max_agent_id_chars",
            "max_description_chars",
            "max_result_chars",
            "max_error_code_chars",
            "max_error_message_chars",
        ):
            positive_int(getattr(self, name), name)

    def payload(self, snapshot: AgentSnapshot) -> dict[str, object]:
        return cast(
            dict[str, object],
            snapshot_payload(
                snapshot,
                max_agent_id_chars=self.max_agent_id_chars,
                max_description_chars=self.max_description_chars,
                max_result_chars=self.max_result_chars,
                max_error_code_chars=self.max_error_code_chars,
                max_error_message_chars=self.max_error_message_chars,
            ),
        )

    def output_schema(self) -> dict[str, object]:
        return snapshot_output_schema(
            max_agent_id_chars=self.max_agent_id_chars,
            max_description_chars=self.max_description_chars,
            max_result_chars=self.max_result_chars,
            max_error_code_chars=self.max_error_code_chars,
            max_error_message_chars=self.max_error_message_chars,
        )

    def suspension_metadata(self, agent_id: str, tool_call_id: str) -> dict[str, object]:
        return {
            "agent_id": agent_id,
            "schema_version": SCHEMA_VERSION,
            "tool_call_id": tool_call_id,
            "max_agent_id_chars": self.max_agent_id_chars,
            "max_description_chars": self.max_description_chars,
            "max_result_chars": self.max_result_chars,
            "max_error_code_chars": self.max_error_code_chars,
            "max_error_message_chars": self.max_error_message_chars,
        }


@dataclass(frozen=True, slots=True, init=False)
class AgentTool:
    """Start one Host-owned Child Run and optionally wait for its completion."""

    backend: AgentBackend = field(repr=False)
    max_agent_id_chars: int
    max_description_chars: int
    max_prompt_chars: int
    max_result_chars: int
    max_error_code_chars: int
    max_error_message_chars: int
    _limits: _OutputLimits = field(repr=False)
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        backend: AgentBackend,
        *,
        max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
        max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
    ) -> None:
        backend = _backend(backend)
        max_prompt_chars = positive_int(max_prompt_chars, "max_prompt_chars")
        limits = _OutputLimits(
            max_agent_id_chars,
            max_description_chars,
            max_result_chars,
            max_error_code_chars,
            max_error_message_chars,
        )
        _set_common(self, backend, limits)
        object.__setattr__(self, "max_prompt_chars", max_prompt_chars)
        object.__setattr__(
            self,
            "spec",
            _agent_spec(limits, max_prompt_chars=max_prompt_chars),
        )

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.cancel_requested:
            return _failure("cancelled", "Agent was cancelled before it started.")
        try:
            request = normalize_agent_request(
                thaw_json_value(call.arguments, label="Agent arguments"),
                max_description_chars=self.max_description_chars,
                max_prompt_chars=self.max_prompt_chars,
            )
        except AgentContractError as exc:
            return _failure("invalid_agent_request", str(exc))
        try:
            snapshot = await self.backend.start_or_get(
                request,
                parent=context.run,
                parent_tool_call_id=call.id,
            )
        except AgentBackendError as exc:
            return _failure(exc.code, exc.message)
        except Exception:
            return _backend_failure("start_or_get")
        invalid_snapshot = _invalid_start_snapshot(snapshot, request)
        if invalid_snapshot is not None:
            return invalid_snapshot
        try:
            payload = self._limits.payload(snapshot)
        except AgentContractError as exc:
            return _failure("invalid_agent_snapshot", str(exc))
        if snapshot.status in _TERMINAL_STATUSES:
            return _snapshot_success(snapshot, payload)
        if request.background:
            return SettledResult(
                ToolAccepted(
                    (ContentPart.text_part(_accepted_text(snapshot)),),
                    correlation_id=snapshot.agent_id,
                    task=_task_ref(snapshot),
                    structured_content=payload,
                )
            )
        return _waiting_result(
            snapshot,
            payload,
            source="Agent",
            call=call,
            context=context,
            limits=self._limits,
        )


@dataclass(frozen=True, slots=True, init=False)
class AgentGetTool:
    """Return the latest Host-owned snapshot without blocking."""

    backend: AgentBackend = field(repr=False)
    max_agent_id_chars: int
    max_description_chars: int
    max_result_chars: int
    max_error_code_chars: int
    max_error_message_chars: int
    _limits: _OutputLimits = field(repr=False)
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        backend: AgentBackend,
        *,
        max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
        max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
    ) -> None:
        backend = _backend(backend)
        limits = _OutputLimits(
            max_agent_id_chars,
            max_description_chars,
            max_result_chars,
            max_error_code_chars,
            max_error_message_chars,
        )
        _set_common(self, backend, limits)
        object.__setattr__(self, "spec", _get_spec(limits))

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.cancel_requested:
            return _failure("cancelled", "AgentGet was cancelled.")
        agent_id = _agent_id(call, self.max_agent_id_chars)
        if isinstance(agent_id, SettledResult):
            return agent_id
        try:
            snapshot = await self.backend.get(agent_id, requester=context.run)
        except AgentBackendError as exc:
            return _failure(exc.code, exc.message)
        except Exception:
            return _backend_failure("get")
        return _validated_snapshot_success(snapshot, agent_id, self._limits)


@dataclass(frozen=True, slots=True, init=False)
class AgentWaitTool:
    """Wait durably for one background Agent without model polling."""

    backend: AgentBackend = field(repr=False)
    max_agent_id_chars: int
    max_description_chars: int
    max_result_chars: int
    max_error_code_chars: int
    max_error_message_chars: int
    _limits: _OutputLimits = field(repr=False)
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        backend: AgentBackend,
        *,
        max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
        max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
    ) -> None:
        backend = _backend(backend)
        limits = _OutputLimits(
            max_agent_id_chars,
            max_description_chars,
            max_result_chars,
            max_error_code_chars,
            max_error_message_chars,
        )
        _set_common(self, backend, limits)
        object.__setattr__(self, "spec", _wait_spec(limits))

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.cancel_requested:
            return _failure("cancelled", "AgentWait was cancelled.")
        agent_id = _agent_id(call, self.max_agent_id_chars)
        if isinstance(agent_id, SettledResult):
            return agent_id
        try:
            snapshot = await self.backend.wait_or_get(
                agent_id,
                requester=context.run,
                requester_tool_call_id=call.id,
            )
        except AgentBackendError as exc:
            return _failure(exc.code, exc.message)
        except Exception:
            return _backend_failure("wait_or_get")
        if not isinstance(cast(object, snapshot), AgentSnapshot):
            return _invalid_snapshot_type()
        if snapshot.agent_id != agent_id:
            return _failure(
                "invalid_agent_snapshot",
                "Agent backend returned a snapshot for a different agent.",
            )
        try:
            payload = self._limits.payload(snapshot)
        except AgentContractError as exc:
            return _failure("invalid_agent_snapshot", str(exc))
        if snapshot.status in _TERMINAL_STATUSES:
            return _snapshot_success(snapshot, payload)
        return _waiting_result(
            snapshot,
            payload,
            source="AgentWait",
            call=call,
            context=context,
            limits=self._limits,
        )


@dataclass(frozen=True, slots=True, init=False)
class AgentCancelTool:
    """Request idempotent Host-owned cancellation for one Agent."""

    backend: AgentBackend = field(repr=False)
    max_agent_id_chars: int
    max_description_chars: int
    max_result_chars: int
    max_error_code_chars: int
    max_error_message_chars: int
    _limits: _OutputLimits = field(repr=False)
    spec: ToolSpec = field(repr=False)

    def __init__(
        self,
        backend: AgentBackend,
        *,
        max_agent_id_chars: int = DEFAULT_MAX_AGENT_ID_CHARS,
        max_description_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
        max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
        max_error_code_chars: int = DEFAULT_MAX_ERROR_CODE_CHARS,
        max_error_message_chars: int = DEFAULT_MAX_ERROR_MESSAGE_CHARS,
    ) -> None:
        backend = _backend(backend)
        limits = _OutputLimits(
            max_agent_id_chars,
            max_description_chars,
            max_result_chars,
            max_error_code_chars,
            max_error_message_chars,
        )
        _set_common(self, backend, limits)
        object.__setattr__(self, "spec", _cancel_spec(limits))

    async def invoke(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.cancel_requested:
            return _failure("cancelled", "AgentCancel was cancelled.")
        agent_id = _agent_id(call, self.max_agent_id_chars)
        if isinstance(agent_id, SettledResult):
            return agent_id
        try:
            snapshot = await self.backend.cancel(
                agent_id,
                requester=context.run,
                requester_tool_call_id=call.id,
            )
        except AgentBackendError as exc:
            return _failure(exc.code, exc.message)
        except Exception:
            return _backend_failure("cancel")
        if not isinstance(cast(object, snapshot), AgentSnapshot):
            return _invalid_snapshot_type()
        if snapshot.agent_id != agent_id:
            return _failure(
                "invalid_agent_snapshot",
                "Agent backend returned a snapshot for a different agent.",
            )
        if snapshot.status not in _TERMINAL_STATUSES and not snapshot.cancellation_requested:
            return _failure(
                "invalid_agent_snapshot",
                "Agent backend did not acknowledge the cancellation request.",
            )
        return _validated_snapshot_success(snapshot, agent_id, self._limits)


def _set_common(tool: object, backend: AgentBackend, limits: _OutputLimits) -> None:
    object.__setattr__(tool, "backend", backend)
    object.__setattr__(tool, "max_agent_id_chars", limits.max_agent_id_chars)
    object.__setattr__(tool, "max_description_chars", limits.max_description_chars)
    object.__setattr__(tool, "max_result_chars", limits.max_result_chars)
    object.__setattr__(tool, "max_error_code_chars", limits.max_error_code_chars)
    object.__setattr__(tool, "max_error_message_chars", limits.max_error_message_chars)
    object.__setattr__(tool, "_limits", limits)


def _backend(value: object) -> AgentBackend:
    if not isinstance(value, AgentBackend):
        raise TypeError("backend must implement AgentBackend")
    return value


def _agent_id(call: ToolCall, maximum: int) -> str | SettledResult:
    try:
        return normalize_agent_id(
            thaw_json_value(call.arguments, label=f"{call.name} arguments"),
            max_agent_id_chars=maximum,
        )
    except AgentContractError as exc:
        return _failure("invalid_agent_id", str(exc))


def _validated_snapshot_success(
    snapshot: AgentSnapshot,
    expected_agent_id: str,
    limits: _OutputLimits,
) -> SettledResult:
    if not isinstance(cast(object, snapshot), AgentSnapshot):
        return _invalid_snapshot_type()
    if snapshot.agent_id != expected_agent_id:
        return _failure(
            "invalid_agent_snapshot",
            "Agent backend returned a snapshot for a different agent.",
        )
    try:
        payload = limits.payload(snapshot)
    except AgentContractError as exc:
        return _failure("invalid_agent_snapshot", str(exc))
    return _snapshot_success(snapshot, payload)


def _invalid_start_snapshot(
    snapshot: object,
    request: AgentRequest,
) -> SettledResult | None:
    if not isinstance(snapshot, AgentSnapshot):
        return _invalid_snapshot_type()
    if snapshot.background != request.background:
        return _failure(
            "invalid_agent_snapshot",
            "Agent backend returned a snapshot with the wrong background mode.",
        )
    if snapshot.description != request.description:
        return _failure(
            "invalid_agent_snapshot",
            "Agent backend returned a snapshot with the wrong description.",
        )
    return None


def _snapshot_success(snapshot: AgentSnapshot, payload: dict[str, object]) -> SettledResult:
    return SettledResult(
        ToolSuccess(
            (ContentPart.text_part(_snapshot_text(snapshot)),),
            structured_content=payload,
        )
    )


def _waiting_result(
    snapshot: AgentSnapshot,
    payload: dict[str, object],
    *,
    source: str,
    call: ToolCall,
    context: ToolContext,
    limits: _OutputLimits,
) -> WaitingResult:
    wait_id = build_wait_id(context.run.run_id, call.id)
    return WaitingResult(
        ToolWaiting(
            (ContentPart.text_part(f"Waiting for agent {snapshot.agent_id} to complete."),),
            task=_task_ref(snapshot),
            structured_content=payload,
        ),
        Suspension(
            reason="agent_completion",
            source=source,
            wait_id=wait_id,
            metadata=limits.suspension_metadata(snapshot.agent_id, call.id),
        ),
    )


def _task_ref(snapshot: AgentSnapshot) -> TaskRef:
    return TaskRef(
        snapshot.agent_id,
        snapshot.status,
        {"background": snapshot.background, "kind": "agent"},
    )


def _accepted_text(snapshot: AgentSnapshot) -> str:
    return f"Agent {snapshot.agent_id} accepted for background execution ({snapshot.status})."


def _snapshot_text(snapshot: AgentSnapshot) -> str:
    if snapshot.status == "completed":
        result = snapshot.result
        if result:
            return f"Agent {snapshot.agent_id} completed.\n\n{result}"
        return f"Agent {snapshot.agent_id} completed with an empty result."
    if snapshot.status == "failed":
        error = snapshot.error
        if error is None:
            raise AgentContractError("failed Agent snapshot has no error")
        return f"Agent {snapshot.agent_id} failed ({error.code}): {error.message}"
    if snapshot.status == "cancelled":
        return f"Agent {snapshot.agent_id} was cancelled."
    suffix = "; cancellation requested" if snapshot.cancellation_requested else ""
    return f"Agent {snapshot.agent_id} is {snapshot.status}{suffix}."


def _failure(code: str, message: str) -> SettledResult:
    return SettledResult(ToolFailure.from_error(code, message))


def _invalid_snapshot_type() -> SettledResult:
    return _failure(
        "invalid_agent_snapshot",
        "Agent backend must return an AgentSnapshot.",
    )


def _backend_failure(operation: str) -> SettledResult:
    _LOGGER.exception("Unexpected Agent backend failure during %s", operation)
    return _failure(
        "agent_backend_error",
        "The Host Agent backend failed while processing the request.",
    )


def _agent_spec(limits: _OutputLimits, *, max_prompt_chars: int) -> ToolSpec:
    return ToolSpec(
        name="Agent",
        description=(
            "Delegate a self-contained task to a Host-owned child Agent. The child inherits "
            "its effective Runtime configuration from the parent under Host policy. Set "
            "background=true to continue immediately; otherwise this must be the only tool "
            "call in the assistant turn because the parent waits durably for completion."
        ),
        input_schema=agent_input_schema(
            max_description_chars=limits.max_description_chars,
            max_prompt_chars=max_prompt_chars,
        ),
        output_schema=limits.output_schema(),
        execution=ToolExecution(concurrency="serial", read_only=False, idempotent=False),
        risk=ToolRisk(
            requires_approval=True,
            extra={"delegates_child_run": True},
        ),
    )


def _get_spec(limits: _OutputLimits) -> ToolSpec:
    return ToolSpec(
        name="AgentGet",
        description=(
            "Return the latest snapshot for an Agent without waiting. Use AgentWait when the "
            "parent should sleep durably until the Agent reaches a terminal state."
        ),
        input_schema=agent_id_input_schema(max_agent_id_chars=limits.max_agent_id_chars),
        output_schema=limits.output_schema(),
        execution=ToolExecution(concurrency="parallel", read_only=True, idempotent=True),
        risk=_read_only_risk(),
    )


def _wait_spec(limits: _OutputLimits) -> ToolSpec:
    return ToolSpec(
        name="AgentWait",
        description=(
            "Wait durably for a background Agent. If it is still active, this must be the only "
            "tool call in the assistant turn so the Host can append its terminal result when "
            "resuming the parent."
        ),
        input_schema=agent_id_input_schema(max_agent_id_chars=limits.max_agent_id_chars),
        output_schema=limits.output_schema(),
        execution=ToolExecution(concurrency="serial", read_only=True, idempotent=True),
        risk=_read_only_risk(),
    )


def _cancel_spec(limits: _OutputLimits) -> ToolSpec:
    return ToolSpec(
        name="AgentCancel",
        description=(
            "Idempotently request cancellation of an Agent. A non-terminal returned snapshot "
            "means cancellation was acknowledged but has not yet reached a safe terminal point."
        ),
        input_schema=agent_id_input_schema(max_agent_id_chars=limits.max_agent_id_chars),
        output_schema=limits.output_schema(),
        execution=ToolExecution(concurrency="serial", read_only=False, idempotent=True),
        risk=ToolRisk(
            filesystem="none",
            network="none",
            subprocess=False,
            destructive=True,
            requires_approval=False,
            extra={"agent_action": "cancel"},
        ),
    )


def _read_only_risk() -> ToolRisk:
    return ToolRisk(
        filesystem="none",
        network="none",
        subprocess=False,
        destructive=False,
        requires_approval=False,
    )


__all__ = ["AgentCancelTool", "AgentGetTool", "AgentTool", "AgentWaitTool"]
