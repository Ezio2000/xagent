# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field, replace
from typing import Any, cast

import pytest

import jharness.tools as tools
import jharness.tools.agent as agent_api
from jharness.kernel import (
    ErrorInfo,
    RunContext,
    SettledResult,
    TaskRef,
    ToolAccepted,
    ToolCall,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolResult,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
    thaw_json_value,
)
from jharness.toolkit import Tool, ToolRegistry
from jharness.tools.agent import (
    AgentBackend,
    AgentBackendError,
    AgentCancelTool,
    AgentGetTool,
    AgentRequest,
    AgentSnapshot,
    AgentTool,
    AgentWaitTool,
)
from jharness.tools.agent import _schema as agent_schema
from jharness.tools.agent import tools as agent_tools_module


@dataclass(slots=True)
class _FakeBackend:
    """Strict in-memory Host boundary; the tools never own Agent state."""

    snapshots: dict[str, AgentSnapshot] = field(default_factory=dict[str, AgentSnapshot])
    owners: dict[str, str] = field(default_factory=dict[str, str])
    starts: dict[tuple[str, str], tuple[AgentRequest, str]] = field(
        default_factory=dict[tuple[str, str], tuple[AgentRequest, str]]
    )
    start_calls: list[tuple[AgentRequest, RunContext, str]] = field(
        default_factory=list[tuple[AgentRequest, RunContext, str]]
    )
    get_calls: list[tuple[str, RunContext]] = field(default_factory=list[tuple[str, RunContext]])
    wait_calls: list[tuple[str, RunContext, str]] = field(
        default_factory=list[tuple[str, RunContext, str]]
    )
    cancel_calls: list[tuple[str, RunContext, str]] = field(
        default_factory=list[tuple[str, RunContext, str]]
    )
    errors: dict[str, Exception] = field(default_factory=dict[str, Exception])
    forced: dict[str, AgentSnapshot] = field(default_factory=dict[str, AgentSnapshot])
    _next_agent_number: int = 1

    def seed(self, snapshot: AgentSnapshot, *, owner_run_id: str = "parent-run") -> None:
        self.snapshots[snapshot.agent_id] = snapshot
        self.owners[snapshot.agent_id] = owner_run_id

    def fail(self, operation: str, code: str = "agent_store_unavailable") -> None:
        self.errors[operation] = AgentBackendError(code, f"{operation} failed safely")

    def crash(self, operation: str) -> None:
        self.errors[operation] = RuntimeError(f"sensitive {operation} backend detail")

    def force(self, operation: str, snapshot: AgentSnapshot) -> None:
        self.forced[operation] = snapshot

    @property
    def created_count(self) -> int:
        return len(self.starts)

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        self.start_calls.append((request, parent, parent_tool_call_id))
        self._raise_configured("start")
        key = (parent.run_id, parent_tool_call_id)
        existing = self.starts.get(key)
        if existing is not None:
            previous_request, agent_id = existing
            if previous_request != request:
                raise AgentBackendError(
                    "agent_conflict",
                    "The parent tool call already owns a different Agent request.",
                )
            return self.snapshots[agent_id]

        snapshot = self.forced.get("start")
        if snapshot is not None and not isinstance(cast(object, snapshot), AgentSnapshot):
            return snapshot
        if snapshot is None:
            agent_id = f"agent-{self._next_agent_number}"
            self._next_agent_number += 1
            snapshot = AgentSnapshot(
                agent_id,
                request.description,
                "queued",
                request.background,
            )
        self.starts[key] = (request, snapshot.agent_id)
        self.seed(snapshot, owner_run_id=parent.run_id)
        return snapshot

    async def get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
    ) -> AgentSnapshot:
        self.get_calls.append((agent_id, requester))
        self._raise_configured("get")
        self._require_owned(agent_id, requester)
        return self.forced.get("get", self.snapshots[agent_id])

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        self.wait_calls.append((agent_id, requester, requester_tool_call_id))
        self._raise_configured("wait")
        self._require_owned(agent_id, requester)
        return self.forced.get("wait", self.snapshots[agent_id])

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        self.cancel_calls.append((agent_id, requester, requester_tool_call_id))
        self._raise_configured("cancel")
        self._require_owned(agent_id, requester)
        forced = self.forced.get("cancel")
        if forced is not None:
            return forced
        snapshot = self.snapshots[agent_id]
        if snapshot.status not in {"completed", "failed", "cancelled"}:
            snapshot = replace(snapshot, cancellation_requested=True)
            self.snapshots[agent_id] = snapshot
        return snapshot

    def transition(self, snapshot: AgentSnapshot) -> None:
        if snapshot.agent_id not in self.snapshots:
            raise KeyError(snapshot.agent_id)
        self.snapshots[snapshot.agent_id] = snapshot

    def _raise_configured(self, operation: str) -> None:
        error = self.errors.get(operation)
        if error is not None:
            raise error

    def _require_owned(self, agent_id: str, requester: RunContext) -> None:
        if agent_id not in self.snapshots or self.owners[agent_id] != requester.run_id:
            raise AgentBackendError("agent_not_found", "Agent not found.")


async def _emit_progress(_progress: Mapping[str, Any]) -> None:
    return None


def _context(*, run_id: str = "parent-run", cancelled: bool = False) -> ToolContext:
    now = time.time()
    return ToolContext(
        RunContext(run_id, now, now + 60),
        _emit_progress,
        lambda: cancelled,
    )


def _invoke(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    call_id: str = "agent-call",
    run_id: str = "parent-run",
    cancelled: bool = False,
    through_registry: bool = False,
) -> ToolResult:
    async def invoke() -> ToolResult:
        call = ToolCall(call_id, tool.spec.name, arguments)
        context = _context(run_id=run_id, cancelled=cancelled)
        if through_registry:
            catalog = await ToolRegistry((tool,)).open_catalog()
            return await catalog.bind(call).invoke(context)
        return await tool.invoke(call, context)

    return asyncio.run(invoke())


def _success(result: ToolResult) -> tuple[str, dict[str, object]]:
    assert isinstance(result, SettledResult)
    outcome = result.outcome
    assert isinstance(outcome, ToolSuccess)
    text = outcome.parts[0].text
    assert text is not None
    payload = thaw_json_value(outcome.structured_content)
    assert isinstance(payload, dict)
    return text, cast(dict[str, object], payload)


def _failure(result: ToolResult, code: str) -> ToolFailure:
    assert isinstance(result, SettledResult)
    outcome = result.outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == code
    assert outcome.structured_content is None
    return outcome


def _completed(
    agent_id: str = "agent-1",
    *,
    description: str = "Inspect auth",
    background: bool = True,
    result: str = "No issues found.",
) -> AgentSnapshot:
    return AgentSnapshot(agent_id, description, "completed", background, result=result)


def _failed(
    agent_id: str = "agent-1",
    *,
    description: str = "Inspect auth",
    background: bool = True,
) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id,
        description,
        "failed",
        background,
        error=ErrorInfo("child_failed", "Child run failed safely."),
    )


def _cancelled(
    agent_id: str = "agent-1",
    *,
    description: str = "Inspect auth",
    background: bool = True,
) -> AgentSnapshot:
    return AgentSnapshot(
        agent_id,
        description,
        "cancelled",
        background,
        cancellation_requested=True,
    )


def _forge_snapshot(**changes: object) -> AgentSnapshot:
    values: dict[str, object] = {
        "agent_id": "agent-1",
        "description": "Inspect auth",
        "status": "running",
        "background": True,
        "result": None,
        "error": None,
        "cancellation_requested": False,
    }
    values.update(changes)
    snapshot = object.__new__(AgentSnapshot)
    for name, value in values.items():
        object.__setattr__(snapshot, name, value)
    return snapshot


def test_agent_public_api_backend_protocol_specs_and_schemas() -> None:
    backend = _FakeBackend()
    presets: tuple[Tool, ...] = (
        AgentTool(
            backend,
            max_agent_id_chars=8,
            max_description_chars=9,
            max_prompt_chars=10,
            max_result_chars=11,
            max_error_code_chars=12,
            max_error_message_chars=13,
        ),
        AgentGetTool(
            backend,
            max_agent_id_chars=8,
            max_description_chars=9,
            max_result_chars=11,
            max_error_code_chars=12,
            max_error_message_chars=13,
        ),
        AgentWaitTool(
            backend,
            max_agent_id_chars=8,
            max_description_chars=9,
            max_result_chars=11,
            max_error_code_chars=12,
            max_error_message_chars=13,
        ),
        AgentCancelTool(
            backend,
            max_agent_id_chars=8,
            max_description_chars=9,
            max_result_chars=11,
            max_error_code_chars=12,
            max_error_message_chars=13,
        ),
    )

    assert agent_api.__all__ == [
        "AgentBackend",
        "AgentBackendError",
        "AgentCancelTool",
        "AgentGetTool",
        "AgentRequest",
        "AgentSnapshot",
        "AgentStatus",
        "AgentTool",
        "AgentWaitRequest",
        "AgentWaitTool",
        "agent_completion_message",
        "extract_agent_wait",
        "resume_agent",
    ]
    assert {"AgentTool", "AgentGetTool", "AgentWaitTool", "AgentCancelTool"} <= set(tools.__all__)
    assert isinstance(backend, AgentBackend)
    assert all(isinstance(tool, Tool) for tool in presets)
    assert [tool.spec.name for tool in presets] == [
        "Agent",
        "AgentGet",
        "AgentWait",
        "AgentCancel",
    ]

    agent, get, wait, cancel = presets
    assert agent.spec.execution.concurrency == "serial"
    assert agent.spec.execution.read_only is False
    assert agent.spec.execution.idempotent is False
    assert agent.spec.risk.extra == {"delegates_child_run": True}
    assert agent.spec.risk.filesystem is None
    assert agent.spec.risk.requires_approval is None

    assert get.spec.execution.concurrency == "parallel"
    assert get.spec.execution.read_only is True
    assert get.spec.execution.idempotent is True
    assert get.spec.parallel_safe is True
    assert wait.spec.execution.concurrency == "serial"
    assert wait.spec.execution.read_only is True
    assert wait.spec.execution.idempotent is True
    assert wait.spec.parallel_safe is False
    for read_only in (get, wait):
        assert read_only.spec.risk.filesystem == "none"
        assert read_only.spec.risk.network == "none"
        assert read_only.spec.risk.subprocess is False
        assert read_only.spec.risk.destructive is False
        assert read_only.spec.risk.requires_approval is False

    assert cancel.spec.execution.concurrency == "serial"
    assert cancel.spec.execution.read_only is False
    assert cancel.spec.execution.idempotent is True
    assert cancel.spec.risk.filesystem == "none"
    assert cancel.spec.risk.network == "none"
    assert cancel.spec.risk.subprocess is False
    assert cancel.spec.risk.destructive is True
    assert cancel.spec.risk.requires_approval is False
    assert cancel.spec.risk.extra == {"agent_action": "cancel"}

    assert thaw_json_value(agent.spec.input_schema) == {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["description", "prompt"],
        "properties": {
            "description": {"type": "string", "minLength": 1, "maxLength": 9},
            "prompt": {"type": "string", "minLength": 1, "maxLength": 10},
            "background": {"type": "boolean", "default": False},
        },
        "additionalProperties": False,
    }
    expected_id_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["agent_id"],
        "properties": {"agent_id": {"type": "string", "minLength": 1, "maxLength": 8}},
        "additionalProperties": False,
    }
    assert all(
        thaw_json_value(tool.spec.input_schema) == expected_id_schema
        for tool in (get, wait, cancel)
    )
    outputs = [thaw_json_value(tool.spec.output_schema) for tool in presets]
    assert outputs.count(outputs[0]) == len(outputs)
    output = cast(dict[str, Any], outputs[0])
    branches = cast(list[dict[str, Any]], output["oneOf"])
    assert [branch.get("properties", {}).get("status", {}).get("const") for branch in branches] == [
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
        None,
    ]
    assert branches[-1] == {"type": "null"}
    by_status = {
        cast(str, branch["properties"]["status"]["const"]): branch for branch in branches[:-1]
    }
    assert by_status["queued"]["properties"]["cancellation_requested"] == {"type": "boolean"}
    assert by_status["running"]["properties"]["cancellation_requested"] == {"type": "boolean"}
    assert by_status["completed"]["properties"]["cancellation_requested"] == {"const": False}
    assert by_status["failed"]["properties"]["cancellation_requested"] == {"const": False}
    assert by_status["cancelled"]["properties"]["cancellation_requested"] == {"const": True}
    assert by_status["completed"]["properties"]["result"]["maxLength"] == 11
    assert by_status["failed"]["properties"]["error"]["properties"] == {
        "code": {"type": "string", "minLength": 1, "maxLength": 12},
        "message": {"type": "string", "minLength": 1, "maxLength": 13},
    }

    async def registry_names() -> tuple[str, ...]:
        catalog = await ToolRegistry(presets).open_catalog()
        return tuple(spec.name for spec in catalog.specs())

    assert asyncio.run(registry_names()) == ("Agent", "AgentGet", "AgentWait", "AgentCancel")


def test_agent_constructors_validate_backend_and_all_limits() -> None:
    invalid_backend = cast(Any, object())
    for construct in (AgentTool, AgentGetTool, AgentWaitTool, AgentCancelTool):
        with pytest.raises(TypeError, match="backend must implement AgentBackend"):
            cast(Any, construct)(invalid_backend)

    backend = _FakeBackend()
    common_keywords = (
        "max_agent_id_chars",
        "max_description_chars",
        "max_result_chars",
        "max_error_code_chars",
        "max_error_message_chars",
    )
    for construct in (AgentTool, AgentGetTool, AgentWaitTool, AgentCancelTool):
        for keyword in common_keywords:
            with pytest.raises(ValueError, match=keyword):
                cast(Any, construct)(backend, **{keyword: 0})
    with pytest.raises(ValueError, match="max_prompt_chars"):
        AgentTool(backend, max_prompt_chars=0)


@pytest.mark.parametrize("value", [0, -1, True, "1"])
def test_positive_int_and_schema_builders_reject_invalid_limits(value: object) -> None:
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        agent_schema.positive_int(value, "limit")
    with pytest.raises(ValueError, match="max_description_chars"):
        agent_schema.agent_input_schema(max_description_chars=cast(Any, value))
    with pytest.raises(ValueError, match="max_prompt_chars"):
        agent_schema.agent_input_schema(max_prompt_chars=cast(Any, value))
    with pytest.raises(ValueError, match="max_agent_id_chars"):
        agent_schema.agent_id_input_schema(max_agent_id_chars=cast(Any, value))


@pytest.mark.parametrize(
    "keyword",
    [
        "max_agent_id_chars",
        "max_description_chars",
        "max_result_chars",
        "max_error_code_chars",
        "max_error_message_chars",
    ],
)
def test_snapshot_schema_rejects_each_invalid_limit(keyword: str) -> None:
    with pytest.raises(ValueError, match=keyword):
        agent_schema.snapshot_output_schema(**cast(Any, {keyword: 0}))


def test_agent_request_and_backend_error_values_validate_and_are_immutable() -> None:
    request = AgentRequest("Inspect auth", "Read the auth module.")
    assert request.background is False
    prompt_attribute = "prompt"
    with pytest.raises(FrozenInstanceError):
        setattr(request, prompt_attribute, "changed")

    invalid_requests: tuple[tuple[dict[str, object], type[Exception], str], ...] = (
        ({"description": "", "prompt": "x"}, ValueError, "description"),
        ({"description": 1, "prompt": "x"}, TypeError, "description"),
        ({"description": "x", "prompt": ""}, ValueError, "prompt"),
        ({"description": "x", "prompt": 1}, TypeError, "prompt"),
        ({"description": "x", "prompt": "y", "background": 1}, TypeError, "background"),
    )
    for arguments, error, pattern in invalid_requests:
        with pytest.raises(error, match=pattern):
            AgentRequest(**cast(Any, arguments))

    backend_error = AgentBackendError("agent_not_found", "Agent not found.")
    assert backend_error.code == "agent_not_found"
    assert backend_error.message == "Agent not found."
    assert str(backend_error) == "Agent not found."
    for arguments, error in (
        (("", "message"), ValueError),
        ((1, "message"), TypeError),
        (("code", ""), ValueError),
        (("code", 1), TypeError),
    ):
        with pytest.raises(error):
            AgentBackendError(*cast(Any, arguments))


def test_agent_snapshot_accepts_every_valid_state_and_is_immutable() -> None:
    queued = AgentSnapshot("queued", "Queue", "queued", False)
    running = AgentSnapshot(
        "running",
        "Run",
        "running",
        True,
        cancellation_requested=True,
    )
    completed = _completed(result="")
    failed = _failed()
    cancelled = _cancelled()
    assert [snapshot.status for snapshot in (queued, running, completed, failed, cancelled)] == [
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
    ]
    status_attribute = "status"
    with pytest.raises(FrozenInstanceError):
        setattr(queued, status_attribute, "running")


@pytest.mark.parametrize(
    ("arguments", "error", "pattern"),
    [
        (
            {"agent_id": "", "description": "d", "status": "queued", "background": False},
            ValueError,
            "agent_id",
        ),
        (
            {"agent_id": 1, "description": "d", "status": "queued", "background": False},
            TypeError,
            "agent_id",
        ),
        (
            {"agent_id": "a", "description": "", "status": "queued", "background": False},
            ValueError,
            "description",
        ),
        (
            {"agent_id": "a", "description": 1, "status": "queued", "background": False},
            TypeError,
            "description",
        ),
        (
            {"agent_id": "a", "description": "d", "status": 1, "background": False},
            TypeError,
            "status",
        ),
        (
            {"agent_id": "a", "description": "d", "status": "unknown", "background": False},
            ValueError,
            "unsupported",
        ),
        (
            {"agent_id": "a", "description": "d", "status": "queued", "background": 1},
            TypeError,
            "background",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "queued",
                "background": False,
                "result": 1,
            },
            TypeError,
            "result",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "queued",
                "background": False,
                "error": "bad",
            },
            TypeError,
            "error",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "queued",
                "background": False,
                "cancellation_requested": 1,
            },
            TypeError,
            "cancellation_requested",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "queued",
                "background": False,
                "result": "bad",
            },
            ValueError,
            "cannot include result",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "running",
                "background": False,
                "error": ErrorInfo("bad", "bad"),
            },
            ValueError,
            "cannot include error",
        ),
        (
            {"agent_id": "a", "description": "d", "status": "cancelled", "background": False},
            ValueError,
            "requires cancellation_requested",
        ),
        (
            {"agent_id": "a", "description": "d", "status": "completed", "background": False},
            ValueError,
            "requires result",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "completed",
                "background": False,
                "result": "ok",
                "error": ErrorInfo("bad", "bad"),
            },
            ValueError,
            "cannot include error",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "completed",
                "background": False,
                "result": "ok",
                "cancellation_requested": True,
            },
            ValueError,
            "cannot have cancellation_requested",
        ),
        (
            {"agent_id": "a", "description": "d", "status": "failed", "background": False},
            ValueError,
            "requires error",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "failed",
                "background": False,
                "result": "bad",
                "error": ErrorInfo("bad", "bad"),
            },
            ValueError,
            "cannot include result",
        ),
        (
            {
                "agent_id": "a",
                "description": "d",
                "status": "failed",
                "background": False,
                "error": ErrorInfo("bad", "bad"),
                "cancellation_requested": True,
            },
            ValueError,
            "cannot have cancellation_requested",
        ),
    ],
)
def test_agent_snapshot_rejects_every_invalid_state(
    arguments: dict[str, object],
    error: type[Exception],
    pattern: str,
) -> None:
    with pytest.raises(error, match=pattern):
        AgentSnapshot(**cast(Any, arguments))


def test_wait_id_is_stable_and_length_prefixed() -> None:
    first = agent_schema.build_wait_id("a:b", "c")
    second = agent_schema.build_wait_id("a", "b:c")
    assert first == "agent-wait:3:a:b:1:c"
    assert second == "agent-wait:1:a:3:b:c"
    assert first != second
    for run_id, call_id, pattern in (
        ("", "call", "run_id"),
        (1, "call", "run_id"),
        ("run", "", "tool_call_id"),
        ("run", 1, "tool_call_id"),
    ):
        with pytest.raises(agent_schema.AgentContractError, match=pattern):
            agent_schema.build_wait_id(cast(Any, run_id), cast(Any, call_id))


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {1: "not-a-string-key"},
        {},
        {"description": "d"},
        {"prompt": "p"},
        {"description": "", "prompt": "p"},
        {"description": "toolong", "prompt": "p"},
        {"description": 1, "prompt": "p"},
        {"description": "d", "prompt": ""},
        {"description": "d", "prompt": "toolong"},
        {"description": "d", "prompt": 1},
        {"description": "d", "prompt": "p", "background": 1},
        {"description": "d", "prompt": "p", "unknown": True},
    ],
)
def test_normalize_agent_request_rejects_contract_boundaries(arguments: object) -> None:
    with pytest.raises(agent_schema.AgentContractError):
        agent_schema.normalize_agent_request(
            arguments,
            max_description_chars=3,
            max_prompt_chars=3,
        )


def test_normalize_agent_request_defaults_background_and_validates_its_limits() -> None:
    assert agent_schema.normalize_agent_request(
        {"description": "abc", "prompt": "xyz"},
        max_description_chars=3,
        max_prompt_chars=3,
    ) == AgentRequest("abc", "xyz", False)
    assert agent_schema.normalize_agent_request(
        {"description": "abc", "prompt": "xyz", "background": True},
        max_description_chars=3,
        max_prompt_chars=3,
    ) == AgentRequest("abc", "xyz", True)
    with pytest.raises(agent_schema.AgentContractError, match="max_description_chars"):
        agent_schema.normalize_agent_request({}, max_description_chars=0)
    with pytest.raises(agent_schema.AgentContractError, match="max_prompt_chars"):
        agent_schema.normalize_agent_request({}, max_prompt_chars=0)


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        {1: "bad-key"},
        {},
        {"agent_id": ""},
        {"agent_id": "toolong"},
        {"agent_id": 1},
        {"agent_id": "ok", "unknown": True},
    ],
)
def test_normalize_agent_id_rejects_contract_boundaries(arguments: object) -> None:
    with pytest.raises(agent_schema.AgentContractError):
        agent_schema.normalize_agent_id(arguments, max_agent_id_chars=3)


def test_normalize_agent_id_accepts_limit_and_validates_config() -> None:
    assert agent_schema.normalize_agent_id({"agent_id": "abc"}, max_agent_id_chars=3) == "abc"
    with pytest.raises(agent_schema.AgentContractError, match="max_agent_id_chars"):
        agent_schema.normalize_agent_id({}, max_agent_id_chars=0)


@pytest.mark.parametrize(
    "snapshot",
    [
        AgentSnapshot("queued", "Queued", "queued", False),
        AgentSnapshot("running", "Running", "running", True, cancellation_requested=True),
        _completed(),
        _failed(),
        _cancelled(),
    ],
)
def test_snapshot_payload_is_canonical_for_every_status(snapshot: AgentSnapshot) -> None:
    payload = agent_schema.snapshot_payload(snapshot)
    assert payload["agent_id"] == snapshot.agent_id
    assert payload["description"] == snapshot.description
    assert payload["status"] == snapshot.status
    assert payload["background"] is snapshot.background
    assert payload["cancellation_requested"] is snapshot.cancellation_requested
    if snapshot.status == "completed":
        assert payload["result"] == snapshot.result
    elif snapshot.status == "failed":
        assert payload["error"] == {
            "code": "child_failed",
            "message": "Child run failed safely.",
        }
    else:
        assert "result" not in payload
        assert "error" not in payload


def test_snapshot_payload_rejects_wrong_type_inconsistent_state_and_each_bound() -> None:
    with pytest.raises(agent_schema.AgentContractError, match="AgentSnapshot"):
        agent_schema.snapshot_payload(object())
    with pytest.raises(agent_schema.AgentContractError, match="inconsistent"):
        agent_schema.snapshot_payload(_forge_snapshot(description=""))

    cases: tuple[tuple[AgentSnapshot, dict[str, int], str], ...] = (
        (AgentSnapshot("abcd", "d", "running", True), {"max_agent_id_chars": 3}, "agent_id"),
        (AgentSnapshot("a", "abcd", "running", True), {"max_description_chars": 3}, "description"),
        (
            _completed(agent_id="a", description="d", result="abcd"),
            {"max_result_chars": 3},
            "result",
        ),
        (
            AgentSnapshot(
                "a",
                "d",
                "failed",
                True,
                error=ErrorInfo("abcd", "message"),
            ),
            {"max_error_code_chars": 3},
            "error.code",
        ),
        (
            AgentSnapshot(
                "a",
                "d",
                "failed",
                True,
                error=ErrorInfo("code", "abcd"),
            ),
            {"max_error_message_chars": 3},
            "error.message",
        ),
    )
    for snapshot, limits, pattern in cases:
        with pytest.raises(agent_schema.AgentContractError, match=pattern):
            agent_schema.snapshot_payload(snapshot, **cast(Any, limits))

    for keyword in (
        "max_agent_id_chars",
        "max_description_chars",
        "max_result_chars",
        "max_error_code_chars",
        "max_error_message_chars",
    ):
        with pytest.raises(agent_schema.AgentContractError, match=keyword):
            agent_schema.snapshot_payload(_completed(), **cast(Any, {keyword: 0}))


def test_agent_foreground_waits_with_exact_durable_contract() -> None:
    backend = _FakeBackend()
    backend.force("start", AgentSnapshot("agent-fg", "Inspect", "queued", False))
    tool = AgentTool(backend)
    result = _invoke(
        tool,
        {"description": "Inspect", "prompt": "Inspect auth."},
        call_id="foreground-call",
        through_registry=True,
    )

    assert isinstance(result, WaitingResult)
    outcome = result.outcome
    assert isinstance(outcome, ToolWaiting)
    assert outcome.parts[0].text == "Waiting for agent agent-fg to complete."
    assert outcome.task == TaskRef(
        "agent-fg",
        "queued",
        {"background": False, "kind": "agent"},
    )
    assert thaw_json_value(outcome.structured_content) == {
        "agent_id": "agent-fg",
        "status": "queued",
        "description": "Inspect",
        "background": False,
        "cancellation_requested": False,
    }
    assert result.suspension.reason == "agent_completion"
    assert result.suspension.source == "Agent"
    assert result.suspension.wait_id == "agent-wait:10:parent-run:15:foreground-call"
    assert result.suspension.metadata == {
        "agent_id": "agent-fg",
        "schema_version": 1,
        "tool_call_id": "foreground-call",
        "max_agent_id_chars": 512,
        "max_description_chars": 200,
        "max_result_chars": 65_536,
        "max_error_code_chars": 128,
        "max_error_message_chars": 4_096,
    }
    assert backend.start_calls == [
        (
            AgentRequest("Inspect", "Inspect auth.", False),
            backend.start_calls[0][1],
            "foreground-call",
        )
    ]
    assert backend.start_calls[0][1].run_id == "parent-run"


def test_agent_background_returns_accepted_and_start_is_idempotent() -> None:
    backend = _FakeBackend()
    backend.force("start", AgentSnapshot("agent-bg", "Inspect", "running", True))
    tool = AgentTool(backend)
    arguments = {"description": "Inspect", "prompt": "Inspect auth.", "background": True}

    first = _invoke(tool, arguments, call_id="background-call", through_registry=True)
    second = _invoke(AgentTool(backend), arguments, call_id="background-call")
    for result in (first, second):
        assert isinstance(result, SettledResult)
        outcome = result.outcome
        assert isinstance(outcome, ToolAccepted)
        assert outcome.correlation_id == "agent-bg"
        assert outcome.parts[0].text == (
            "Agent agent-bg accepted for background execution (running)."
        )
        assert outcome.task == TaskRef(
            "agent-bg",
            "running",
            {"background": True, "kind": "agent"},
        )
        assert thaw_json_value(outcome.structured_content) == {
            "agent_id": "agent-bg",
            "status": "running",
            "description": "Inspect",
            "background": True,
            "cancellation_requested": False,
        }
    assert backend.created_count == 1
    assert len(backend.start_calls) == 2


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        (_completed(), "Agent agent-1 completed.\n\nNo issues found."),
        (_completed(result=""), "Agent agent-1 completed with an empty result."),
        (_failed(), "Agent agent-1 failed (child_failed): Child run failed safely."),
        (_cancelled(), "Agent agent-1 was cancelled."),
    ],
)
def test_agent_terminal_start_is_a_success_fast_path(
    snapshot: AgentSnapshot,
    expected: str,
) -> None:
    backend = _FakeBackend()
    backend.force("start", snapshot)
    text, payload = _success(
        _invoke(
            AgentTool(backend),
            {
                "description": snapshot.description,
                "prompt": "Inspect auth.",
                "background": snapshot.background,
            },
            through_registry=True,
        )
    )
    assert text == expected
    assert payload == agent_schema.snapshot_payload(snapshot)


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"description": "d"},
        {"prompt": "p"},
        {"description": "", "prompt": "p"},
        {"description": "long", "prompt": "p"},
        {"description": 1, "prompt": "p"},
        {"description": "d", "prompt": ""},
        {"description": "d", "prompt": "long"},
        {"description": "d", "prompt": 1},
        {"description": "d", "prompt": "p", "background": 1},
        {"description": "d", "prompt": "p", "extra": True},
    ],
)
def test_agent_direct_invocation_returns_stable_request_failures(
    arguments: Mapping[str, Any],
) -> None:
    backend = _FakeBackend()
    _failure(
        _invoke(AgentTool(backend, max_description_chars=3, max_prompt_chars=3), arguments),
        "invalid_agent_request",
    )
    assert backend.start_calls == []


def test_agent_rejects_wrong_background_invalid_snapshot_and_backend_error() -> None:
    wrong_mode = _FakeBackend()
    wrong_mode.force("start", AgentSnapshot("agent-1", "Inspect", "running", False))
    _failure(
        _invoke(
            AgentTool(wrong_mode),
            {"description": "Inspect", "prompt": "Do it", "background": True},
        ),
        "invalid_agent_snapshot",
    )

    wrong_description = _FakeBackend()
    wrong_description.force(
        "start",
        AgentSnapshot("agent-1", "Different", "running", True),
    )
    outcome = _failure(
        _invoke(
            AgentTool(wrong_description),
            {"description": "Inspect", "prompt": "Do it", "background": True},
        ),
        "invalid_agent_snapshot",
    )
    assert "wrong description" in outcome.error.message

    oversized = _FakeBackend()
    oversized.force("start", AgentSnapshot("agent-too-long", "ok", "running", True))
    _failure(
        _invoke(
            AgentTool(oversized, max_agent_id_chars=3),
            {"description": "ok", "prompt": "Do it", "background": True},
        ),
        "invalid_agent_snapshot",
    )

    failed_backend = _FakeBackend()
    failed_backend.fail("start")
    outcome = _failure(
        _invoke(
            AgentTool(failed_backend),
            {"description": "Inspect", "prompt": "Do it"},
        ),
        "agent_store_unavailable",
    )
    assert outcome.error.message == "start failed safely"


@pytest.mark.parametrize(
    ("snapshot", "expected_text"),
    [
        (AgentSnapshot("agent-1", "Inspect", "queued", True), "Agent agent-1 is queued."),
        (
            AgentSnapshot(
                "agent-1",
                "Inspect",
                "running",
                True,
                cancellation_requested=True,
            ),
            "Agent agent-1 is running; cancellation requested.",
        ),
        (_completed(), "Agent agent-1 completed.\n\nNo issues found."),
        (_failed(), "Agent agent-1 failed (child_failed): Child run failed safely."),
        (_cancelled(), "Agent agent-1 was cancelled."),
    ],
)
def test_agent_get_returns_every_snapshot_status(
    snapshot: AgentSnapshot,
    expected_text: str,
) -> None:
    backend = _FakeBackend()
    backend.seed(snapshot)
    text, payload = _success(
        _invoke(
            AgentGetTool(backend),
            {"agent_id": snapshot.agent_id},
            call_id="get-call",
            through_registry=True,
        )
    )
    assert text == expected_text
    assert payload == agent_schema.snapshot_payload(snapshot)
    assert backend.get_calls[0][0] == snapshot.agent_id
    assert backend.get_calls[0][1].run_id == "parent-run"


@pytest.mark.parametrize(
    "snapshot",
    [
        AgentSnapshot("agent-1", "Inspect", "queued", True),
        AgentSnapshot(
            "agent-1",
            "Inspect",
            "running",
            True,
            cancellation_requested=True,
        ),
    ],
)
def test_agent_wait_registers_atomically_and_returns_waiting(snapshot: AgentSnapshot) -> None:
    backend = _FakeBackend()
    backend.seed(snapshot)
    result = _invoke(
        AgentWaitTool(backend),
        {"agent_id": snapshot.agent_id},
        call_id="wait-call",
        through_registry=True,
    )
    assert isinstance(result, WaitingResult)
    assert result.outcome.task == TaskRef(
        snapshot.agent_id,
        snapshot.status,
        {"background": True, "kind": "agent"},
    )
    assert thaw_json_value(result.outcome.structured_content) == agent_schema.snapshot_payload(
        snapshot
    )
    assert result.suspension.reason == "agent_completion"
    assert result.suspension.source == "AgentWait"
    assert result.suspension.wait_id == "agent-wait:10:parent-run:9:wait-call"
    assert backend.wait_calls == [(snapshot.agent_id, backend.wait_calls[0][1], "wait-call")]
    assert backend.wait_calls[0][1].run_id == "parent-run"


@pytest.mark.parametrize("snapshot", [_completed(), _failed(), _cancelled()])
def test_agent_wait_terminal_snapshot_is_an_immediate_success(snapshot: AgentSnapshot) -> None:
    backend = _FakeBackend()
    backend.seed(snapshot)
    _, payload = _success(
        _invoke(
            AgentWaitTool(backend),
            {"agent_id": snapshot.agent_id},
            call_id="terminal-wait",
            through_registry=True,
        )
    )
    assert payload == agent_schema.snapshot_payload(snapshot)
    assert backend.wait_calls[0][2] == "terminal-wait"


def test_agent_cancel_acknowledges_active_state_and_preserves_terminal_state() -> None:
    backend = _FakeBackend()
    running = AgentSnapshot("agent-running", "Run", "running", True)
    backend.seed(running)
    text, payload = _success(
        _invoke(
            AgentCancelTool(backend),
            {"agent_id": running.agent_id},
            call_id="cancel-running",
            through_registry=True,
        )
    )
    assert text == "Agent agent-running is running; cancellation requested."
    assert payload["status"] == "running"
    assert payload["cancellation_requested"] is True
    assert backend.cancel_calls[0][2] == "cancel-running"

    for snapshot in (_completed("completed"), _failed("failed"), _cancelled("cancelled")):
        backend.seed(snapshot)
        _, terminal_payload = _success(
            _invoke(
                AgentCancelTool(backend),
                {"agent_id": snapshot.agent_id},
                call_id=f"cancel-{snapshot.agent_id}",
            )
        )
        assert terminal_payload == agent_schema.snapshot_payload(snapshot)
        assert backend.snapshots[snapshot.agent_id] is snapshot


def test_agent_cancel_rejects_unacknowledged_nonterminal_snapshot() -> None:
    backend = _FakeBackend()
    snapshot = AgentSnapshot("agent-1", "Inspect", "running", True)
    backend.seed(snapshot)
    backend.force("cancel", snapshot)
    outcome = _failure(
        _invoke(AgentCancelTool(backend), {"agent_id": snapshot.agent_id}),
        "invalid_agent_snapshot",
    )
    assert "acknowledge" in outcome.error.message


@pytest.mark.parametrize("operation", ["get", "wait", "cancel"])
def test_agent_id_tools_reject_wrong_snapshot_identity(operation: str) -> None:
    backend = _FakeBackend()
    backend.seed(AgentSnapshot("agent-1", "Inspect", "running", True))
    backend.force(operation, AgentSnapshot("agent-other", "Inspect", "running", True))
    selected: Tool
    if operation == "get":
        selected = AgentGetTool(backend)
    elif operation == "wait":
        selected = AgentWaitTool(backend)
    else:
        selected = AgentCancelTool(backend)
    _failure(_invoke(selected, {"agent_id": "agent-1"}), "invalid_agent_snapshot")


def test_agent_id_tool_rejects_backend_snapshot_that_fails_contract_validation() -> None:
    backend = _FakeBackend()
    backend.seed(AgentSnapshot("agent-1", "Inspect", "running", True))
    backend.force("get", _forge_snapshot(description=""))
    _failure(
        _invoke(AgentGetTool(backend), {"agent_id": "agent-1"}),
        "invalid_agent_snapshot",
    )


@pytest.mark.parametrize("operation", ["get", "wait", "cancel"])
def test_agent_id_tools_reject_internally_inconsistent_snapshots(operation: str) -> None:
    backend = _FakeBackend()
    backend.seed(AgentSnapshot("agent-1", "Inspect", "running", True))
    backend.force(
        operation,
        _forge_snapshot(
            description="",
            cancellation_requested=operation == "cancel",
        ),
    )
    selected: Tool
    if operation == "get":
        selected = AgentGetTool(backend)
    elif operation == "wait":
        selected = AgentWaitTool(backend)
    else:
        selected = AgentCancelTool(backend)
    _failure(_invoke(selected, {"agent_id": "agent-1"}), "invalid_agent_snapshot")


@pytest.mark.parametrize("operation", ["start", "get", "wait", "cancel"])
def test_agent_tools_reject_backend_values_that_are_not_snapshots(operation: str) -> None:
    backend = _FakeBackend()
    if operation != "start":
        backend.seed(AgentSnapshot("agent-1", "Inspect", "running", True))
    backend.force(operation, cast(AgentSnapshot, object()))
    if operation == "start":
        selected: Tool = AgentTool(backend)
        arguments: Mapping[str, Any] = {
            "description": "Inspect",
            "prompt": "Inspect auth.",
            "background": True,
        }
    elif operation == "get":
        selected = AgentGetTool(backend)
        arguments = {"agent_id": "agent-1"}
    elif operation == "wait":
        selected = AgentWaitTool(backend)
        arguments = {"agent_id": "agent-1"}
    else:
        selected = AgentCancelTool(backend)
        arguments = {"agent_id": "agent-1"}
    outcome = _failure(_invoke(selected, arguments), "invalid_agent_snapshot")
    assert outcome.error.message == "Agent backend must return an AgentSnapshot."


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"agent_id": ""},
        {"agent_id": "long"},
        {"agent_id": 1},
        {"agent_id": "ok", "extra": True},
    ],
)
def test_agent_id_tools_return_stable_direct_validation_failures(
    arguments: Mapping[str, Any],
) -> None:
    backend = _FakeBackend()
    for tool in (
        AgentGetTool(backend, max_agent_id_chars=3),
        AgentWaitTool(backend, max_agent_id_chars=3),
        AgentCancelTool(backend, max_agent_id_chars=3),
    ):
        _failure(_invoke(tool, arguments), "invalid_agent_id")
    assert backend.get_calls == []
    assert backend.wait_calls == []
    assert backend.cancel_calls == []


def test_registry_rejects_invalid_arguments_before_any_backend_call() -> None:
    backend = _FakeBackend()
    presets: tuple[Tool, ...] = (
        AgentTool(backend, max_description_chars=3, max_prompt_chars=3),
        AgentGetTool(backend, max_agent_id_chars=3),
        AgentWaitTool(backend, max_agent_id_chars=3),
        AgentCancelTool(backend, max_agent_id_chars=3),
    )

    async def validate() -> None:
        catalog = await ToolRegistry(presets).open_catalog()
        invalid_agent: tuple[Mapping[str, Any], ...] = (
            {},
            {"description": "", "prompt": "p"},
            {"description": "long", "prompt": "p"},
            {"description": "d", "prompt": "long"},
            {"description": "d", "prompt": "p", "background": 1},
            {"description": "d", "prompt": "p", "extra": True},
        )
        for index, arguments in enumerate(invalid_agent):
            with pytest.raises(ToolError, match="do not match input_schema"):
                catalog.bind(ToolCall(f"agent-{index}", "Agent", arguments))
        invalid_id: tuple[Mapping[str, Any], ...] = (
            {},
            {"agent_id": ""},
            {"agent_id": "long"},
            {"agent_id": 1},
            {"agent_id": "ok", "extra": True},
        )
        for name in ("AgentGet", "AgentWait", "AgentCancel"):
            for index, arguments in enumerate(invalid_id):
                with pytest.raises(ToolError, match="do not match input_schema"):
                    catalog.bind(ToolCall(f"{name}-{index}", name, arguments))

    asyncio.run(validate())
    assert backend.start_calls == []
    assert backend.get_calls == []
    assert backend.wait_calls == []
    assert backend.cancel_calls == []


def test_cancel_requested_precedes_validation_and_never_calls_backend() -> None:
    backend = _FakeBackend()
    calls: tuple[tuple[Tool, str], ...] = (
        (AgentTool(backend), "Agent"),
        (AgentGetTool(backend), "AgentGet"),
        (AgentWaitTool(backend), "AgentWait"),
        (AgentCancelTool(backend), "AgentCancel"),
    )
    for tool, name in calls:
        outcome = _failure(_invoke(tool, {}, cancelled=True), "cancelled")
        assert name in outcome.error.message
    assert backend.start_calls == []
    assert backend.get_calls == []
    assert backend.wait_calls == []
    assert backend.cancel_calls == []


@pytest.mark.parametrize("operation", ["get", "wait", "cancel"])
def test_agent_backend_errors_are_mapped_and_null_output_passes_registry(operation: str) -> None:
    backend = _FakeBackend()
    backend.fail(operation)
    selected: Tool
    if operation == "get":
        selected = AgentGetTool(backend)
    elif operation == "wait":
        selected = AgentWaitTool(backend)
    else:
        selected = AgentCancelTool(backend)
    outcome = _failure(
        _invoke(
            selected,
            {"agent_id": "agent-missing"},
            through_registry=True,
        ),
        "agent_store_unavailable",
    )
    assert outcome.error.message == f"{operation} failed safely"


@pytest.mark.parametrize("operation", ["start", "get", "wait", "cancel"])
def test_unexpected_backend_exceptions_are_normalized_without_leaking_details(
    operation: str,
) -> None:
    backend = _FakeBackend()
    backend.crash(operation)
    if operation == "start":
        selected: Tool = AgentTool(backend)
        arguments: Mapping[str, Any] = {
            "description": "Inspect",
            "prompt": "Inspect auth.",
        }
    elif operation == "get":
        selected = AgentGetTool(backend)
        arguments = {"agent_id": "agent-1"}
    elif operation == "wait":
        selected = AgentWaitTool(backend)
        arguments = {"agent_id": "agent-1"}
    else:
        selected = AgentCancelTool(backend)
        arguments = {"agent_id": "agent-1"}
    outcome = _failure(_invoke(selected, arguments), "agent_backend_error")
    assert outcome.error.message == ("The Host Agent backend failed while processing the request.")
    assert "sensitive" not in outcome.error.message


def test_tool_instances_share_host_backend_state_ownership_and_idempotency() -> None:
    backend = _FakeBackend()
    arguments = {
        "description": "Inspect",
        "prompt": "Inspect auth.",
        "background": True,
    }
    accepted = _invoke(AgentTool(backend), arguments, call_id="shared-start")
    assert isinstance(accepted, SettledResult)
    accepted_outcome = accepted.outcome
    assert isinstance(accepted_outcome, ToolAccepted)
    agent_id = accepted_outcome.correlation_id

    _, get_payload = _success(_invoke(AgentGetTool(backend), {"agent_id": agent_id}))
    assert get_payload["status"] == "queued"
    waited = _invoke(
        AgentWaitTool(backend),
        {"agent_id": agent_id},
        call_id="shared-wait",
    )
    assert isinstance(waited, WaitingResult)
    _, cancel_payload = _success(
        _invoke(
            AgentCancelTool(backend),
            {"agent_id": agent_id},
            call_id="shared-cancel",
        )
    )
    assert cancel_payload["cancellation_requested"] is True

    retried = _invoke(AgentTool(backend), arguments, call_id="shared-start")
    assert isinstance(retried, SettledResult)
    assert isinstance(retried.outcome, ToolAccepted)
    assert retried.outcome.correlation_id == agent_id
    assert backend.created_count == 1

    conflict = _failure(
        _invoke(
            AgentTool(backend),
            {**arguments, "prompt": "Different task."},
            call_id="shared-start",
        ),
        "agent_conflict",
    )
    assert "different Agent request" in conflict.error.message

    unauthorized = _failure(
        _invoke(
            AgentGetTool(backend),
            {"agent_id": agent_id},
            run_id="other-parent",
        ),
        "agent_not_found",
    )
    assert unauthorized.error.message == "Agent not found."


def test_fake_backend_host_transition_is_visible_without_mutating_old_accepted_result() -> None:
    backend = _FakeBackend()
    accepted = _invoke(
        AgentTool(backend),
        {"description": "Inspect", "prompt": "Do it", "background": True},
    )
    assert isinstance(accepted, SettledResult)
    outcome = accepted.outcome
    assert isinstance(outcome, ToolAccepted)
    assert outcome.task is not None
    assert outcome.task.status == "queued"

    backend.transition(_completed(outcome.correlation_id, description="Inspect"))
    _, payload = _success(_invoke(AgentGetTool(backend), {"agent_id": outcome.correlation_id}))
    assert payload["status"] == "completed"
    assert outcome.task.status == "queued"
    with pytest.raises(KeyError):
        backend.transition(_completed("missing"))


def test_private_snapshot_text_defensively_rejects_failed_snapshot_without_error() -> None:
    broken = _forge_snapshot(status="failed", error=None)
    with pytest.raises(agent_schema.AgentContractError, match="has no error"):
        agent_tools_module._snapshot_text(broken)


def test_private_bounded_string_defensively_rejects_non_text() -> None:
    with pytest.raises(agent_schema.AgentContractError, match="value must be a string"):
        agent_schema._bounded_string(1, "value", 3)
