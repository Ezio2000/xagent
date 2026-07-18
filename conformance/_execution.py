"""Execute one fixture invocation through Runtime and its single Invocation."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from conformance._model import CaseModel
from conformance._schemas import SchemaSuite
from conformance._tools import fixture_batch_policy
from conformance._values import integer, mapping, number, sequence, string
from jharness.kernel import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalSuspend,
    Checkpoint,
    CommitError,
    Event,
    HistoryRewrite,
    Invocation,
    ModelRequest,
    RepositoryError,
    RequestError,
    RevisionConflict,
    RunLimits,
    RunSnapshot,
    Runtime,
    SuspensionSelector,
    thaw_json_value,
)
from jharness.kernel.diagnostics import RequestKind, RunTrace, build_trace, verify_trace
from jharness.kernel.wire import (
    decode_checkpoint,
    decode_context,
    decode_event,
    decode_message,
    decode_suspension,
    decode_trace,
    encode_checkpoint,
    encode_event,
    encode_message,
    encode_tool_spec,
    encode_trace,
)
from jharness.toolkit import ToolRegistry


@dataclass(frozen=True, slots=True)
class InvocationOutcome:
    checkpoint: Checkpoint
    events: tuple[Event, ...]
    commits: tuple[Checkpoint, ...]
    model: CaseModel | None
    trace: RunTrace | None
    repository_error: str | None = None
    request_error: str | None = None
    repository_idempotent: bool | None = None

    @property
    def snapshot(self) -> RunSnapshot:
        return self.checkpoint.snapshot


class CaseRepository:
    """Fixture repository with CAS, idempotency, delay, and failure controls."""

    __slots__ = ("_by_id", "_delay", "_failure", "_head", "commits")

    def __init__(
        self,
        initial: Checkpoint | None,
        failure: Mapping[str, Any] | None,
        delay: Mapping[str, Any] | None,
    ) -> None:
        self._head = initial
        self._failure = failure
        self._delay = delay
        self._by_id = {} if initial is None else {initial.id: initial}
        self.commits: list[Checkpoint] = []

    @property
    def checkpoint(self) -> Checkpoint:
        if self._head is None:
            raise RuntimeError("invocation has no durable checkpoint")
        return self._head

    async def commit(self, checkpoint: Checkpoint) -> None:
        existing = self._by_id.get(checkpoint.id)
        if existing is not None:
            if existing == checkpoint:
                return
            raise RepositoryError(f"checkpoint id {checkpoint.id!r} was reused with new content")

        head = self._head
        run_id = checkpoint.snapshot.context.run_id
        expected = checkpoint.snapshot.revision - 1 if checkpoint.snapshot.revision else None
        if head is None:
            actual = None
        elif head.snapshot.context.run_id != run_id:
            raise RepositoryError("fixture repository cannot contain more than one run")
        else:
            actual = head.snapshot.revision
        if actual != expected:
            raise RevisionConflict(run_id, expected, actual)

        if self._delay is not None and checkpoint.fact.kind == string(
            self._delay["fact_kind"], "repository delay fact_kind"
        ):
            await asyncio.sleep(number(self._delay["delay_seconds"], "repository delay_seconds"))
        if self._failure is not None and checkpoint.snapshot.revision == integer(
            self._failure["revision"], "repository failure revision"
        ):
            raise RuntimeError(string(self._failure["message"], "repository failure message"))

        self._head = checkpoint
        self._by_id[checkpoint.id] = checkpoint
        self.commits.append(checkpoint)

    async def verify_idempotency(self) -> bool:
        original = tuple(self.commits)
        for checkpoint in original:
            await self.commit(checkpoint)
        if original:
            checkpoint = original[-1]
            changed_fact = replace(checkpoint.fact, at=checkpoint.fact.at + 1)
            collision = replace(checkpoint, fact=changed_fact)
            try:
                await self.commit(collision)
            except RepositoryError:
                pass
            else:
                return False
            stale = replace(checkpoint, id=f"{checkpoint.id}-stale")
            try:
                await self.commit(stale)
            except RevisionConflict:
                pass
            else:
                return False
        return self.commits == list(original)


async def run_invocation(
    fixture: Mapping[str, Any],
    *,
    seed: Checkpoint | None,
    previous: Checkpoint | None,
    tools: ToolRegistry,
    schemas: SchemaSuite,
) -> InvocationOutcome:
    expected = mapping(fixture["expected"], "invocation expected")
    request = mapping(fixture["request"], "fixture request")
    request_kind = cast(RequestKind, string(request["kind"], "request kind"))
    source = _source_for_request(request, seed, previous)
    raw_steps = tuple(sequence(fixture["model_steps"], "model steps"))
    model = CaseModel(raw_steps)
    streaming = model.streaming
    repository = CaseRepository(
        source,
        _optional_mapping(fixture.get("repository_failure"), "repository failure"),
        _optional_mapping(fixture.get("repository_delay"), "repository delay"),
    )
    runtime = Runtime(
        model=model,
        tools=tools,
        limits=_limits(fixture.get("limits")),
        approval_policy=_approval_policy(
            _optional_mapping(fixture.get("approval_decisions"), "approval decisions") or {},
            number(fixture.get("approval_delay_seconds", 0), "approval delay_seconds"),
        ),
        history_reducer=_history_reducer(fixture.get("history_rewrite")),
        batch_policy=fixture_batch_policy(fixture.get("batch_policy")),
        repository=repository,
    )
    try:
        invocation = _invoke(runtime, request, source, stream=streaming)
    except Exception as exc:
        expected_error = expected.get("request_error")
        if expected_error is None:
            raise
        error_code = _request_error_code(exc)
        if error_code != expected_error:
            raise AssertionError(
                f"expected request error {expected_error!r}, got {error_code!r}: {exc}"
            ) from exc
        if source is None:
            raise AssertionError("request error case has no source checkpoint") from exc
        return InvocationOutcome(
            source,
            (),
            (),
            None,
            None,
            request_error=error_code,
        )

    events: list[Event] = []
    occurrences: Counter[str] = Counter()
    actions = _index_actions(fixture.get("actions", ()))
    repository_error: str | None = None
    checkpoint: Checkpoint
    try:
        async for event in invocation.events():
            events.append(event)
            occurrences[event.kind.value] += 1
            _apply_actions(
                invocation,
                event.kind.value,
                occurrences[event.kind.value],
                actions,
            )
        checkpoint = await invocation.result()
    except CommitError as exc:
        repository_error = str(exc)
        checkpoint = _last_checkpoint(exc, repository)

    model.assert_consumed()
    if checkpoint != repository.checkpoint:
        raise AssertionError("invocation result differs from the durable repository head")
    repository_idempotent = (
        await repository.verify_idempotency() if "repository_idempotent" in expected else None
    )
    trace = build_trace(
        events,
        request_kind,
        metadata_keys=tuple(sorted(checkpoint.snapshot.context.metadata)),
    )
    _validate_wire(schemas, checkpoint, events, repository.commits, model, trace)
    return InvocationOutcome(
        checkpoint,
        tuple(events),
        tuple(repository.commits),
        model,
        trace,
        repository_error=repository_error,
        repository_idempotent=repository_idempotent,
    )


def _source_for_request(
    request: Mapping[str, Any],
    seed: Checkpoint | None,
    previous: Checkpoint | None,
) -> Checkpoint | None:
    if request["kind"] == "start":
        return None
    source = string(request["source"], "request source")
    checkpoint = seed if source == "seed" else previous if source == "previous" else None
    if checkpoint is None:
        raise ValueError(f"request source {source!r} is unavailable")
    return checkpoint


def _invoke(
    runtime: Runtime,
    request: Mapping[str, Any],
    source: Checkpoint | None,
    *,
    stream: bool,
) -> Invocation:
    kind = string(request["kind"], "request kind")
    if kind == "start":
        messages = tuple(
            decode_message(message) for message in sequence(request["messages"], "start messages")
        )
        raw_context = request.get("context")
        context = None if raw_context is None else decode_context(raw_context)
        return runtime.start(messages, context=context, stream=stream)
    if source is None:
        raise ValueError(f"{kind} request requires a source checkpoint")
    if kind == "continue":
        return runtime.continue_from(source, stream=stream)
    if kind == "resume":
        return runtime.resume(
            source,
            selector=_selector(request.get("selector")),
            append_messages=tuple(
                decode_message(message)
                for message in sequence(request.get("append_messages", ()), "append_messages")
            ),
            metadata=mapping(request.get("metadata", {}), "resume metadata"),
            stream=stream,
        )
    raise ValueError(f"unsupported fixture request kind: {kind}")


def _selector(value: object) -> SuspensionSelector | None:
    if value is None:
        return None
    raw = mapping(value, "suspension selector")
    return SuspensionSelector(
        reason=_optional_string(raw.get("reason"), "selector reason"),
        source=_optional_string(raw.get("source"), "selector source"),
        wait_id=_optional_string(raw.get("wait_id"), "selector wait_id"),
        metadata=mapping(raw.get("metadata", {}), "selector metadata"),
    )


class _FixtureApprovalPolicy:
    __slots__ = ("_decisions", "_delay")

    def __init__(self, decisions: Mapping[str, ApprovalDecision], delay: float) -> None:
        self._decisions = dict(decisions)
        self._delay = delay

    async def decide(
        self,
        requests: tuple[ApprovalRequest, ...],
    ) -> tuple[ApprovalDecision, ...]:
        if self._delay:
            await asyncio.sleep(self._delay)
        return tuple(
            self._decisions[request.call.id]
            for request in requests
            if request.call.id in self._decisions
        )


def _approval_policy(
    decisions: Mapping[str, Any],
    delay: float,
) -> ApprovalPolicy | None:
    if not decisions:
        return None
    parsed = {
        call_id: _approval_decision(mapping(value, f"approval decision {call_id}"))
        for call_id, value in decisions.items()
    }
    return _FixtureApprovalPolicy(parsed, delay)


def _approval_decision(value: Mapping[str, Any]) -> ApprovalDecision:
    call_id = string(value["call_id"], "approval call_id")
    kind = string(value["kind"], "approval kind")
    if kind == "allow":
        return ApprovalAllow(call_id)
    if kind == "deny":
        return ApprovalDeny(call_id, string(value["reason"], "approval denial reason"))
    if kind == "suspend":
        return ApprovalSuspend(call_id, decode_suspension(value["suspension"]))
    raise ValueError(f"unsupported approval decision: {kind!r}")


class _FixtureHistoryReducer:
    __slots__ = ("_fixture", "_used")

    def __init__(self, fixture: Mapping[str, Any]) -> None:
        self._fixture = fixture
        self._used = False

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None:
        if self._used or len(snapshot.history) != integer(
            self._fixture["trigger_history_count"], "history trigger"
        ):
            return None
        self._used = True
        delay = number(self._fixture.get("delay_seconds", 0), "history rewrite delay_seconds")
        if delay:
            await asyncio.sleep(delay)
        return HistoryRewrite(
            tuple(
                decode_message(message)
                for message in sequence(self._fixture["messages"], "rewrite messages")
            ),
            string(self._fixture["reason"], "rewrite reason"),
            mapping(self._fixture["metadata"], "rewrite metadata"),
        )


def _history_reducer(value: object) -> _FixtureHistoryReducer | None:
    return None if value is None else _FixtureHistoryReducer(mapping(value, "history rewrite"))


def _index_actions(
    value: object,
) -> dict[tuple[str, int], tuple[Mapping[str, Any], ...]]:
    indexed: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for raw_action in sequence(value, "actions"):
        action = mapping(raw_action, "action")
        trigger = mapping(action["when"], "action trigger")
        key = (
            string(trigger["event_kind"], "action event_kind"),
            integer(trigger["occurrence"], "action occurrence"),
        )
        indexed.setdefault(key, []).append(mapping(action["command"], "control command"))
    return {key: tuple(commands) for key, commands in indexed.items()}


def _apply_actions(
    invocation: Invocation,
    event_kind: str,
    occurrence: int,
    actions: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> None:
    for command in actions.get((event_kind, occurrence), ()):
        _apply_command(invocation, command)


def _apply_command(invocation: Invocation, command: Mapping[str, Any]) -> None:
    kind = string(command["kind"], "control command kind")
    if kind == "pause":
        invocation.pause(decode_suspension(command["suspension"]))
    elif kind == "insert":
        invocation.insert(decode_message(command["message"]))
    elif kind == "cancel_tool":
        invocation.cancel_tool(string(command["tool_call_id"], "cancel tool_call_id"))
    else:
        raise ValueError(f"unsupported control command: {kind}")


def _limits(value: object) -> RunLimits | None:
    if value is None:
        return None
    raw = mapping(value, "limits")
    defaults = RunLimits()
    max_total = raw.get("max_total_tokens", defaults.max_total_tokens)
    timeout = raw.get("timeout_seconds", defaults.timeout_seconds)
    return RunLimits(
        max_planning_steps=integer(
            raw.get("max_planning_steps", defaults.max_planning_steps),
            "max_planning_steps",
        ),
        max_tool_calls=integer(
            raw.get("max_tool_calls", defaults.max_tool_calls),
            "max_tool_calls",
        ),
        max_total_tokens=(None if max_total is None else integer(max_total, "max_total_tokens")),
        timeout_seconds=None if timeout is None else number(timeout, "timeout_seconds"),
        max_tool_concurrency=integer(
            raw.get("max_tool_concurrency", defaults.max_tool_concurrency),
            "max_tool_concurrency",
        ),
        max_tool_batch_size=integer(
            raw.get("max_tool_batch_size", defaults.max_tool_batch_size),
            "max_tool_batch_size",
        ),
        max_buffered_progress=integer(
            raw.get("max_buffered_progress", defaults.max_buffered_progress),
            "max_buffered_progress",
        ),
    )


def _last_checkpoint(exc: CommitError, repository: CaseRepository) -> Checkpoint:
    checkpoint = exc.last_checkpoint
    if not isinstance(checkpoint, Checkpoint):
        return repository.checkpoint
    return checkpoint


def _validate_wire(
    schemas: SchemaSuite,
    checkpoint: Checkpoint,
    events: Sequence[Event],
    commits: Sequence[Checkpoint],
    model: CaseModel,
    trace: RunTrace,
) -> None:
    unique_checkpoints = {item.id: item for item in (*commits, checkpoint)}
    for committed in unique_checkpoints.values():
        wire = encode_checkpoint(committed)
        schemas.validate("checkpoint.schema.json", wire)
        if decode_checkpoint(wire) != committed:
            raise AssertionError("checkpoint wire round trip changed the value")
    for event in events:
        wire = encode_event(event)
        schemas.validate("events.schema.json", wire)
        if decode_event(wire) != event:
            raise AssertionError("event wire round trip changed the value")
    for request in model.requests:
        schemas.validate("model-request.schema.json", _encode_model_request(request))
    trace_wire = encode_trace(trace)
    schemas.validate("run-trace.schema.json", trace_wire)
    decoded_trace = decode_trace(trace_wire)
    verify_trace(decoded_trace)
    if decoded_trace != trace:
        raise AssertionError("trace wire round trip changed the value")


def _encode_model_request(request: ModelRequest) -> dict[str, Any]:
    options = request.options
    choice = request.tool_choice
    choice_wire: dict[str, Any] = {
        "type": choice.type,
        "allow_parallel_tool_calls": choice.allow_parallel_tool_calls,
    }
    if choice.name is not None:
        choice_wire["name"] = choice.name
    response = request.response_format
    if response is None:
        response_wire = None
    elif response.type == "json_schema":
        response_wire = {
            "type": response.type,
            "schema": thaw_json_value(response.schema),
            "strict": response.strict,
        }
    else:
        response_wire = {"type": response.type}
    return {
        "messages": [encode_message(message) for message in request.messages],
        "tools": [encode_tool_spec(tool) for tool in request.tools],
        "options": {
            "model": options.model,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_output_tokens": options.max_output_tokens,
            "stop": None if options.stop is None else list(options.stop),
            "seed": options.seed,
            "metadata": thaw_json_value(options.metadata),
        },
        "tool_choice": choice_wire,
        "response_format": response_wire,
    }


def _optional_mapping(value: object, label: str) -> Mapping[str, Any] | None:
    return None if value is None else mapping(value, label)


def _optional_string(value: object, label: str) -> str | None:
    return None if value is None else string(value, label)


def _request_error_code(exc: Exception) -> str:
    if not isinstance(exc, RequestError):
        raise TypeError(f"unexpected request exception: {type(exc).__name__}") from exc
    return exc.code
