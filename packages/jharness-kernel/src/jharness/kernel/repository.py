"""Atomic checkpoint persistence port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jharness.kernel._digest import (
    DigestWriter,
    write_error,
    write_optional_integer,
    write_optional_string,
    write_parts,
    write_tool_calls,
)
from jharness.kernel._validation import expect_instance
from jharness.kernel.checkpoint import (
    Checkpoint,
    ControlDecision,
    ConversationInsertFact,
    Fact,
    FailedControl,
    HistoryRewriteFact,
    ModelTurnFact,
    ResumedFact,
    StartedFact,
    SuspendedControl,
    SuspensionView,
    ToolBatchFact,
)
from jharness.kernel.context import RunContext
from jharness.kernel.errors import RepositoryError, RevisionConflict
from jharness.kernel.models import ModelUsage
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import (
    Completed,
    Failed,
    Planning,
    RunMetrics,
    RunState,
    Suspended,
    Suspension,
    ToolsPending,
)


@runtime_checkable
class RunRepository(Protocol):
    """Atomically make one checkpoint authoritative."""

    async def commit(self, checkpoint: Checkpoint) -> None: ...


class EphemeralRepository:
    """Single-run CAS repository retaining its head and fixed-size id fingerprints."""

    __slots__ = ("_by_id", "_head")

    def __init__(self, initial: Checkpoint | None = None) -> None:
        if initial is not None:
            expect_instance(initial, Checkpoint, "initial checkpoint")
        self._head = initial
        self._by_id: dict[str, bytes] = (
            {} if initial is None else {initial.id: _fingerprint(initial)}
        )

    async def commit(self, checkpoint: Checkpoint) -> None:
        checkpoint = expect_instance(checkpoint, Checkpoint, "checkpoint")
        existing = self._by_id.get(checkpoint.id)
        if existing is not None:
            if _fingerprint(checkpoint) == existing:
                return
            raise RepositoryError(f"checkpoint id {checkpoint.id!r} was reused with new content")

        head = self._head
        run_id = checkpoint.snapshot.context.run_id
        expected = checkpoint.snapshot.revision - 1 if checkpoint.snapshot.revision else None
        if head is None:
            actual = None
        elif head.snapshot.context.run_id != run_id:
            raise RepositoryError(
                "an ephemeral invocation repository cannot contain more than one run"
            )
        else:
            actual = head.snapshot.revision
        if actual != expected:
            raise RevisionConflict(run_id, expected, actual)
        fingerprint = _fingerprint(checkpoint)
        self._head = checkpoint
        self._by_id[checkpoint.id] = fingerprint


def _fingerprint(checkpoint: Checkpoint) -> bytes:
    writer = DigestWriter("jharness.kernel.checkpoint.v0")
    writer.field("id")
    writer.string(checkpoint.id)
    writer.field("snapshot")
    _write_snapshot(writer, checkpoint.snapshot)
    writer.field("fact")
    _write_fact(writer, checkpoint.fact)
    return writer.finish()


def _write_snapshot(writer: DigestWriter, value: RunSnapshot) -> None:
    writer.field("revision")
    writer.integer(value.revision)
    writer.field("context")
    _write_context(writer, value.context)
    writer.field("history_count")
    writer.integer(len(value.history))
    writer.field("history_digest")
    writer.bytes(value._history_digest())  # pyright: ignore[reportPrivateUsage]
    writer.field("metrics")
    _write_metrics(writer, value.metrics)
    writer.field("state")
    _write_state(writer, value.state)


def _write_context(writer: DigestWriter, value: RunContext) -> None:
    writer.field("run_id")
    writer.string(value.run_id)
    writer.field("started_at")
    writer.number(value.started_at)
    writer.field("deadline")
    if value.deadline is None:
        writer.none()
    else:
        writer.number(value.deadline)
    writer.field("parent_run_id")
    write_optional_string(writer, value.parent_run_id)
    writer.field("parent_tool_call_id")
    write_optional_string(writer, value.parent_tool_call_id)
    writer.field("run_kind")
    write_optional_string(writer, value.run_kind)
    writer.field("metadata")
    writer.json(value.metadata)


def _write_metrics(writer: DigestWriter, value: RunMetrics) -> None:
    writer.field("planning_steps")
    writer.integer(value.planning_steps)
    writer.field("tool_calls")
    writer.integer(value.tool_calls)
    writer.field("usage")
    _write_usage(writer, value.usage)


def _write_usage(writer: DigestWriter, value: ModelUsage) -> None:
    writer.field("input_tokens")
    write_optional_integer(writer, value.input_tokens)
    writer.field("output_tokens")
    write_optional_integer(writer, value.output_tokens)
    writer.field("total_tokens")
    write_optional_integer(writer, value.total_tokens)
    writer.field("reasoning_tokens")
    write_optional_integer(writer, value.reasoning_tokens)
    writer.field("cache_read_tokens")
    write_optional_integer(writer, value.cache_read_tokens)
    writer.field("cache_write_tokens")
    write_optional_integer(writer, value.cache_write_tokens)


def _write_state(writer: DigestWriter, value: RunState) -> None:
    writer.field("kind")
    writer.string(value.kind)
    if isinstance(value, Planning):
        return
    if isinstance(value, ToolsPending):
        writer.field("pending")
        write_tool_calls(writer, value.pending)
    elif isinstance(value, Suspended):
        writer.field("resume_to")
        _write_active_state(writer, value.resume_to)
        writer.field("suspension")
        _write_suspension(writer, value.suspension)
    elif isinstance(value, Completed):
        writer.field("parts")
        write_parts(writer, value.parts)
    elif isinstance(value, Failed):
        writer.field("error")
        write_error(writer, value.error)
    else:
        writer.field("reason")
        writer.string(value.reason.value)


def _write_active_state(writer: DigestWriter, value: Planning | ToolsPending) -> None:
    writer.field("kind")
    writer.string(value.kind)
    if isinstance(value, ToolsPending):
        writer.field("pending")
        write_tool_calls(writer, value.pending)


def _write_suspension(writer: DigestWriter, value: Suspension) -> None:
    writer.field("reason")
    writer.string(value.reason)
    writer.field("source")
    writer.string(value.source)
    writer.field("wait_id")
    write_optional_string(writer, value.wait_id)
    writer.field("metadata")
    writer.json(value.metadata)


def _write_fact(writer: DigestWriter, value: Fact) -> None:
    writer.field("kind")
    writer.string(value.kind)
    writer.field("at")
    writer.number(value.at)
    if isinstance(value, StartedFact):
        writer.field("history_roles")
        _write_strings(writer, value.history_roles)
    elif isinstance(value, ResumedFact):
        _write_resumed_fact(writer, value)
    elif isinstance(value, ModelTurnFact):
        _write_model_fact(writer, value)
    elif isinstance(value, ToolBatchFact):
        _write_tool_fact(writer, value)
    elif isinstance(value, ConversationInsertFact):
        writer.field("source")
        writer.string(value.source)
    elif isinstance(value, HistoryRewriteFact):
        _write_history_fact(writer, value)
    else:
        writer.field("decision")
        _write_control(writer, value.decision)


def _write_resumed_fact(writer: DigestWriter, value: ResumedFact) -> None:
    writer.field("appended_roles")
    _write_strings(writer, value.appended_roles)
    writer.field("metadata_keys")
    _write_strings(writer, value.metadata_keys)


def _write_model_fact(writer: DigestWriter, value: ModelTurnFact) -> None:
    writer.field("result")
    writer.string(value.result.value)
    writer.field("part_count")
    writer.integer(value.part_count)
    writer.field("tool_call_ids")
    _write_strings(writer, value.tool_call_ids)
    writer.field("finish_reason")
    write_optional_string(writer, value.finish_reason)
    writer.field("usage")
    if value.usage is None:
        writer.none()
    else:
        _write_usage(writer, value.usage)
    writer.field("limit_reason")
    if value.limit_reason is None:
        writer.none()
    else:
        writer.string(value.limit_reason.value)


def _write_tool_fact(writer: DigestWriter, value: ToolBatchFact) -> None:
    writer.field("batch_id")
    writer.string(value.batch_id)
    writer.field("call_ids")
    _write_strings(writer, value.call_ids)
    writer.field("parallel")
    writer.boolean(value.parallel)
    writer.field("outcome_kinds")
    writer.sequence(len(value.outcome_kinds))
    for outcome in value.outcome_kinds:
        writer.string(outcome.value)
    writer.field("suspension")
    if value.suspension is None:
        writer.none()
    else:
        _write_suspension_view(writer, value.suspension)


def _write_suspension_view(writer: DigestWriter, value: SuspensionView) -> None:
    writer.field("reason")
    writer.string(value.reason)
    writer.field("source")
    writer.string(value.source)
    writer.field("wait_id")
    write_optional_string(writer, value.wait_id)
    writer.field("metadata_keys")
    _write_strings(writer, value.metadata_keys)


def _write_history_fact(writer: DigestWriter, value: HistoryRewriteFact) -> None:
    writer.field("before_count")
    writer.integer(value.before_count)
    writer.field("after_roles")
    _write_strings(writer, value.after_roles)
    writer.field("reason")
    writer.string(value.reason)
    writer.field("metadata_keys")
    _write_strings(writer, value.metadata_keys)


def _write_control(writer: DigestWriter, value: ControlDecision) -> None:
    writer.field("action")
    writer.string(value.action)
    if isinstance(value, SuspendedControl):
        writer.field("reason")
        writer.string(value.reason)
        writer.field("source")
        writer.string(value.source)
        writer.field("wait_id")
        write_optional_string(writer, value.wait_id)
        writer.field("metadata_keys")
        _write_strings(writer, value.metadata_keys)
    elif isinstance(value, FailedControl):
        writer.field("code")
        writer.string(value.code)
    else:
        writer.field("reason")
        writer.string(value.reason.value)


def _write_strings(writer: DigestWriter, values: tuple[str, ...]) -> None:
    writer.sequence(len(values))
    for value in values:
        writer.string(value)
