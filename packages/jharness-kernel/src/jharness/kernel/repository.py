"""Atomic checkpoint persistence port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol, TypeAlias, cast, runtime_checkable

from jharness.kernel._digest import (
    DigestWriter,
    append_history_digest,
    write_error,
    write_optional_integer,
    write_optional_string,
    write_parts,
)
from jharness.kernel._validation import expect_instance, expect_instance_tuple
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
from jharness.kernel.history import RunHistory
from jharness.kernel.messages import Message
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


def _digest_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    if len(value) != 32:
        raise ValueError(f"{label} must contain 32 bytes")
    return value


def _base_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("history base_count must be int")
    if value < 1:
        raise ValueError("history base_count must be >= 1")
    return value


@dataclass(frozen=True, slots=True)
class InitialHistory:
    """The complete history written by a run's first checkpoint."""

    history: RunHistory

    kind: ClassVar[str] = "initial"

    def __post_init__(self) -> None:
        expect_instance(self.history, RunHistory, "initial history")


@dataclass(frozen=True, slots=True)
class HistoryAppend:
    """A non-empty history suffix based on one exact committed history."""

    base_count: int
    base_digest: bytes
    messages: tuple[Message, ...]

    kind: ClassVar[str] = "append"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_count", _base_count(self.base_count))
        object.__setattr__(
            self,
            "base_digest",
            _digest_bytes(self.base_digest, "history base_digest"),
        )
        messages = expect_instance_tuple(self.messages, Message, "history append messages")
        if not messages:
            raise ValueError("history append messages must not be empty")
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class HistoryReplace:
    """A complete replacement based on one exact committed history."""

    base_count: int
    base_digest: bytes
    history: RunHistory

    kind: ClassVar[str] = "replace"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_count", _base_count(self.base_count))
        object.__setattr__(
            self,
            "base_digest",
            _digest_bytes(self.base_digest, "history base_digest"),
        )
        expect_instance(self.history, RunHistory, "replacement history")


@dataclass(frozen=True, slots=True)
class HistoryUnchanged:
    """An unchanged history based on one exact committed history."""

    base_count: int
    base_digest: bytes

    kind: ClassVar[str] = "unchanged"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_count", _base_count(self.base_count))
        object.__setattr__(
            self,
            "base_digest",
            _digest_bytes(self.base_digest, "history base_digest"),
        )


HistoryChange: TypeAlias = InitialHistory | HistoryAppend | HistoryReplace | HistoryUnchanged


@dataclass(frozen=True, slots=True)
class DurableCommit:
    """One validated checkpoint plus its minimal atomic history mutation."""

    checkpoint: Checkpoint
    parent_checkpoint_id: str | None
    history: HistoryChange
    digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        checkpoint = expect_instance(self.checkpoint, Checkpoint, "durable checkpoint")
        parent = cast(object, self.parent_checkpoint_id)
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ValueError("parent_checkpoint_id must be None or a non-empty string")
        change = cast(object, self.history)
        if not isinstance(change, HistoryChange):
            raise TypeError("durable history change has an unsupported type")
        if checkpoint.snapshot.revision == 0:
            if parent is not None or not isinstance(change, InitialHistory):
                raise ValueError("revision 0 durable commit requires initial history and no parent")
        elif parent is None or isinstance(change, InitialHistory):
            raise ValueError("advanced durable commit requires a parent and non-initial history")
        _validate_history_change(checkpoint.snapshot.history, self.history)
        object.__setattr__(self, "digest", checkpoint_digest(checkpoint))

    @property
    def run_id(self) -> str:
        return self.checkpoint.snapshot.context.run_id

    @property
    def checkpoint_id(self) -> str:
        return self.checkpoint.id

    @property
    def revision(self) -> int:
        return self.checkpoint.snapshot.revision

    @property
    def expected_revision(self) -> int | None:
        return None if self.revision == 0 else self.revision - 1

    @property
    def history_count(self) -> int:
        return len(self.checkpoint.snapshot.history)

    @property
    def history_digest(self) -> bytes:
        return self.checkpoint.snapshot._history_digest()  # pyright: ignore[reportPrivateUsage]

    @property
    def base_history_count(self) -> int | None:
        return None if isinstance(self.history, InitialHistory) else self.history.base_count

    @property
    def base_history_digest(self) -> bytes | None:
        return None if isinstance(self.history, InitialHistory) else self.history.base_digest


@runtime_checkable
class RunRepository(Protocol):
    """Atomically make one checkpoint authoritative."""

    async def commit(self, commit: DurableCommit) -> None: ...


class EphemeralRepository:
    """Single-run CAS repository retaining its head and fixed-size id fingerprints."""

    __slots__ = ("_by_id", "_head")

    def __init__(self, initial: Checkpoint | None = None) -> None:
        if initial is not None:
            expect_instance(initial, Checkpoint, "initial checkpoint")
        self._head = initial
        self._by_id: dict[tuple[str, str], bytes] = (
            {}
            if initial is None
            else {(initial.snapshot.context.run_id, initial.id): checkpoint_digest(initial)}
        )

    async def commit(self, commit: DurableCommit) -> None:
        commit = expect_instance(commit, DurableCommit, "durable commit")
        checkpoint = commit.checkpoint
        key = (commit.run_id, commit.checkpoint_id)
        existing = self._by_id.get(key)
        if existing is not None:
            if commit.digest == existing:
                return
            raise RepositoryError(
                f"checkpoint id {checkpoint.id!r} was reused with new content "
                f"in run {commit.run_id!r}"
            )

        head = self._head
        run_id = commit.run_id
        expected = commit.expected_revision
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
        if head is not None:
            if commit.parent_checkpoint_id != head.id:
                raise RepositoryError("parent checkpoint does not match the authoritative head")
            _validate_base(commit.history, head.snapshot.history)
        elif not isinstance(commit.history, InitialHistory):
            raise RepositoryError("first durable commit requires initial history")
        self._head = checkpoint
        self._by_id[key] = commit.digest


def checkpoint_digest(checkpoint: Checkpoint) -> bytes:
    """Return the canonical checkpoint digest without scanning its old history."""

    checkpoint = expect_instance(checkpoint, Checkpoint, "checkpoint")
    writer = DigestWriter("jharness.kernel.checkpoint.v1")
    writer.field("id")
    writer.string(checkpoint.id)
    writer.field("snapshot")
    _write_snapshot(writer, checkpoint.snapshot)
    writer.field("fact")
    _write_fact(writer, checkpoint.fact)
    return writer.finish()


def _validate_history_change(target: RunHistory, change: HistoryChange) -> None:
    target_count = len(target)
    target_digest = target._digest_bytes()  # pyright: ignore[reportPrivateUsage]
    if isinstance(change, InitialHistory):
        expected_count = len(change.history)
        expected_digest = change.history._digest_bytes()  # pyright: ignore[reportPrivateUsage]
    elif isinstance(change, HistoryAppend):
        expected_count = change.base_count + len(change.messages)
        expected_digest = change.base_digest
        for message in change.messages:
            expected_digest = append_history_digest(expected_digest, message)
    elif isinstance(change, HistoryReplace):
        expected_count = len(change.history)
        expected_digest = change.history._digest_bytes()  # pyright: ignore[reportPrivateUsage]
    else:
        expected_count = change.base_count
        expected_digest = change.base_digest
    if target_count != expected_count or target_digest != expected_digest:
        raise ValueError("history change does not produce the checkpoint history")


def _validate_base(change: HistoryChange, actual: RunHistory) -> None:
    if isinstance(change, InitialHistory):
        raise RepositoryError("advanced durable commit cannot contain initial history")
    if change.base_count != len(actual) or change.base_digest != actual._digest_bytes():  # pyright: ignore[reportPrivateUsage]
        raise RepositoryError("history change base does not match the authoritative head")


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
        _write_pending(writer, value)
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
        _write_pending(writer, value)


def _write_pending(writer: DigestWriter, value: ToolsPending) -> None:
    writer.field("pending_count")
    writer.integer(value.pending.pending_count)
    writer.field("pending_digest")
    writer.bytes(value.pending.digest)


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
