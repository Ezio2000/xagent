"""Atomic checkpoint and closed semantic fact values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TypeAlias, cast

from jharness.kernel._validation import (
    expect_bool,
    expect_instance,
    expect_int,
    expect_non_empty_str,
    expect_number,
    expect_optional_str,
    expect_sequence,
    expect_str,
)
from jharness.kernel.limits import LimitReason
from jharness.kernel.messages import ContentPart, Message
from jharness.kernel.models import ModelUsage
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import Completed, Failed, Limited, Planning, Suspended, ToolsPending

_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool", "external"})


def _at(value: object) -> float:
    number = expect_number(value, "fact at")
    if number < 0:
        raise ValueError("fact at must be >= 0")
    return number


def _strings(
    value: object,
    label: str,
    *,
    non_empty: bool = False,
    unique: bool = False,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    items = tuple(expect_str(item, f"{label} item") for item in expect_sequence(value, label))
    if non_empty and not items:
        raise ValueError(f"{label} must not be empty")
    if any(not item for item in items):
        raise ValueError(f"{label} items must not be empty")
    if unique and len(items) != len(set(items)):
        raise ValueError(f"{label} items must be unique")
    if allowed is not None and any(item not in allowed for item in items):
        raise ValueError(f"{label} contains an unsupported value")
    return items


def _metadata_keys(value: object, label: str) -> tuple[str, ...]:
    return _strings(value, label, unique=True)


def _roles(value: object, label: str, *, non_empty: bool = False) -> tuple[str, ...]:
    return _strings(value, label, non_empty=non_empty, allowed=_MESSAGE_ROLES)


class ModelTurnResult(StrEnum):
    COMPLETED = "completed"
    TOOLS_PENDING = "tools_pending"
    LIMITED = "limited"


class ToolOutcomeKind(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    ACCEPTED = "accepted"
    WAITING = "waiting"


@dataclass(frozen=True, slots=True)
class SuspensionView:
    """Compact suspension projection stored in a semantic fact."""

    reason: str
    source: str
    wait_id: str | None
    metadata_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        expect_non_empty_str(self.reason, "suspension reason")
        expect_non_empty_str(self.source, "suspension source")
        wait_id = expect_optional_str(self.wait_id, "suspension wait_id")
        if wait_id == "":
            raise ValueError("suspension wait_id must not be empty")
        object.__setattr__(
            self,
            "metadata_keys",
            _metadata_keys(self.metadata_keys, "suspension metadata_keys"),
        )


@dataclass(frozen=True, slots=True)
class StartedFact:
    at: float
    history_roles: tuple[str, ...]

    kind: ClassVar[str] = "started"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        object.__setattr__(
            self,
            "history_roles",
            _roles(self.history_roles, "started history_roles", non_empty=True),
        )


@dataclass(frozen=True, slots=True)
class ResumedFact:
    at: float
    appended_roles: tuple[str, ...]
    metadata_keys: tuple[str, ...]

    kind: ClassVar[str] = "resumed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        object.__setattr__(
            self,
            "appended_roles",
            _roles(self.appended_roles, "resumed appended_roles"),
        )
        object.__setattr__(
            self,
            "metadata_keys",
            _metadata_keys(self.metadata_keys, "resumed metadata_keys"),
        )


@dataclass(frozen=True, slots=True)
class ModelTurnFact:
    at: float
    result: ModelTurnResult
    part_count: int
    tool_call_ids: tuple[str, ...]
    finish_reason: str | None
    usage: ModelUsage | None
    limit_reason: LimitReason | None

    kind: ClassVar[str] = "model_turn"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        expect_instance(self.result, ModelTurnResult, "model turn result")
        part_count = expect_int(self.part_count, "model turn part_count")
        if part_count < 0:
            raise ValueError("model turn part_count must be >= 0")
        call_ids = _strings(self.tool_call_ids, "model turn tool_call_ids", unique=True)
        object.__setattr__(self, "tool_call_ids", call_ids)
        expect_optional_str(self.finish_reason, "model turn finish_reason")
        if self.usage is not None:
            expect_instance(self.usage, ModelUsage, "model turn usage")
        if self.limit_reason is not None:
            expect_instance(self.limit_reason, LimitReason, "model turn limit_reason")
        self._validate_result(part_count, call_ids)

    def _validate_result(self, part_count: int, call_ids: tuple[str, ...]) -> None:
        if self.result is ModelTurnResult.COMPLETED:
            if part_count < 1 or call_ids or self.limit_reason is not None:
                raise ValueError("completed model turn requires parts and no calls or limit")
        elif self.result is ModelTurnResult.TOOLS_PENDING:
            if not call_ids or self.limit_reason is not None:
                raise ValueError("tools_pending model turn requires calls and no limit")
        elif self.limit_reason is not LimitReason.MAX_TOTAL_TOKENS:
            raise ValueError("limited model turn requires max_total_tokens")


@dataclass(frozen=True, slots=True)
class ToolBatchFact:
    at: float
    batch_id: str
    call_ids: tuple[str, ...]
    parallel: bool
    outcome_kinds: tuple[ToolOutcomeKind, ...]
    suspension: SuspensionView | None

    kind: ClassVar[str] = "tool_batch"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        expect_non_empty_str(self.batch_id, "tool batch id")
        call_ids = _strings(
            self.call_ids,
            "tool batch call_ids",
            non_empty=True,
            unique=True,
        )
        object.__setattr__(self, "call_ids", call_ids)
        expect_bool(self.parallel, "tool batch parallel")
        outcomes = tuple(self.outcome_kinds)
        if not outcomes or any(
            not isinstance(cast(object, item), ToolOutcomeKind) for item in outcomes
        ):
            raise TypeError("tool batch outcome_kinds must contain ToolOutcomeKind values")
        if len(call_ids) != len(outcomes):
            raise ValueError("tool batch call_ids and outcome_kinds must have equal length")
        object.__setattr__(self, "outcome_kinds", outcomes)
        if self.suspension is not None:
            expect_instance(self.suspension, SuspensionView, "tool batch suspension")
        if ToolOutcomeKind.WAITING in outcomes and self.suspension is None:
            raise ValueError("a waiting tool outcome requires suspension")


@dataclass(frozen=True, slots=True)
class ConversationInsertFact:
    at: float
    source: str

    kind: ClassVar[str] = "conversation_insert"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        expect_non_empty_str(self.source, "conversation insert source")


@dataclass(frozen=True, slots=True)
class HistoryRewriteFact:
    at: float
    before_count: int
    after_roles: tuple[str, ...]
    reason: str
    metadata_keys: tuple[str, ...]

    kind: ClassVar[str] = "history_rewrite"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        before = expect_int(self.before_count, "history rewrite before_count")
        if before < 1:
            raise ValueError("history rewrite before_count must be >= 1")
        roles = _roles(self.after_roles, "history rewrite after_roles", non_empty=True)
        if len(roles) > before:
            raise ValueError("history rewrite cannot increase message count")
        object.__setattr__(self, "after_roles", roles)
        expect_non_empty_str(self.reason, "history rewrite reason")
        object.__setattr__(
            self,
            "metadata_keys",
            _metadata_keys(self.metadata_keys, "history rewrite metadata_keys"),
        )


@dataclass(frozen=True, slots=True)
class SuspendedControl:
    reason: str
    source: str
    wait_id: str | None
    metadata_keys: tuple[str, ...]

    action: ClassVar[str] = "suspended"

    def __post_init__(self) -> None:
        expect_non_empty_str(self.reason, "control suspension reason")
        expect_non_empty_str(self.source, "control suspension source")
        wait_id = expect_optional_str(self.wait_id, "control suspension wait_id")
        if wait_id == "":
            raise ValueError("control suspension wait_id must not be empty")
        object.__setattr__(
            self,
            "metadata_keys",
            _metadata_keys(self.metadata_keys, "control suspension metadata_keys"),
        )


@dataclass(frozen=True, slots=True)
class FailedControl:
    code: str

    action: ClassVar[str] = "failed"

    def __post_init__(self) -> None:
        expect_non_empty_str(self.code, "control failure code")


@dataclass(frozen=True, slots=True)
class LimitedControl:
    reason: LimitReason

    action: ClassVar[str] = "limited"

    def __post_init__(self) -> None:
        expect_instance(self.reason, LimitReason, "control limit reason")


ControlDecision: TypeAlias = SuspendedControl | FailedControl | LimitedControl


@dataclass(frozen=True, slots=True)
class ControlFact:
    at: float
    decision: ControlDecision

    kind: ClassVar[str] = "control"

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _at(self.at))
        if not isinstance(cast(object, self.decision), ControlDecision):
            raise TypeError("control decision has an unsupported type")


Fact: TypeAlias = (
    StartedFact
    | ResumedFact
    | ModelTurnFact
    | ToolBatchFact
    | ConversationInsertFact
    | HistoryRewriteFact
    | ControlFact
)


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One idempotent atomic snapshot-plus-fact durability unit."""

    id: str
    snapshot: RunSnapshot
    fact: Fact

    def __post_init__(self) -> None:
        expect_non_empty_str(self.id, "checkpoint id")
        snapshot = expect_instance(self.snapshot, RunSnapshot, "checkpoint snapshot")
        fact = cast(object, self.fact)
        if not isinstance(fact, Fact):
            raise TypeError("checkpoint fact has an unsupported type")
        _validate_boundary(snapshot, self.fact)


def _validate_boundary(snapshot: RunSnapshot, fact: Fact) -> None:
    if isinstance(fact, StartedFact):
        _validate_started_boundary(snapshot, fact)
        return
    if snapshot.revision == 0:
        raise ValueError("revision 0 checkpoint requires a started fact")
    if isinstance(fact, ResumedFact):
        _validate_resumed_boundary(snapshot)
    elif isinstance(fact, ModelTurnFact):
        _validate_model_boundary(snapshot, fact)
    elif isinstance(fact, ToolBatchFact):
        _validate_tool_boundary(snapshot, fact)
    elif isinstance(fact, ConversationInsertFact | HistoryRewriteFact):
        _validate_history_boundary(snapshot, fact)
    else:
        _validate_control_boundary(snapshot, fact.decision)


def _validate_started_boundary(snapshot: RunSnapshot, fact: StartedFact) -> None:
    if snapshot.revision != 0 or not isinstance(snapshot.state, Planning):
        raise ValueError("started checkpoint must be revision 0 in Planning")
    if fact.history_roles != tuple(message.role for message in snapshot.history):
        raise ValueError("started fact roles must match snapshot history")


def _validate_resumed_boundary(snapshot: RunSnapshot) -> None:
    if not isinstance(snapshot.state, Planning | ToolsPending):
        raise ValueError("resumed checkpoint must restore an active state")


def _validate_tool_boundary(snapshot: RunSnapshot, fact: ToolBatchFact) -> None:
    if fact.suspension is None and not isinstance(snapshot.state, Planning | ToolsPending):
        raise ValueError("settled tool batch must leave an active state")
    if fact.suspension is not None and not isinstance(snapshot.state, Suspended):
        raise ValueError("suspended tool batch must leave Suspended state")
    if isinstance(snapshot.state, Suspended) and not _matches_suspension(
        fact.suspension,
        snapshot.state.suspension,
    ):
        raise ValueError("tool batch suspension must match Suspended state")
    count = len(fact.call_ids)
    messages = snapshot.history[-count:]
    if len(messages) != count or any(message.role != "tool" for message in messages):
        raise ValueError("tool batch fact requires matching trailing tool messages")
    if tuple(message.tool_call_id for message in messages) != fact.call_ids:
        raise ValueError("tool batch call ids must match trailing tool messages")
    if tuple(
        None if message.outcome is None else message.outcome.kind for message in messages
    ) != tuple(kind.value for kind in fact.outcome_kinds):
        raise ValueError("tool batch outcomes must match trailing tool messages")


def _validate_planning_boundary(snapshot: RunSnapshot, kind: str) -> None:
    if not isinstance(snapshot.state, Planning):
        raise ValueError(f"{kind} checkpoint must leave Planning state")


def _validate_history_boundary(
    snapshot: RunSnapshot,
    fact: ConversationInsertFact | HistoryRewriteFact,
) -> None:
    _validate_planning_boundary(snapshot, fact.kind)
    if isinstance(fact, ConversationInsertFact):
        if snapshot.history[-1].role != "external":
            raise ValueError("conversation insert must append an external message")
    elif fact.after_roles != tuple(message.role for message in snapshot.history):
        raise ValueError("history rewrite roles must match snapshot history")


def _validate_model_boundary(snapshot: RunSnapshot, fact: ModelTurnFact) -> None:
    assistant = _validate_model_message(snapshot, fact)
    if fact.result is ModelTurnResult.COMPLETED:
        _validate_completed_model(snapshot, fact, assistant.parts)
    elif fact.result is ModelTurnResult.TOOLS_PENDING:
        if not isinstance(snapshot.state, ToolsPending):
            raise ValueError("tools_pending model fact must match ToolsPending state")
        if snapshot.state.pending != assistant.tool_calls:
            raise ValueError("pending calls must match the assistant message")
    elif not isinstance(snapshot.state, Limited) or snapshot.state.reason is not fact.limit_reason:
        raise ValueError("limited model fact must match Limited state")


def _validate_model_message(snapshot: RunSnapshot, fact: ModelTurnFact) -> Message:
    assistant = snapshot.history[-1]
    if assistant.role != "assistant":
        raise ValueError("model turn fact requires a trailing assistant message")
    if len(assistant.parts) != fact.part_count:
        raise ValueError("model turn part_count must match the assistant message")
    if tuple(call.id for call in assistant.tool_calls) != fact.tool_call_ids:
        raise ValueError("model turn call ids must match the assistant message")
    return assistant


def _validate_completed_model(
    snapshot: RunSnapshot,
    fact: ModelTurnFact,
    assistant_parts: tuple[ContentPart, ...],
) -> None:
    if not isinstance(snapshot.state, Completed) or len(snapshot.state.parts) != fact.part_count:
        raise ValueError("completed model fact must match Completed state")
    if snapshot.state.parts != assistant_parts:
        raise ValueError("Completed parts must match the assistant message")


def _validate_control_boundary(snapshot: RunSnapshot, decision: ControlDecision) -> None:
    if isinstance(decision, SuspendedControl):
        if not isinstance(snapshot.state, Suspended):
            raise ValueError("suspended control fact must match Suspended state")
        view = SuspensionView(
            decision.reason,
            decision.source,
            decision.wait_id,
            decision.metadata_keys,
        )
        if not _matches_suspension(view, snapshot.state.suspension):
            raise ValueError("suspended control must match Suspended state")
    elif isinstance(decision, FailedControl):
        if not isinstance(snapshot.state, Failed) or snapshot.state.error.code != decision.code:
            raise ValueError("failed control fact must match Failed state")
    elif not isinstance(snapshot.state, Limited) or snapshot.state.reason is not decision.reason:
        raise ValueError("limited control fact must match Limited state")


def _matches_suspension(view: SuspensionView | None, suspension: object) -> bool:
    from jharness.kernel.state import Suspension

    if view is None or not isinstance(suspension, Suspension):
        return False
    return (
        view.reason == suspension.reason
        and view.source == suspension.source
        and view.wait_id == suspension.wait_id
        and set(view.metadata_keys) == set(suspension.metadata)
    )
