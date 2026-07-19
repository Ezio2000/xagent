"""Single-pass verification for compact invocation traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from jharness.kernel._digest import compose_call_id_digest
from jharness.kernel._engine.verification import verify_change
from jharness.kernel.diagnostics.trace import RunTrace, TraceEntry
from jharness.kernel.events import EventKind

_ACTIVE_STATES = frozenset({"planning", "tools_pending"})
_CLOSED_STATES = frozenset({"suspended", "completed", "failed", "limited"})
_TERMINAL_STATES = frozenset({"completed", "failed", "limited"})
_STOP_REASONS = frozenset(
    {"terminal", "suspended", "cancelled", "consumer_closed", "repository_error"}
)
_OUTCOME_KINDS = frozenset({"success", "failure", "accepted", "waiting"})

RunView = Mapping[str, Any]


class TraceError(ValueError):
    """Stable failure raised when trace evidence is internally inconsistent."""

    __slots__ = ("code", "sequence")

    def __init__(self, code: str, message: str, *, sequence: int | None = None) -> None:
        if not code:
            raise ValueError("trace error code must not be empty")
        prefix = "trace" if sequence is None else f"trace entry {sequence}"
        super().__init__(f"{prefix} [{code}]: {message}")
        self.code = code
        self.sequence = sequence


@dataclass(frozen=True, slots=True)
class TraceVerification:
    """Verified durable endpoint and compact event counts."""

    entry_count: int
    checkpoint_count: int
    live_event_count: int
    final_checkpoint_id: str | None
    final_view: RunView | None


@dataclass(frozen=True, slots=True)
class _ToolObservation:
    batch_id: str
    index: int
    parallel: bool
    outcome_kind: str | None = None


@dataclass(frozen=True, slots=True)
class _BatchSelection:
    batch_id: str
    call_ids: tuple[str, ...]
    parallel: bool
    remaining_count: int
    remaining_call_id_digest: bytes


class _Verifier:
    __slots__ = (
        "_active_tools",
        "_approvals_decided",
        "_approvals_waiting",
        "_checkpoint_count",
        "_checkpoint_ids",
        "_closed",
        "_current",
        "_first_commit",
        "_last_checkpoint_id",
        "_model_active",
        "_model_finished",
        "_seen_tools",
        "_selected_batch",
        "_settled_tools",
        "_trace",
    )

    def __init__(self, trace: RunTrace) -> None:
        self._trace = trace
        self._current: RunView | None = None
        self._last_checkpoint_id: str | None = None
        self._checkpoint_ids: set[str] = set()
        self._checkpoint_count = 0
        self._first_commit = True
        self._closed = False
        self._model_active = False
        self._model_finished: Mapping[str, Any] | None = None
        self._selected_batch: _BatchSelection | None = None
        self._active_tools: dict[str, _ToolObservation] = {}
        self._settled_tools: dict[str, _ToolObservation] = {}
        self._seen_tools: set[str] = set()
        self._approvals_waiting: set[str] = set()
        self._approvals_decided: set[str] = set()

    def run(self) -> TraceVerification:
        entries = self._trace.entries
        self._verify_order(entries)
        self._start(entries[0])
        for entry in entries[1:-1]:
            self._visit(entry)
        self._stop(entries[-1])
        return TraceVerification(
            entry_count=len(entries),
            checkpoint_count=self._checkpoint_count,
            live_event_count=len(entries) - self._checkpoint_count,
            final_checkpoint_id=self._last_checkpoint_id,
            final_view=self._current,
        )

    def _verify_order(self, entries: tuple[TraceEntry, ...]) -> None:
        if entries[0].kind is not EventKind.INVOCATION_STARTED:
            _fail(entries[0], "lifecycle", "trace must start with invocation_started")
        if entries[-1].kind is not EventKind.INVOCATION_STOPPED:
            _fail(entries[-1], "lifecycle", "trace must end with invocation_stopped")
        previous = 0
        for entry in entries:
            if entry.sequence <= previous:
                _fail(entry, "sequence_order", "entry sequences must strictly increase")
            previous = entry.sequence

    def _start(self, entry: TraceEntry) -> None:
        data = entry.data
        request_kind = _string(entry, _required(entry, data, "request_kind"), "request_kind")
        if request_kind != self._trace.header.request_kind:
            _fail(entry, "request_mismatch", "header and invocation_started request_kind differ")
        checkpoint_id = _optional_string(
            entry,
            _required(entry, data, "starting_checkpoint_id"),
            "starting_checkpoint_id",
        )
        starting = _optional_mapping(entry, _required(entry, data, "starting"), "starting")
        if request_kind == "start":
            if checkpoint_id is not None or starting is not None:
                _fail(entry, "request_mismatch", "start cannot carry a starting checkpoint")
        else:
            if checkpoint_id is None or starting is None:
                _fail(entry, "request_mismatch", "continue/resume requires a starting checkpoint")
            _validate_view(entry, starting)
            state_kind = _state_kind(entry, starting)
            if request_kind == "continue" and state_kind not in _ACTIVE_STATES:
                _fail(entry, "request_mismatch", "continue requires an active starting view")
            if request_kind == "resume" and state_kind != "suspended":
                _fail(entry, "request_mismatch", "resume requires a suspended starting view")
        self._current = starting
        self._last_checkpoint_id = checkpoint_id
        if checkpoint_id is not None:
            self._checkpoint_ids.add(checkpoint_id)

    def _visit(self, entry: TraceEntry) -> None:
        if self._closed:
            _fail(entry, "lifecycle", "a closed checkpoint must be followed by invocation_stopped")
        handlers = {
            EventKind.MODEL_STARTED: self._model_started,
            EventKind.MODEL_DELTA: self._model_delta,
            EventKind.MODEL_FINISHED: self._model_completed,
            EventKind.TOOL_BATCH_SELECTED: self._tool_batch_selected,
            EventKind.APPROVAL_REQUESTED: self._approval_requested,
            EventKind.APPROVAL_DECIDED: self._approval_decided,
            EventKind.TOOL_STARTED: self._tool_started,
            EventKind.TOOL_PROGRESS: self._tool_progress,
            EventKind.TOOL_FINISHED: self._tool_finished,
            EventKind.TOOL_CANCEL_REQUESTED: self._tool_cancel_requested,
            EventKind.CHECKPOINT_COMMITTED: self._checkpoint,
        }
        handler = handlers.get(entry.kind)
        if handler is None:
            _fail(entry, "lifecycle", f"{entry.kind.value} may not appear here")
        handler(entry)

    def _model_started(self, entry: TraceEntry) -> None:
        if self._current is None or _state_kind(entry, self._current) != "planning":
            _fail(entry, "model_lifecycle", "model_started requires Planning")
        if self._model_active or self._active_tools:
            _fail(entry, "model_lifecycle", "another effect is already active")
        metrics = _mapping(entry, _required(entry, self._current, "metrics"), "metrics")
        planning_steps = _integer(
            entry,
            _required(entry, metrics, "planning_steps"),
            "planning_steps",
            minimum=0,
        )
        observed = _integer(
            entry,
            _required(entry, entry.data, "planning_step"),
            "planning_step",
            minimum=1,
        )
        if observed != planning_steps + 1:
            _fail(entry, "model_lifecycle", "planning_step does not follow durable metrics")
        self._model_active = True
        self._model_finished = None

    def _model_delta(self, entry: TraceEntry) -> None:
        if not self._model_active:
            _fail(entry, "model_lifecycle", "model_delta requires an active model")

    def _model_completed(self, entry: TraceEntry) -> None:
        if not self._model_active:
            _fail(entry, "model_lifecycle", "model_finished requires an active model")
        self._model_active = False
        self._model_finished = entry.data

    def _tool_batch_selected(self, entry: TraceEntry) -> None:
        self._require_pending_state(entry, "tool_batch_selected")
        if (
            self._model_active
            or self._selected_batch is not None
            or self._active_tools
            or self._approvals_waiting
        ):
            _fail(entry, "tool_lifecycle", "tool batch selection requires an idle boundary")
        batch_id = _string(
            entry,
            _required(entry, entry.data, "batch_id"),
            "selected batch_id",
            non_empty=True,
        )
        call_ids = _string_sequence(
            entry,
            _required(entry, entry.data, "call_ids"),
            "selected call_ids",
        )
        if not call_ids or len(call_ids) != len(set(call_ids)):
            _fail(entry, "tool_lifecycle", "selected call_ids must be non-empty and unique")
        parallel = _boolean(
            entry,
            _required(entry, entry.data, "parallel"),
            "selected parallel",
        )
        remaining_count = _integer(
            entry,
            _required(entry, entry.data, "remaining_count"),
            "selected remaining_count",
            minimum=0,
        )
        remaining_digest = _digest_hex(
            entry,
            _required(entry, entry.data, "remaining_call_id_digest"),
            "selected remaining_call_id_digest",
        )
        current = cast(RunView, self._current)
        state = _mapping(entry, _required(entry, current, "state"), "view state")
        pending_count = _integer(
            entry,
            _required(entry, state, "pending_count"),
            "pending_count",
            minimum=1,
        )
        pending_digest = _digest_hex(
            entry,
            _required(entry, state, "call_id_digest"),
            "call_id_digest",
        )
        if (
            len(call_ids) + remaining_count != pending_count
            or compose_call_id_digest(
                call_ids,
                remaining_digest,
            )
            != pending_digest
        ):
            _fail(entry, "tool_lifecycle", "selected batch is not the pending prefix")
        self._selected_batch = _BatchSelection(
            batch_id,
            call_ids,
            parallel,
            remaining_count,
            remaining_digest,
        )

    def _approval_requested(self, entry: TraceEntry) -> None:
        self._require_pending_state(entry, "approval_requested")
        selected = self._selected_batch
        if self._model_active or self._active_tools or selected is None:
            _fail(entry, "approval_lifecycle", "approval requires idle tool preparation")
        call = _mapping(entry, _required(entry, entry.data, "call"), "approval call")
        call_id = _string(entry, _required(entry, call, "id"), "approval call id", non_empty=True)
        _validate_selected_envelope(entry, selected, call_id, approval=True)
        if call_id in self._approvals_waiting | self._approvals_decided:
            _fail(entry, "approval_lifecycle", "approval was requested more than once")
        self._approvals_waiting.add(call_id)

    def _approval_decided(self, entry: TraceEntry) -> None:
        call_id = _string(
            entry,
            _required(entry, entry.data, "call_id"),
            "approval call_id",
            non_empty=True,
        )
        if call_id not in self._approvals_waiting:
            _fail(entry, "approval_lifecycle", "approval decision has no open request")
        self._approvals_waiting.remove(call_id)
        self._approvals_decided.add(call_id)

    def _tool_started(self, entry: TraceEntry) -> None:
        self._require_pending_state(entry, "tool_started")
        selected = self._selected_batch
        if self._model_active or self._approvals_waiting or selected is None:
            _fail(entry, "tool_lifecycle", "tool_started requires settled preparation")
        call = _mapping(entry, _required(entry, entry.data, "call"), "tool call")
        call_id = _string(entry, _required(entry, call, "id"), "tool call id", non_empty=True)
        _validate_selected_envelope(entry, selected, call_id, approval=False)
        if call_id in self._seen_tools:
            _fail(entry, "tool_lifecycle", "tool call started more than once in a boundary")
        batch_id = _string(
            entry,
            _required(entry, entry.data, "batch_id"),
            "tool batch_id",
            non_empty=True,
        )
        index = _integer(
            entry,
            _required(entry, entry.data, "index"),
            "tool index",
            minimum=0,
        )
        parallel = _boolean(entry, _required(entry, entry.data, "parallel"), "tool parallel")
        self._seen_tools.add(call_id)
        self._active_tools[call_id] = _ToolObservation(batch_id, index, parallel)

    def _tool_progress(self, entry: TraceEntry) -> None:
        self._active_tool(entry, "tool_progress")

    def _tool_cancel_requested(self, entry: TraceEntry) -> None:
        self._active_tool(entry, "tool_cancel_requested")

    def _tool_finished(self, entry: TraceEntry) -> None:
        call_id, started = self._active_tool(entry, "tool_finished")
        batch_id = _string(
            entry,
            _required(entry, entry.data, "batch_id"),
            "tool batch_id",
            non_empty=True,
        )
        index = _integer(
            entry,
            _required(entry, entry.data, "index"),
            "tool index",
            minimum=0,
        )
        outcome = _string(
            entry,
            _required(entry, entry.data, "outcome_kind"),
            "tool outcome_kind",
        )
        if outcome not in _OUTCOME_KINDS:
            _fail(entry, "tool_lifecycle", "tool outcome_kind is unsupported")
        if (batch_id, index) != (started.batch_id, started.index):
            _fail(entry, "tool_lifecycle", "tool finish does not match its start")
        del self._active_tools[call_id]
        self._settled_tools[call_id] = _ToolObservation(
            batch_id,
            index,
            started.parallel,
            outcome,
        )

    def _active_tool(self, entry: TraceEntry, label: str) -> tuple[str, _ToolObservation]:
        call_id = _string(
            entry,
            _required(entry, entry.data, "tool_call_id"),
            f"{label} tool_call_id",
            non_empty=True,
        )
        started = self._active_tools.get(call_id)
        if started is None:
            _fail(entry, "tool_lifecycle", f"{label} requires an active tool")
        return call_id, started

    def _require_pending_state(self, entry: TraceEntry, label: str) -> None:
        if self._current is None or _state_kind(entry, self._current) != "tools_pending":
            _fail(entry, "tool_lifecycle", f"{label} requires ToolsPending")

    def _checkpoint(self, entry: TraceEntry) -> None:
        checkpoint_id = _string(
            entry,
            _required(entry, entry.data, "checkpoint_id"),
            "checkpoint_id",
            non_empty=True,
        )
        if checkpoint_id in self._checkpoint_ids:
            _fail(entry, "duplicate_checkpoint_id", "checkpoint ids must be unique")
        fact = _mapping(entry, _required(entry, entry.data, "fact"), "checkpoint fact")
        after = _mapping(entry, _required(entry, entry.data, "after"), "checkpoint after")
        _validate_view(entry, after)
        fact_kind = _string(entry, _required(entry, fact, "kind"), "fact kind")
        if fact_kind == "tool_batch":
            _tool_fact_fields(entry, fact)
        if (
            self._first_commit
            and self._trace.header.request_kind == "resume"
            and fact_kind != "resumed"
            and not _is_deadline_control(fact)
        ):
            _fail(
                entry,
                "request_mismatch",
                "resume must first commit resumed or an expired-deadline terminal fact",
            )
        try:
            verify_change(self._current, fact, after)
        except ValueError as exc:
            code = str(exc) or "change_invalid"
            _fail(entry, code, "checkpoint fact does not produce its after view")
        except (KeyError, TypeError) as exc:
            _fail(entry, "change_invalid", f"invalid checkpoint evidence: {exc}")
        self._verify_boundary_events(entry, fact_kind, fact)
        self._checkpoint_ids.add(checkpoint_id)
        self._last_checkpoint_id = checkpoint_id
        self._checkpoint_count += 1
        self._first_commit = False
        self._current = after
        self._closed = _state_kind(entry, after) in _CLOSED_STATES
        self._reset_boundary_events()

    def _verify_boundary_events(
        self,
        entry: TraceEntry,
        fact_kind: str,
        fact: Mapping[str, Any],
    ) -> None:
        if fact_kind == "model_turn":
            self._verify_model_boundary(entry, fact)
        elif fact_kind == "tool_batch":
            self._verify_tool_boundary(entry, fact)
        elif fact_kind not in {"control", "conversation_insert"} and self._has_active_effect():
            _fail(entry, "effect_unsettled", f"{fact_kind} cannot settle an active effect")

    def _verify_model_boundary(self, entry: TraceEntry, fact: Mapping[str, Any]) -> None:
        if self._model_active or self._model_finished is None or self._active_tools:
            _fail(entry, "model_lifecycle", "model_turn requires a finished model effect")
        if self._approvals_waiting:
            _fail(entry, "approval_lifecycle", "model_turn cannot settle approvals")
        data = _mapping(entry, _required(entry, fact, "data"), "model fact data")
        calls = _string_sequence(
            entry,
            _required(entry, data, "tool_call_ids"),
            "model fact tool_call_ids",
        )
        finished = self._model_finished
        count = _integer(
            entry,
            _required(entry, finished, "tool_call_count"),
            "model_finished tool_call_count",
            minimum=0,
        )
        if count != len(calls):
            _fail(entry, "model_fact_mismatch", "model tool-call count differs from its fact")
        for key in ("finish_reason", "usage"):
            if _required(entry, finished, key) != _required(entry, data, key):
                _fail(entry, "model_fact_mismatch", f"model {key} differs from its fact")

    def _verify_tool_boundary(self, entry: TraceEntry, fact: Mapping[str, Any]) -> None:
        if self._model_active or self._active_tools:
            _fail(entry, "tool_lifecycle", "tool_batch requires every started tool to settle")
        if self._approvals_waiting:
            _fail(entry, "approval_lifecycle", "tool_batch requires every approval to settle")
        batch_id, parallel, call_ids, outcomes = _tool_fact_fields(entry, fact)
        selected = self._selected_batch
        if selected is None or (batch_id, call_ids, parallel) != (
            selected.batch_id,
            selected.call_ids,
            selected.parallel,
        ):
            _fail(entry, "tool_fact_mismatch", "tool fact differs from selected batch")
        fact_calls = set(call_ids)
        observed_calls = set(self._settled_tools)
        required_calls = {
            call_id
            for call_id, outcome in zip(call_ids, outcomes, strict=True)
            if outcome != "failure"
        }
        if not observed_calls <= fact_calls or not required_calls <= observed_calls:
            _fail(
                entry,
                "tool_fact_mismatch",
                "tool fact calls do not match required observed completions",
            )
        expected = {
            call_id: (index, outcome)
            for index, (call_id, outcome) in enumerate(zip(call_ids, outcomes, strict=True))
        }
        for call_id, observed in self._settled_tools.items():
            pair = expected.get(call_id)
            if pair != (observed.index, observed.outcome_kind):
                _fail(entry, "tool_fact_mismatch", "tool completion differs from its fact")
            if observed.batch_id != batch_id or observed.parallel is not parallel:
                _fail(entry, "tool_fact_mismatch", "tool batch envelope differs from its fact")

    def _has_active_effect(self) -> bool:
        return bool(
            self._model_active
            or self._selected_batch is not None
            or self._active_tools
            or self._approvals_waiting
        )

    def _reset_boundary_events(self) -> None:
        self._model_active = False
        self._model_finished = None
        self._selected_batch = None
        self._active_tools.clear()
        self._settled_tools.clear()
        self._seen_tools.clear()
        self._approvals_waiting.clear()
        self._approvals_decided.clear()

    def _stop(self, entry: TraceEntry) -> None:
        reason = _string(entry, _required(entry, entry.data, "reason"), "stop reason")
        if reason not in _STOP_REASONS:
            _fail(entry, "stop_mismatch", "invocation stop reason is unsupported")
        checkpoint_id = _optional_string(
            entry,
            _required(entry, entry.data, "last_checkpoint_id"),
            "last_checkpoint_id",
        )
        if checkpoint_id != self._last_checkpoint_id:
            _fail(entry, "stop_mismatch", "last_checkpoint_id is not the durable endpoint")
        state_kind = None if self._current is None else _state_kind(entry, self._current)
        if reason == "terminal" and state_kind not in _TERMINAL_STATES:
            _fail(entry, "stop_mismatch", "terminal stop requires a terminal view")
        if reason == "suspended" and state_kind != "suspended":
            _fail(entry, "stop_mismatch", "suspended stop requires a suspended view")
        if reason in {"terminal", "suspended"} and self._has_active_effect():
            _fail(entry, "effect_unsettled", "normal stop cannot retain active effects")
        if self._closed:
            expected = "suspended" if state_kind == "suspended" else "terminal"
            if reason != expected:
                _fail(entry, "stop_mismatch", "closed checkpoint and stop reason differ")


def _is_deadline_control(fact: Mapping[str, Any]) -> bool:
    if fact.get("kind") != "control":
        return False
    raw_data = fact.get("data")
    if not isinstance(cast(object, raw_data), Mapping):
        return False
    data = cast(Mapping[object, object], raw_data)
    return data.get("action") == "limited" and data.get("reason") == "deadline"


def _tool_fact_fields(
    entry: TraceEntry,
    fact: Mapping[str, Any],
) -> tuple[str, bool, tuple[str, ...], tuple[str, ...]]:
    data = _mapping(entry, _required(entry, fact, "data"), "tool fact data")
    batch_id = _string(
        entry,
        _required(entry, data, "batch_id"),
        "tool fact batch_id",
        non_empty=True,
    )
    parallel = _boolean(entry, _required(entry, data, "parallel"), "tool fact parallel")
    call_ids = _string_sequence(
        entry,
        _required(entry, data, "call_ids"),
        "tool fact call_ids",
    )
    outcomes = _string_sequence(
        entry,
        _required(entry, data, "outcome_kinds"),
        "tool fact outcome_kinds",
    )
    if not call_ids or len(call_ids) != len(outcomes) or len(call_ids) != len(set(call_ids)):
        _fail(
            entry,
            "tool_fact_mismatch",
            "tool fact calls and outcomes must be non-empty, aligned, and unique",
        )
    return batch_id, parallel, call_ids, outcomes


def verify_trace(trace: RunTrace) -> TraceVerification:
    """Verify ordering and compact durable evidence without performing I/O."""

    if not isinstance(cast(object, trace), RunTrace):
        raise TypeError("trace must be RunTrace")
    return _Verifier(trace).run()


def _required(entry: TraceEntry, data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        _fail(entry, "invalid_entry", f"missing {key}")
    return data[key]


def _mapping(entry: TraceEntry, value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(entry, "invalid_entry", f"{label} must be an object")
    if any(not isinstance(key, str) for key in cast(Mapping[object, object], value)):
        _fail(entry, "invalid_entry", f"{label} keys must be strings")
    return cast(Mapping[str, Any], value)


def _optional_mapping(
    entry: TraceEntry,
    value: object,
    label: str,
) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _mapping(entry, value, label)


def _string(
    entry: TraceEntry,
    value: object,
    label: str,
    *,
    non_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail(entry, "invalid_entry", f"{label} must be a string")
    if non_empty and not value:
        _fail(entry, "invalid_entry", f"{label} must not be empty")
    return value


def _optional_string(entry: TraceEntry, value: object, label: str) -> str | None:
    if value is None:
        return None
    return _string(entry, value, label, non_empty=True)


def _integer(
    entry: TraceEntry,
    value: object,
    label: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(entry, "invalid_entry", f"{label} must be an integer")
    if value < minimum:
        _fail(entry, "invalid_entry", f"{label} must be >= {minimum}")
    return value


def _boolean(entry: TraceEntry, value: object, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(entry, "invalid_entry", f"{label} must be boolean")
    return value


def _string_sequence(entry: TraceEntry, value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        _fail(entry, "invalid_entry", f"{label} must be an array")
    items = tuple(cast(Sequence[object], value))
    if any(not isinstance(item, str) for item in items):
        _fail(entry, "invalid_entry", f"{label} must contain strings")
    return cast(tuple[str, ...], items)


def _digest_hex(entry: TraceEntry, value: object, label: str) -> bytes:
    digest = _string(entry, value, label)
    try:
        decoded = bytes.fromhex(digest)
    except ValueError:
        _fail(entry, "invalid_entry", f"{label} must contain 32-byte hex")
    if len(digest) != 64 or len(decoded) != 32 or decoded.hex() != digest:
        _fail(entry, "invalid_entry", f"{label} must contain 32-byte hex")
    return decoded


def _validate_selected_envelope(
    entry: TraceEntry,
    selected: _BatchSelection,
    call_id: str,
    *,
    approval: bool,
) -> None:
    code = "approval_lifecycle" if approval else "tool_lifecycle"
    batch_id = _string(
        entry,
        _required(entry, entry.data, "batch_id"),
        "selected batch_id",
        non_empty=True,
    )
    index = _integer(
        entry,
        _required(entry, entry.data, "index"),
        "selected call index",
        minimum=0,
    )
    if (
        batch_id != selected.batch_id
        or index >= len(selected.call_ids)
        or selected.call_ids[index] != call_id
    ):
        _fail(entry, code, "call does not match the selected batch")
    if not approval:
        parallel = _boolean(
            entry,
            _required(entry, entry.data, "parallel"),
            "tool parallel",
        )
        if parallel is not selected.parallel:
            _fail(entry, code, "tool parallel does not match the selected batch")


def _validate_view(entry: TraceEntry, view: RunView) -> None:
    _integer(entry, _required(entry, view, "revision"), "view revision", minimum=0)
    _integer(entry, _required(entry, view, "history_count"), "history_count", minimum=1)
    metrics = _mapping(entry, _required(entry, view, "metrics"), "view metrics")
    _integer(
        entry,
        _required(entry, metrics, "planning_steps"),
        "planning_steps",
        minimum=0,
    )
    _integer(entry, _required(entry, metrics, "tool_calls"), "tool_calls", minimum=0)
    _mapping(entry, _required(entry, metrics, "usage"), "view usage")
    _state_kind(entry, view)


def _state_kind(entry: TraceEntry, view: RunView) -> str:
    state = _mapping(entry, _required(entry, view, "state"), "view state")
    kind = _string(entry, _required(entry, state, "kind"), "state kind")
    if kind not in _ACTIVE_STATES | _CLOSED_STATES:
        _fail(entry, "invalid_entry", "state kind is unsupported")
    if kind == "tools_pending":
        _integer(
            entry,
            _required(entry, state, "pending_count"),
            "pending_count",
            minimum=1,
        )
        _digest_hex(
            entry,
            _required(entry, state, "call_id_digest"),
            "call_id_digest",
        )
    return kind


def _fail(entry: TraceEntry, code: str, message: str) -> NoReturn:
    raise TraceError(code, message, sequence=entry.sequence)


__all__ = ["TraceError", "TraceVerification", "verify_trace"]
