"""Explicit codecs for checkpoints, facts, and compact run views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jharness.kernel._engine.verification import fact_data
from jharness.kernel.checkpoint import (
    Checkpoint,
    ControlFact,
    ConversationInsertFact,
    Fact,
    FailedControl,
    HistoryRewriteFact,
    LimitedControl,
    ModelTurnFact,
    ModelTurnResult,
    ResumedFact,
    StartedFact,
    SuspendedControl,
    SuspensionView,
    ToolBatchFact,
    ToolOutcomeKind,
)
from jharness.kernel.errors import ProtocolError
from jharness.kernel.limits import LimitReason
from jharness.kernel.wire._helpers import (
    array,
    boolean,
    decode_document,
    enum_string,
    integer,
    json_object,
    number,
    object_fields,
    optional_string,
    string,
    thaw_object,
    unique_strings,
)
from jharness.kernel.wire.models import decode_model_usage_value
from jharness.kernel.wire.snapshot import decode_snapshot_value, encode_snapshot
from jharness.kernel.wire.state import decode_metrics_value, encode_metrics

__all__ = [
    "decode_checkpoint",
    "decode_fact",
    "decode_run_view",
    "encode_checkpoint",
    "encode_fact",
    "encode_run_view",
]

_FACT_KINDS = frozenset(
    {
        "started",
        "resumed",
        "model_turn",
        "tool_batch",
        "conversation_insert",
        "history_rewrite",
        "control",
    }
)
_ROLES = frozenset({"system", "user", "assistant", "tool", "external"})
_MODEL_RESULTS = frozenset(item.value for item in ModelTurnResult)
_OUTCOME_KINDS = frozenset(item.value for item in ToolOutcomeKind)
_LIMIT_REASONS = frozenset(item.value for item in LimitReason)
_VIEW_KINDS = frozenset(
    {"planning", "tools_pending", "suspended", "completed", "failed", "limited"}
)


def encode_checkpoint(value: Checkpoint) -> dict[str, Any]:
    return {
        "schema_version": "v0",
        "id": value.id,
        "snapshot": encode_snapshot(value.snapshot),
        "fact": encode_fact(value.fact),
    }


def decode_checkpoint(value: object) -> Checkpoint:
    return decode_document(value, "checkpoint", decode_checkpoint_value)


def decode_checkpoint_value(value: object) -> Checkpoint:
    data = object_fields(
        value,
        "checkpoint",
        frozenset({"schema_version", "id", "snapshot", "fact"}),
    )
    if string(data["schema_version"], "checkpoint schema_version") != "v0":
        raise ProtocolError("checkpoint schema_version must be v0")
    return Checkpoint(
        string(data["id"], "checkpoint id", non_empty=True),
        decode_snapshot_value(data["snapshot"]),
        decode_fact_value(data["fact"]),
    )


def encode_fact(value: Fact) -> dict[str, Any]:
    return fact_data(value)


def decode_fact(value: object) -> Fact:
    return decode_document(value, "checkpoint fact", decode_fact_value)


def decode_fact_value(value: object) -> Fact:
    raw = object_fields(value, "checkpoint fact", frozenset({"kind", "at", "data"}))
    kind = enum_string(raw["kind"], "fact kind", _FACT_KINDS)
    at = number(raw["at"], "fact at", minimum=0)
    data = raw["data"]
    if kind == "started":
        return _decode_started(at, data)
    if kind == "resumed":
        return _decode_resumed(at, data)
    if kind == "model_turn":
        return _decode_model_turn(at, data)
    if kind == "tool_batch":
        return _decode_tool_batch(at, data)
    if kind == "conversation_insert":
        fields = object_fields(data, "conversation insert fact", frozenset({"source"}))
        return ConversationInsertFact(
            at,
            string(fields["source"], "conversation insert source", non_empty=True),
        )
    if kind == "history_rewrite":
        return _decode_history_rewrite(at, data)
    return ControlFact(at, _decode_control(data))


def _decode_started(at: float, value: object) -> StartedFact:
    data = object_fields(value, "started fact", frozenset({"history_roles"}))
    roles = _decode_roles(data["history_roles"], "started history_roles")
    if not roles:
        raise ProtocolError("started history_roles must not be empty")
    return StartedFact(at, roles)


def _decode_resumed(at: float, value: object) -> ResumedFact:
    data = object_fields(
        value,
        "resumed fact",
        frozenset({"appended_roles", "metadata_keys"}),
    )
    return ResumedFact(
        at,
        _decode_roles(data["appended_roles"], "resumed appended_roles"),
        unique_strings(data["metadata_keys"], "resumed metadata_keys"),
    )


def _decode_model_turn(at: float, value: object) -> ModelTurnFact:
    data = object_fields(
        value,
        "model turn fact",
        frozenset(
            {
                "result",
                "part_count",
                "tool_call_ids",
                "finish_reason",
                "usage",
                "limit_reason",
            }
        ),
    )
    raw_usage = data["usage"]
    raw_limit = data["limit_reason"]
    return ModelTurnFact(
        at=at,
        result=ModelTurnResult(enum_string(data["result"], "model turn result", _MODEL_RESULTS)),
        part_count=integer(data["part_count"], "model turn part_count", minimum=0),
        tool_call_ids=unique_strings(
            data["tool_call_ids"],
            "model turn tool_call_ids",
            non_empty_items=True,
        ),
        finish_reason=optional_string(data["finish_reason"], "model turn finish_reason"),
        usage=None if raw_usage is None else decode_model_usage_value(raw_usage),
        limit_reason=(
            None
            if raw_limit is None
            else LimitReason(enum_string(raw_limit, "model turn limit_reason", _LIMIT_REASONS))
        ),
    )


def _decode_tool_batch(at: float, value: object) -> ToolBatchFact:
    data = object_fields(
        value,
        "tool batch fact",
        frozenset({"batch_id", "call_ids", "parallel", "outcome_kinds", "suspension"}),
    )
    outcomes = tuple(
        ToolOutcomeKind(enum_string(item, "tool outcome kind", _OUTCOME_KINDS))
        for item in array(data["outcome_kinds"], "tool batch outcome_kinds")
    )
    if not outcomes:
        raise ProtocolError("tool batch outcome_kinds must not be empty")
    raw_suspension = data["suspension"]
    return ToolBatchFact(
        at=at,
        batch_id=string(data["batch_id"], "tool batch id", non_empty=True),
        call_ids=_non_empty_unique_ids(data["call_ids"], "tool batch call_ids"),
        parallel=boolean(data["parallel"], "tool batch parallel"),
        outcome_kinds=outcomes,
        suspension=(None if raw_suspension is None else _decode_suspension_view(raw_suspension)),
    )


def _decode_history_rewrite(at: float, value: object) -> HistoryRewriteFact:
    data = object_fields(
        value,
        "history rewrite fact",
        frozenset({"before_count", "after_roles", "reason", "metadata_keys"}),
    )
    roles = _decode_roles(data["after_roles"], "history rewrite after_roles")
    if not roles:
        raise ProtocolError("history rewrite after_roles must not be empty")
    return HistoryRewriteFact(
        at=at,
        before_count=integer(data["before_count"], "history rewrite before_count", minimum=1),
        after_roles=roles,
        reason=string(data["reason"], "history rewrite reason", non_empty=True),
        metadata_keys=unique_strings(
            data["metadata_keys"],
            "history rewrite metadata_keys",
        ),
    )


def _decode_control(value: object) -> SuspendedControl | FailedControl | LimitedControl:
    raw = json_object(value, "control fact")
    if "action" not in raw:
        raise ProtocolError("control fact is missing field(s): action")
    action = enum_string(
        raw["action"], "control action", frozenset({"suspended", "failed", "limited"})
    )
    if action == "suspended":
        data = object_fields(
            raw,
            "suspended control",
            frozenset({"action", "reason", "source", "wait_id", "metadata_keys"}),
        )
        return SuspendedControl(
            string(data["reason"], "control reason", non_empty=True),
            string(data["source"], "control source", non_empty=True),
            optional_string(data["wait_id"], "control wait_id", non_empty=True),
            unique_strings(data["metadata_keys"], "control metadata_keys"),
        )
    if action == "failed":
        data = object_fields(raw, "failed control", frozenset({"action", "code"}))
        return FailedControl(string(data["code"], "control failure code", non_empty=True))
    data = object_fields(raw, "limited control", frozenset({"action", "reason"}))
    return LimitedControl(
        LimitReason(enum_string(data["reason"], "control limit reason", _LIMIT_REASONS))
    )


def _decode_suspension_view(value: object) -> SuspensionView:
    data = object_fields(
        value,
        "suspension view",
        frozenset({"reason", "source", "wait_id", "metadata_keys"}),
    )
    return SuspensionView(
        string(data["reason"], "suspension reason", non_empty=True),
        string(data["source"], "suspension source", non_empty=True),
        optional_string(data["wait_id"], "suspension wait_id", non_empty=True),
        unique_strings(data["metadata_keys"], "suspension metadata_keys"),
    )


def _decode_roles(value: object, label: str) -> tuple[str, ...]:
    return tuple(enum_string(item, f"{label} item", _ROLES) for item in array(value, label))


def _non_empty_unique_ids(value: object, label: str) -> tuple[str, ...]:
    result = unique_strings(value, label, non_empty_items=True)
    if not result:
        raise ProtocolError(f"{label} must not be empty")
    return result


def encode_run_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one trusted compact run view into canonical JSON containers."""

    return thaw_object(value)


def decode_run_view(value: object) -> dict[str, Any]:
    return decode_document(value, "run view", decode_run_view_value)


def decode_run_view_value(value: object) -> dict[str, Any]:
    data = object_fields(
        value,
        "run view",
        frozenset({"revision", "history_count", "metrics", "state"}),
    )
    return {
        "revision": integer(data["revision"], "run view revision", minimum=0),
        "history_count": integer(data["history_count"], "run view history_count", minimum=1),
        "metrics": encode_metrics(decode_metrics_value(data["metrics"])),
        "state": _decode_state_view(data["state"]),
    }


def _decode_state_view(value: object, *, active: bool = False) -> dict[str, Any]:
    raw = json_object(value, "state view")
    if "kind" not in raw:
        raise ProtocolError("state view is missing field(s): kind")
    kind = enum_string(raw["kind"], "state view kind", _VIEW_KINDS)
    if active and kind not in {"planning", "tools_pending"}:
        raise ProtocolError("active state view must be planning or tools_pending")
    if kind == "planning":
        object_fields(raw, "planning view", frozenset({"kind"}))
        return {"kind": "planning"}
    if kind == "tools_pending":
        data = object_fields(raw, "tools pending view", frozenset({"kind", "call_ids"}))
        return {
            "kind": "tools_pending",
            "call_ids": list(_non_empty_unique_ids(data["call_ids"], "view call_ids")),
        }
    if kind == "suspended":
        data = object_fields(
            raw,
            "suspended view",
            frozenset({"kind", "resume_to", "suspension"}),
        )
        return {
            "kind": "suspended",
            "resume_to": _decode_state_view(data["resume_to"], active=True),
            "suspension": _encode_suspension_view(_decode_suspension_view(data["suspension"])),
        }
    if kind == "completed":
        data = object_fields(raw, "completed view", frozenset({"kind", "part_count"}))
        return {
            "kind": "completed",
            "part_count": integer(data["part_count"], "view part_count", minimum=1),
        }
    if kind == "failed":
        data = object_fields(raw, "failed view", frozenset({"kind", "code"}))
        return {"kind": "failed", "code": string(data["code"], "view code", non_empty=True)}
    data = object_fields(raw, "limited view", frozenset({"kind", "reason"}))
    return {
        "kind": "limited",
        "reason": enum_string(data["reason"], "view limit reason", _LIMIT_REASONS),
    }


def _encode_suspension_view(value: SuspensionView) -> dict[str, Any]:
    return {
        "reason": value.reason,
        "source": value.source,
        "wait_id": value.wait_id,
        "metadata_keys": list(value.metadata_keys),
    }
