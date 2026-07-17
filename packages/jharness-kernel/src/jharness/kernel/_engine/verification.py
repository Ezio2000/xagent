"""Shared compact checkpoint projections and deterministic fact verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel.checkpoint import (
    ControlFact,
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
from jharness.kernel.models import ModelUsage
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import Completed, Failed, Limited, Planning, Suspended, ToolsPending

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def usage_data(usage: ModelUsage) -> dict[str, int | None]:
    return {name: cast(int | None, getattr(usage, name)) for name in _USAGE_FIELDS}


def suspension_data(view: SuspensionView) -> dict[str, Any]:
    return {
        "reason": view.reason,
        "source": view.source,
        "wait_id": view.wait_id,
        "metadata_keys": list(view.metadata_keys),
    }


def fact_data(fact: Fact) -> dict[str, Any]:
    """Project one semantic Fact into its canonical portable mapping."""

    if isinstance(fact, StartedFact):
        data: dict[str, Any] = {"history_roles": list(fact.history_roles)}
    elif isinstance(fact, ResumedFact):
        data = {
            "appended_roles": list(fact.appended_roles),
            "metadata_keys": list(fact.metadata_keys),
        }
    elif isinstance(fact, ModelTurnFact):
        data = {
            "result": fact.result.value,
            "part_count": fact.part_count,
            "tool_call_ids": list(fact.tool_call_ids),
            "finish_reason": fact.finish_reason,
            "usage": None if fact.usage is None else usage_data(fact.usage),
            "limit_reason": None if fact.limit_reason is None else fact.limit_reason.value,
        }
    elif isinstance(fact, ToolBatchFact):
        data = {
            "batch_id": fact.batch_id,
            "call_ids": list(fact.call_ids),
            "parallel": fact.parallel,
            "outcome_kinds": [item.value for item in fact.outcome_kinds],
            "suspension": (None if fact.suspension is None else suspension_data(fact.suspension)),
        }
    elif isinstance(fact, ConversationInsertFact):
        data = {"source": fact.source}
    elif isinstance(fact, HistoryRewriteFact):
        data = {
            "before_count": fact.before_count,
            "after_roles": list(fact.after_roles),
            "reason": fact.reason,
            "metadata_keys": list(fact.metadata_keys),
        }
    else:
        data = _control_data(fact)
    return {"kind": fact.kind, "at": fact.at, "data": data}


def _control_data(fact: ControlFact) -> dict[str, Any]:
    decision = fact.decision
    if isinstance(decision, SuspendedControl):
        return {
            "action": "suspended",
            "reason": decision.reason,
            "source": decision.source,
            "wait_id": decision.wait_id,
            "metadata_keys": list(decision.metadata_keys),
        }
    if isinstance(decision, FailedControl):
        return {"action": "failed", "code": decision.code}
    return {"action": "limited", "reason": decision.reason.value}


def run_view(snapshot: RunSnapshot) -> dict[str, Any]:
    return {
        "revision": snapshot.revision,
        "history_count": len(snapshot.history),
        "metrics": {
            "planning_steps": snapshot.metrics.planning_steps,
            "tool_calls": snapshot.metrics.tool_calls,
            "usage": usage_data(snapshot.metrics.usage),
        },
        "state": state_view(snapshot.state),
    }


def state_view(state: object) -> dict[str, Any]:
    if isinstance(state, Planning):
        return {"kind": "planning"}
    if isinstance(state, ToolsPending):
        return {"kind": "tools_pending", "call_ids": [call.id for call in state.pending]}
    if isinstance(state, Suspended):
        suspension = state.suspension
        return {
            "kind": "suspended",
            "resume_to": state_view(state.resume_to),
            "suspension": {
                "reason": suspension.reason,
                "source": suspension.source,
                "wait_id": suspension.wait_id,
                "metadata_keys": sorted(suspension.metadata),
            },
        }
    if isinstance(state, Completed):
        return {"kind": "completed", "part_count": len(state.parts)}
    if isinstance(state, Failed):
        return {"kind": "failed", "code": state.error.code}
    if isinstance(state, Limited):
        return {"kind": "limited", "reason": state.reason.value}
    raise TypeError("state must be a RunState")


def verify_change(
    before: Mapping[str, Any] | None,
    fact: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """Verify one compact durable transition without replaying effects."""

    expected = _advance_view(before, fact)
    if expected["revision"] != after.get("revision"):
        raise ValueError("revision_gap")
    if expected != dict(after):
        raise ValueError("change_mismatch")


def _advance_view(before: Mapping[str, Any] | None, fact: Mapping[str, Any]) -> dict[str, Any]:
    kind = fact["kind"]
    data = cast(Mapping[str, Any], fact["data"])
    if kind == "started":
        if before is not None:
            raise ValueError("started_requires_empty_before")
        return _started(data)
    if before is None:
        raise ValueError("missing_before_view")
    base = _base(before)
    handlers = {
        "resumed": _resumed,
        "model_turn": _model_turn,
        "tool_batch": _tool_batch,
        "conversation_insert": _conversation_insert,
        "history_rewrite": _history_rewrite,
        "control": _control,
    }
    try:
        handler = handlers[cast(str, kind)]
    except KeyError as exc:
        raise ValueError("unsupported_fact") from exc
    handler(base, before, data)
    return base


def _started(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "revision": 0,
        "history_count": len(cast(Sequence[object], data["history_roles"])),
        "metrics": {
            "planning_steps": 0,
            "tool_calls": 0,
            "usage": {name: None for name in _USAGE_FIELDS},
        },
        "state": {"kind": "planning"},
    }


def _base(before: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cast(Mapping[str, Any], before["metrics"])
    return {
        "revision": cast(int, before["revision"]) + 1,
        "history_count": before["history_count"],
        "metrics": {
            "planning_steps": metrics["planning_steps"],
            "tool_calls": metrics["tool_calls"],
            "usage": dict(cast(Mapping[str, Any], metrics["usage"])),
        },
        "state": dict(cast(Mapping[str, Any], before["state"])),
    }


def _resumed(result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    state = cast(Mapping[str, Any], before["state"])
    if state.get("kind") != "suspended":
        raise ValueError("resume_requires_suspended")
    result["history_count"] = cast(int, result["history_count"]) + len(
        cast(Sequence[object], data["appended_roles"])
    )
    result["state"] = dict(cast(Mapping[str, Any], state["resume_to"]))


def _model_turn(result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    if cast(Mapping[str, Any], before["state"]).get("kind") != "planning":
        raise ValueError("model_turn_requires_planning")
    result["history_count"] = cast(int, result["history_count"]) + 1
    metrics = cast(dict[str, Any], result["metrics"])
    metrics["planning_steps"] = cast(int, metrics["planning_steps"]) + 1
    metrics["usage"] = _add_usage(
        cast(Mapping[str, Any], metrics["usage"]),
        cast(Mapping[str, Any] | None, data["usage"]),
    )
    outcome = data["result"]
    if outcome == "completed":
        result["state"] = {"kind": "completed", "part_count": data["part_count"]}
    elif outcome == "tools_pending":
        result["state"] = {"kind": "tools_pending", "call_ids": data["tool_call_ids"]}
    else:
        result["state"] = {"kind": "limited", "reason": data["limit_reason"]}


def _tool_batch(result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    state = cast(Mapping[str, Any], before["state"])
    if state.get("kind") != "tools_pending":
        raise ValueError("tool_batch_requires_pending")
    pending = list(cast(Sequence[str], state["call_ids"]))
    calls = list(cast(Sequence[str], data["call_ids"]))
    if pending[: len(calls)] != calls:
        raise ValueError("tool_batch_not_prefix")
    result["history_count"] = cast(int, result["history_count"]) + len(calls)
    metrics = cast(dict[str, Any], result["metrics"])
    metrics["tool_calls"] = cast(int, metrics["tool_calls"]) + len(calls)
    remaining = pending[len(calls) :]
    active: dict[str, Any] = (
        {"kind": "tools_pending", "call_ids": remaining} if remaining else {"kind": "planning"}
    )
    suspension = data["suspension"]
    result["state"] = (
        active
        if suspension is None
        else {"kind": "suspended", "resume_to": active, "suspension": suspension}
    )


def _conversation_insert(
    result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]
) -> None:
    del data
    if cast(Mapping[str, Any], before["state"]).get("kind") != "planning":
        raise ValueError("insert_requires_planning")
    result["history_count"] = cast(int, result["history_count"]) + 1
    result["state"] = {"kind": "planning"}


def _history_rewrite(
    result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]
) -> None:
    if cast(Mapping[str, Any], before["state"]).get("kind") != "planning":
        raise ValueError("rewrite_requires_planning")
    if data["before_count"] != before["history_count"]:
        raise ValueError("rewrite_before_mismatch")
    result["history_count"] = len(cast(Sequence[object], data["after_roles"]))
    result["state"] = {"kind": "planning"}


def _control(result: dict[str, Any], before: Mapping[str, Any], data: Mapping[str, Any]) -> None:
    action = data["action"]
    if action == "failed":
        result["state"] = {"kind": "failed", "code": data["code"]}
    elif action == "limited":
        result["state"] = {"kind": "limited", "reason": data["reason"]}
    else:
        state = cast(Mapping[str, Any], before["state"])
        if state.get("kind") not in {"planning", "tools_pending"}:
            raise ValueError("suspend_requires_active")
        result["state"] = {
            "kind": "suspended",
            "resume_to": dict(state),
            "suspension": {
                "reason": data["reason"],
                "source": data["source"],
                "wait_id": data["wait_id"],
                "metadata_keys": data["metadata_keys"],
            },
        }


def _add_usage(before: Mapping[str, Any], delta: Mapping[str, Any] | None) -> dict[str, int | None]:
    if delta is None:
        return {name: cast(int | None, before[name]) for name in _USAGE_FIELDS}
    return {
        name: (
            cast(int | None, before[name])
            if delta[name] is None
            else (cast(int | None, before[name]) or 0) + cast(int, delta[name])
        )
        for name in _USAGE_FIELDS
    }
