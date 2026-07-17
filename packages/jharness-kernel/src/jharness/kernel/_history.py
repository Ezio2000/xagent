"""Private incremental history validation and proof construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jharness.kernel._digest import append_history_digest, empty_history_digest
from jharness.kernel._validation import expect_instance_tuple
from jharness.kernel.messages import Message, ToolCall
from jharness.kernel.state import Failed, Limited, RunState, Suspended, ToolsPending


@dataclass(frozen=True, slots=True)
class HistoryProof:
    """Unexported evidence retained only by a validated snapshot."""

    digest: bytes
    seen_call_ids: frozenset[str]
    unresolved: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if len(self.digest) != 32:
            raise ValueError("history proof digest must contain 32 bytes")


_EMPTY_HISTORY_DIGEST = empty_history_digest()
_EMPTY_HISTORY_PROOF = HistoryProof(_EMPTY_HISTORY_DIGEST, frozenset(), ())


def analyze_history(
    history: Sequence[Message],
    state: RunState,
    *,
    label: str = "run history",
    empty_message: str = "run history must not be empty",
) -> tuple[tuple[Message, ...], HistoryProof]:
    """Normalize and validate an external history, returning its private proof."""

    messages = normalize_history(history, label=label, empty_message=empty_message)
    return messages, analyze_messages(messages, state)


def normalize_history(
    history: Sequence[Message], *, label: str, empty_message: str
) -> tuple[Message, ...]:
    messages = expect_instance_tuple(history, Message, label)
    if not messages:
        raise ValueError(empty_message)
    return messages


def analyze_messages(messages: tuple[Message, ...], state: RunState) -> HistoryProof:
    proof = _extend_proof(_EMPTY_HISTORY_PROOF, messages)
    _validate_unresolved(proof.unresolved, state)
    return proof


def evolve_history(
    history: tuple[Message, ...],
    proof: HistoryProof,
    *,
    append: tuple[Message, ...],
    replace: tuple[Message, ...] | None,
    state: RunState,
) -> tuple[tuple[Message, ...], HistoryProof]:
    """Validate only one trusted edit while retaining full-history evidence."""

    if replace is not None:
        return replace, analyze_messages(replace, state)
    if not append:
        _validate_unresolved(proof.unresolved, state)
        return history, proof
    next_proof = _extend_proof(proof, append)
    _validate_unresolved(next_proof.unresolved, state)
    return (*history, *append), next_proof


def _extend_proof(proof: HistoryProof, messages: tuple[Message, ...]) -> HistoryProof:
    seen = proof.seen_call_ids
    unresolved = proof.unresolved
    digest = proof.digest
    for message in messages:
        unresolved, seen = _advance_linkage(message, unresolved, seen)
        digest = append_history_digest(digest, message)
    return HistoryProof(digest, seen, unresolved)


def _advance_linkage(
    message: Message,
    unresolved: tuple[ToolCall, ...],
    seen: frozenset[str],
) -> tuple[tuple[ToolCall, ...], frozenset[str]]:
    if unresolved:
        if message.role != "tool":
            raise ValueError("unresolved tool request must be at the end of history")
        expected = unresolved[0].id
        if message.tool_call_id != expected:
            raise ValueError(
                f"tool result order mismatch: expected {expected!r}, found {message.tool_call_id!r}"
            )
        return unresolved[1:], seen
    if message.role == "tool":
        raise ValueError("tool message requires a preceding assistant tool request")
    if message.role == "assistant" and message.tool_calls:
        return message.tool_calls, _record_call_ids(message.tool_calls, seen)
    return (), seen


def _validate_unresolved(unresolved: tuple[ToolCall, ...], state: RunState) -> None:
    pending = _pending_state(state)
    if pending is not None:
        if unresolved != pending.pending:
            raise ValueError("pending state must match unresolved history tool calls")
    elif unresolved and not isinstance(state, Failed | Limited):
        raise ValueError("planning or completed state cannot retain unresolved tool calls")


def _pending_state(state: RunState) -> ToolsPending | None:
    if isinstance(state, ToolsPending):
        return state
    if isinstance(state, Suspended) and isinstance(state.resume_to, ToolsPending):
        return state.resume_to
    return None


def _record_call_ids(calls: tuple[ToolCall, ...], seen: frozenset[str]) -> frozenset[str]:
    ids = {call.id for call in calls}
    duplicates = seen.intersection(ids)
    if duplicates:
        raise ValueError(f"tool call id reused in history: {min(duplicates)}")
    return seen.union(ids)
