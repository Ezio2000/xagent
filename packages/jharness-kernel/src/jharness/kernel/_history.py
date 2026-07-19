"""Private incremental history validation and proof construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jharness.kernel._digest import append_history_digest, empty_history_digest
from jharness.kernel._validation import expect_instance_tuple
from jharness.kernel.history import RunHistory
from jharness.kernel.messages import Message, ToolCall
from jharness.kernel.state import (
    Failed,
    Limited,
    PendingToolCalls,
    RunState,
    Suspended,
    ToolsPending,
)


@dataclass(frozen=True, slots=True)
class _IdNode:
    terminal: bool = False
    children: tuple[tuple[int, _IdNode], ...] = ()


@dataclass(frozen=True, slots=True)
class _PersistentIdSet:
    root: _IdNode = _IdNode()

    def __contains__(self, key: str) -> bool:
        node = self.root
        for edge in key.encode("utf-8"):
            child = _child(node, edge)
            if child is None:
                return False
            node = child
        return node.terminal

    def add(self, key: str) -> _PersistentIdSet:
        encoded = key.encode("utf-8")
        node = self.root
        path: list[tuple[_IdNode, int]] = []
        missing_at: int | None = None
        for index, edge in enumerate(encoded):
            path.append((node, edge))
            child = _child(node, edge)
            if child is None:
                missing_at = index
                break
            node = child
        if missing_at is None:
            if node.terminal:
                return self
            updated = _IdNode(True, node.children)
        else:
            updated = _IdNode(True)
            for edge in reversed(encoded[missing_at + 1 :]):
                updated = _IdNode(False, ((edge, updated),))
        for parent, edge in reversed(path):
            updated = _with_child(parent, edge, updated)
        return _PersistentIdSet(updated)


@dataclass(frozen=True, slots=True)
class HistoryProof:
    """Unexported evidence retained only by a validated snapshot."""

    digest: bytes
    seen_call_ids: _PersistentIdSet
    unresolved: PendingToolCalls | None

    def __post_init__(self) -> None:
        if len(self.digest) != 32:
            raise ValueError("history proof digest must contain 32 bytes")


_EMPTY_HISTORY_DIGEST = empty_history_digest()
_EMPTY_HISTORY_PROOF = HistoryProof(_EMPTY_HISTORY_DIGEST, _PersistentIdSet(), None)


def analyze_history(
    history: Sequence[Message],
    state: RunState,
    *,
    label: str = "run history",
    empty_message: str = "run history must not be empty",
) -> tuple[RunHistory, HistoryProof]:
    """Normalize and validate an external history, returning its private proof."""

    messages = normalize_history(history, label=label, empty_message=empty_message)
    return messages, analyze_messages(messages, state)


def normalize_history(history: Sequence[Message], *, label: str, empty_message: str) -> RunHistory:
    if isinstance(history, RunHistory):
        return history
    messages = expect_instance_tuple(history, Message, label)
    if not messages:
        raise ValueError(empty_message)
    return RunHistory(messages)


def analyze_messages(messages: RunHistory, state: RunState) -> HistoryProof:
    proof = _extend_proof(_EMPTY_HISTORY_PROOF, messages)
    unresolved = _validate_unresolved(proof.unresolved, state)
    if unresolved is proof.unresolved:
        return proof
    return HistoryProof(proof.digest, proof.seen_call_ids, unresolved)


def evolve_history(
    history: RunHistory,
    proof: HistoryProof,
    *,
    append: tuple[Message, ...],
    replace: RunHistory | None,
    state: RunState,
) -> tuple[RunHistory, HistoryProof]:
    """Validate only one trusted edit while retaining full-history evidence."""

    if replace is not None:
        return replace, analyze_messages(replace, state)
    if not append:
        unresolved = _validate_unresolved(proof.unresolved, state)
        if unresolved is proof.unresolved:
            return history, proof
        return history, HistoryProof(proof.digest, proof.seen_call_ids, unresolved)
    next_proof = _extend_proof(proof, append)
    unresolved = _validate_unresolved(next_proof.unresolved, state)
    if unresolved is not next_proof.unresolved:
        next_proof = HistoryProof(next_proof.digest, next_proof.seen_call_ids, unresolved)
    return history._append(  # pyright: ignore[reportPrivateUsage]
        append,
        digest=next_proof.digest,
    ), next_proof


def _extend_proof(proof: HistoryProof, messages: Sequence[Message]) -> HistoryProof:
    seen = proof.seen_call_ids
    unresolved = proof.unresolved
    digest = proof.digest
    for message in messages:
        unresolved, seen = _advance_linkage(message, unresolved, seen)
        digest = append_history_digest(digest, message)
    return HistoryProof(digest, seen, unresolved)


def _advance_linkage(
    message: Message,
    unresolved: PendingToolCalls | None,
    seen: _PersistentIdSet,
) -> tuple[PendingToolCalls | None, _PersistentIdSet]:
    if unresolved is not None:
        if message.role != "tool":
            raise ValueError("unresolved tool request must be at the end of history")
        expected = unresolved[0].id
        if message.tool_call_id != expected:
            raise ValueError(
                f"tool result order mismatch: expected {expected!r}, found {message.tool_call_id!r}"
            )
        return unresolved.advance(1), seen
    if message.role == "tool":
        raise ValueError("tool message requires a preceding assistant tool request")
    if message.role == "assistant" and message.tool_calls:
        return PendingToolCalls(message.tool_calls), _record_call_ids(message.tool_calls, seen)
    return None, seen


def _validate_unresolved(
    unresolved: PendingToolCalls | None,
    state: RunState,
) -> PendingToolCalls | None:
    pending = _pending_state(state)
    if pending is not None:
        if unresolved is None or unresolved != pending.pending:
            raise ValueError("pending state must match unresolved history tool calls")
        # External recovery requires one exact comparison. Canonicalizing the proof
        # onto the state's cursor makes every later paired advance identity-fast.
        return pending.pending
    elif unresolved is not None and not isinstance(state, Failed | Limited):
        raise ValueError("planning or completed state cannot retain unresolved tool calls")
    return unresolved


def _pending_state(state: RunState) -> ToolsPending | None:
    if isinstance(state, ToolsPending):
        return state
    if isinstance(state, Suspended) and isinstance(state.resume_to, ToolsPending):
        return state.resume_to
    return None


def _record_call_ids(calls: tuple[ToolCall, ...], seen: _PersistentIdSet) -> _PersistentIdSet:
    updated = seen
    for call in calls:
        if call.id in updated:
            raise ValueError(f"tool call id reused in history: {call.id}")
        updated = updated.add(call.id)
    return updated


def _child(node: _IdNode, edge: int) -> _IdNode | None:
    for candidate, child in node.children:
        if candidate == edge:
            return child
        if candidate > edge:
            break
    return None


def _with_child(node: _IdNode, edge: int, child: _IdNode) -> _IdNode:
    children = list(node.children)
    for index, (candidate, _) in enumerate(children):
        if candidate == edge:
            children[index] = (edge, child)
            break
        if candidate > edge:
            children.insert(index, (edge, child))
            break
    else:
        children.append((edge, child))
    return _IdNode(node.terminal, tuple(children))
