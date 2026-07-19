"""Explicit incremental storage codec shared by repository adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from jharness.kernel import (
    Checkpoint,
    DurableCommit,
    HistoryAppend,
    HistoryReplace,
    HistoryUnchanged,
    Message,
    PendingToolCalls,
    Planning,
    ProtocolError,
    RepositoryError,
    RunHistory,
    RunSnapshot,
    RunState,
    Suspended,
    ToolCall,
    ToolsPending,
    checkpoint_digest,
)
from jharness.kernel.wire import (
    decode_context,
    decode_fact,
    decode_message,
    decode_metrics,
    decode_state,
    decode_suspension,
    encode_context,
    encode_fact,
    encode_message,
    encode_metrics,
    encode_state,
    encode_suspension,
)

HISTORY_CHUNK_SIZE = 64
_CORE_FIELDS = {
    "storage_version",
    "id",
    "parent_checkpoint_id",
    "revision",
    "context",
    "metrics",
    "state",
    "fact",
    "history_count",
    "history_digest",
}


@dataclass(frozen=True, slots=True)
class CommitIdentity:
    """Normalized scalar fields needed before any payload is encoded."""

    commit: DurableCommit
    checkpoint_id: str
    run_id: str
    parent_checkpoint_id: str | None
    revision: int
    expected_revision: int | None
    digest: bytes
    history_count: int
    history_digest: bytes
    base_history_count: int | None
    base_history_digest: bytes | None


@dataclass(frozen=True, slots=True)
class EncodedCore:
    """Canonical bytes for checkpoint data other than message history."""

    payload: bytes
    digest: bytes


@dataclass(frozen=True, slots=True)
class EncodedHistoryChunk:
    """One bounded canonical message-history chunk."""

    message_count: int
    payload: bytes
    digest: bytes


@dataclass(frozen=True, slots=True)
class DecodedCore:
    """Decoded checkpoint core awaiting its separately stored history."""

    checkpoint_id: str
    parent_checkpoint_id: str | None
    revision: int
    context: object
    metrics: object
    state: object
    fact: object
    history_count: int
    history_digest: bytes

    def checkpoint(self, history: RunHistory) -> Checkpoint:
        """Join this core to one decoded history."""

        if len(history) != self.history_count:
            raise RepositoryError("stored checkpoint history count is inconsistent")
        try:
            snapshot = RunSnapshot(
                self.revision,
                cast(Any, self.context),
                history,
                cast(Any, self.metrics),
                _decode_storage_state(self.state, history),
            )
            return Checkpoint(self.checkpoint_id, snapshot, cast(Any, self.fact))
        except (ProtocolError, TypeError, ValueError) as exc:
            raise RepositoryError("stored checkpoint core is invalid") from exc


def commit_identity(commit: DurableCommit) -> CommitIdentity:
    """Validate and normalize the O(1) identity/manifest portion of a commit."""

    if not isinstance(cast(object, commit), DurableCommit):
        raise TypeError("commit must be a DurableCommit")
    checkpoint_id = str.__str__(commit.checkpoint_id)
    run_id = str.__str__(commit.run_id)
    parent = commit.parent_checkpoint_id
    parent = None if parent is None else str.__str__(parent)
    if not checkpoint_id:
        raise ValueError("checkpoint id must not be empty")
    if not run_id:
        raise ValueError("run id must not be empty")
    digest = _digest_bytes(commit.digest, "commit digest")
    history_digest = _digest_bytes(commit.history_digest, "history digest")
    base_digest = commit.base_history_digest
    if base_digest is not None:
        base_digest = _digest_bytes(base_digest, "base history digest")
    return CommitIdentity(
        commit=commit,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        parent_checkpoint_id=parent,
        revision=commit.revision,
        expected_revision=commit.expected_revision,
        digest=digest,
        history_count=commit.history_count,
        history_digest=history_digest,
        base_history_count=commit.base_history_count,
        base_history_digest=base_digest,
    )


def encode_core(identity: CommitIdentity) -> EncodedCore:
    """Encode only the checkpoint fields whose size is independent of history."""

    checkpoint = identity.commit.checkpoint
    snapshot = checkpoint.snapshot
    document = {
        "storage_version": "v2",
        "id": identity.checkpoint_id,
        "parent_checkpoint_id": identity.parent_checkpoint_id,
        "revision": identity.revision,
        "context": encode_context(snapshot.context),
        "metrics": encode_metrics(snapshot.metrics),
        "state": _encode_storage_state(snapshot.state),
        "fact": encode_fact(checkpoint.fact),
        "history_count": identity.history_count,
        "history_digest": identity.history_digest.hex(),
    }
    payload = _canonical_json(document)
    return EncodedCore(payload, sha256(payload).digest())


def encode_history_change(commit: DurableCommit) -> tuple[EncodedHistoryChunk, ...]:
    """Encode only messages introduced by this history mutation."""

    change = commit.history
    if isinstance(change, HistoryUnchanged):
        return ()
    if isinstance(change, HistoryAppend):
        messages: Iterable[Message] = change.messages
    elif isinstance(change, HistoryReplace):
        messages = change.history
    else:
        messages = change.history
    return _encode_chunks(messages)


def decode_core(payload: bytes, digest: bytes) -> DecodedCore:
    """Decode and authenticate one stored checkpoint core."""

    payload = _stored_bytes(payload, "checkpoint core payload")
    digest = _stored_digest(digest, "checkpoint core digest")
    if sha256(payload).digest() != digest:
        raise RepositoryError("stored checkpoint core has an invalid digest")
    try:
        document: object = json.loads(payload)
        if type(document) is not dict:
            raise TypeError("checkpoint core must be an object")
        data = cast(dict[str, object], document)
        if set(data) != _CORE_FIELDS:
            raise ValueError("checkpoint core fields differ")
        if data["storage_version"] != "v2":
            raise ValueError("checkpoint core storage version differs")
        checkpoint_id = _non_empty_text(data["id"], "checkpoint id")
        raw_parent = data["parent_checkpoint_id"]
        parent = None if raw_parent is None else _non_empty_text(raw_parent, "parent id")
        revision = _nonnegative_integer(data["revision"], "revision")
        history_count = _positive_integer(data["history_count"], "history count")
        history_digest = _hex_digest(data["history_digest"], "history digest")
        return DecodedCore(
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=parent,
            revision=revision,
            context=decode_context(data["context"]),
            metrics=decode_metrics(data["metrics"]),
            state=data["state"],
            fact=decode_fact(data["fact"]),
            history_count=history_count,
            history_digest=history_digest,
        )
    except (
        KeyError,
        OverflowError,
        ProtocolError,
        RecursionError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise RepositoryError("stored checkpoint core is invalid") from exc


def decode_history_chunk(
    payload: bytes,
    digest: bytes,
    expected_count: int,
) -> tuple[Message, ...]:
    """Decode and authenticate one bounded history chunk."""

    payload = _stored_bytes(payload, "history chunk payload")
    digest = _stored_digest(digest, "history chunk digest")
    expected_count = _positive_integer(expected_count, "history chunk message count")
    if expected_count > HISTORY_CHUNK_SIZE:
        raise RepositoryError("stored history chunk message count is invalid")
    if sha256(payload).digest() != digest:
        raise RepositoryError("stored history chunk has an invalid digest")
    try:
        document: object = json.loads(payload)
        if type(document) is not list:
            raise ValueError("history chunk must be an array")
        items = cast(list[object], document)
        if len(items) != expected_count:
            raise ValueError("history chunk count differs")
        return tuple(decode_message(item) for item in items)
    except (
        OverflowError,
        ProtocolError,
        RecursionError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise RepositoryError("stored history chunk payload is invalid") from exc


def reconstruct_checkpoint(
    *,
    core_payload: bytes,
    core_digest: bytes,
    chunks: Sequence[tuple[bytes, bytes, int]],
    expected_checkpoint_digest: bytes,
) -> tuple[Checkpoint, DecodedCore]:
    """Materialize and cross-check one complete checkpoint from split storage."""

    core = decode_core(core_payload, core_digest)
    messages: list[Message] = []
    for payload, digest, count in chunks:
        messages.extend(decode_history_chunk(payload, digest, count))
    try:
        history = RunHistory(messages)
    except (TypeError, ValueError) as exc:
        raise RepositoryError("stored checkpoint history is invalid") from exc
    checkpoint = core.checkpoint(history)
    expected = _stored_digest(expected_checkpoint_digest, "checkpoint digest")
    if checkpoint_digest(checkpoint) != expected:
        raise RepositoryError("stored checkpoint content digest is invalid")
    return checkpoint, core


def _encode_chunks(messages: Iterable[Message]) -> tuple[EncodedHistoryChunk, ...]:
    chunks: list[EncodedHistoryChunk] = []
    pending: list[dict[str, Any]] = []
    for message in messages:
        pending.append(encode_message(message))
        if len(pending) == HISTORY_CHUNK_SIZE:
            chunks.append(_encoded_chunk(pending))
            pending = []
    if pending:
        chunks.append(_encoded_chunk(pending))
    return tuple(chunks)


def _encoded_chunk(messages: list[dict[str, Any]]) -> EncodedHistoryChunk:
    payload = _canonical_json(messages)
    return EncodedHistoryChunk(len(messages), payload, sha256(payload).digest())


def _encode_storage_state(state: RunState) -> dict[str, Any]:
    if isinstance(state, ToolsPending):
        return _encode_pending_state(state)
    if isinstance(state, Suspended):
        resume_to = state.resume_to
        return {
            "kind": "suspended",
            "resume_to": (
                _encode_pending_state(resume_to)
                if isinstance(resume_to, ToolsPending)
                else {"kind": "planning"}
            ),
            "suspension": encode_suspension(state.suspension),
        }
    return encode_state(state)


def _encode_pending_state(state: ToolsPending) -> dict[str, Any]:
    return {
        "kind": "tools_pending",
        "pending_count": state.pending.pending_count,
        "pending_digest": state.pending.digest.hex(),
    }


def _decode_storage_state(document: object, history: RunHistory) -> RunState:
    if type(document) is not dict:
        raise ProtocolError("stored run state must be an object")
    data = cast(dict[str, object], document)
    kind = data.get("kind")
    if kind == "tools_pending":
        return _decode_pending_state(data, history)
    if kind == "suspended":
        if set(data) != {"kind", "resume_to", "suspension"}:
            raise ProtocolError("stored suspended state fields differ")
        raw_resume = data["resume_to"]
        if type(raw_resume) is not dict:
            raise ProtocolError("stored suspended resume state must be an object")
        resume_data = cast(dict[str, object], raw_resume)
        resume_kind = resume_data.get("kind")
        if resume_kind == "tools_pending":
            resume_to = _decode_pending_state(resume_data, history)
        elif resume_kind == "planning" and set(resume_data) == {"kind"}:
            resume_to = Planning()
        else:
            raise ProtocolError("stored suspended resume state is invalid")
        return Suspended(resume_to, decode_suspension(data["suspension"]))
    return decode_state(cast(object, data))


def _decode_pending_state(
    data: dict[str, object],
    history: RunHistory,
) -> ToolsPending:
    if set(data) != {"kind", "pending_count", "pending_digest"}:
        raise ProtocolError("stored pending state fields differ")
    count = _positive_integer(data["pending_count"], "pending count")
    digest = _hex_digest(data["pending_digest"], "pending digest")
    calls = _unresolved_tool_calls(history)
    if not calls:
        raise ProtocolError("stored pending state has no unresolved history calls")
    pending = PendingToolCalls(calls)
    if pending.pending_count != count or pending.digest != digest:
        raise ProtocolError("stored pending state manifest is inconsistent")
    return ToolsPending(pending)


def _unresolved_tool_calls(history: RunHistory) -> tuple[ToolCall, ...]:
    pending: tuple[ToolCall, ...] = ()
    cursor = 0
    for message in history:
        if cursor < len(pending):
            if message.role != "tool" or message.tool_call_id != pending[cursor].id:
                raise ProtocolError("stored history tool linkage is invalid")
            cursor += 1
            continue
        pending = ()
        cursor = 0
        if message.role == "tool":
            raise ProtocolError("stored history tool result has no request")
        if message.role == "assistant" and message.tool_calls:
            pending = message.tool_calls
    return pending[cursor:]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _stored_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise RepositoryError(f"stored {label} is invalid")
    return value


def _stored_digest(value: object, label: str) -> bytes:
    digest = _stored_bytes(value, label)
    if len(digest) != 32:
        raise RepositoryError(f"stored {label} is invalid")
    return digest


def _digest_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be bytes")
    digest = value
    if len(digest) != 32:
        raise ValueError(f"{label} must contain 32 bytes")
    return digest


def _non_empty_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} is invalid")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} is invalid")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _nonnegative_integer(value, label)
    if result == 0:
        raise ValueError(f"{label} is invalid")
    return result


def _hex_digest(value: object, label: str) -> bytes:
    text = _non_empty_text(value, label)
    try:
        digest = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if len(digest) != 32 or digest.hex() != text:
        raise ValueError(f"{label} is invalid")
    return digest
