"""SQLite-backed incremental durable-commit repository."""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock
from typing import TypeVar, cast

from jharness.kernel import (
    Checkpoint,
    DurableCommit,
    HistoryAppend,
    HistoryReplace,
    HistoryUnchanged,
    RepositoryError,
    RevisionConflict,
)

from ._codec import (
    HISTORY_CHUNK_SIZE,
    CommitIdentity,
    EncodedHistoryChunk,
    commit_identity,
    decode_core,
    encode_core,
    encode_history_change,
    reconstruct_checkpoint,
)

_T = TypeVar("_T")

_CREATE_HEADS = """
CREATE TABLE IF NOT EXISTS jharness_v2_run_heads (
    run_id TEXT NOT NULL PRIMARY KEY CHECK (length(run_id) > 0),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    checkpoint_id TEXT NOT NULL CHECK (length(checkpoint_id) > 0),
    parent_checkpoint_id TEXT,
    checkpoint_digest BLOB NOT NULL
        CHECK (typeof(checkpoint_digest) = 'blob' AND length(checkpoint_digest) = 32),
    checkpoint_core BLOB NOT NULL CHECK (typeof(checkpoint_core) = 'blob'),
    checkpoint_core_digest BLOB NOT NULL
        CHECK (typeof(checkpoint_core_digest) = 'blob' AND length(checkpoint_core_digest) = 32),
    history_generation INTEGER NOT NULL CHECK (history_generation >= 0),
    history_chunk_count INTEGER NOT NULL CHECK (history_chunk_count > 0),
    history_message_count INTEGER NOT NULL CHECK (history_message_count > 0),
    history_digest BLOB NOT NULL
        CHECK (typeof(history_digest) = 'blob' AND length(history_digest) = 32),
    CHECK (
        (revision = 0 AND parent_checkpoint_id IS NULL)
        OR (revision > 0 AND length(parent_checkpoint_id) > 0)
    )
)
"""

_CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS jharness_v2_checkpoint_ledger (
    run_id TEXT NOT NULL CHECK (length(run_id) > 0),
    checkpoint_id TEXT NOT NULL CHECK (length(checkpoint_id) > 0),
    revision INTEGER NOT NULL CHECK (revision >= 0),
    checkpoint_digest BLOB NOT NULL
        CHECK (typeof(checkpoint_digest) = 'blob' AND length(checkpoint_digest) = 32),
    PRIMARY KEY (run_id, checkpoint_id),
    UNIQUE (run_id, revision)
) WITHOUT ROWID
"""

_CREATE_HISTORY = f"""
CREATE TABLE IF NOT EXISTS jharness_v2_history_chunks (
    run_id TEXT NOT NULL CHECK (length(run_id) > 0),
    history_generation INTEGER NOT NULL CHECK (history_generation >= 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    message_count INTEGER NOT NULL
        CHECK (message_count > 0 AND message_count <= {HISTORY_CHUNK_SIZE}),
    chunk_payload BLOB NOT NULL CHECK (typeof(chunk_payload) = 'blob'),
    chunk_digest BLOB NOT NULL
        CHECK (typeof(chunk_digest) = 'blob' AND length(chunk_digest) = 32),
    PRIMARY KEY (run_id, history_generation, chunk_index)
) WITHOUT ROWID
"""


@dataclass(frozen=True, slots=True)
class _Head:
    run_id: str
    revision: int
    checkpoint_id: str
    parent_checkpoint_id: str | None
    checkpoint_digest: bytes
    core_digest: bytes
    history_generation: int
    history_chunk_count: int
    history_message_count: int
    history_digest: bytes


@dataclass(frozen=True, slots=True)
class _CompleteHead(_Head):
    core_payload: bytes


class SQLiteRunRepository:
    """Multi-run CAS repository backed by an embedded SQLite database.

    SQLite work is confined to one dedicated worker per repository instance so
    calls never block the event-loop thread. Separate instances and processes
    coordinate through SQLite's write transaction.
    """

    def __init__(
        self,
        database: str | os.PathLike[str],
        *,
        timeout: float = 5.0,
        uri: bool = False,
    ) -> None:
        database_path = os.fspath(database)
        if not isinstance(database_path, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("database must resolve to a string path")
        database_path = str.__str__(database_path)
        if not database_path:
            raise ValueError("database must not be empty")
        if isinstance(timeout, bool) or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            timeout, (int, float)
        ):
            raise TypeError("timeout must be a number")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be greater than zero")
        if not isinstance(uri, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("uri must be a boolean")

        self._database = database_path
        self._timeout = float(timeout)
        self._uri = uri
        self._connection: sqlite3.Connection | None = None
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jharness-sqlite")
        self._lifecycle_lock = Lock()
        self._close_future: Future[None] | None = None
        self._closed = False

    async def initialize(self) -> None:
        """Create the v2 repository schema if needed."""

        await self._run(self._initialize_sync)

    async def commit(self, commit: DurableCommit) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""

        identity = commit_identity(commit)
        await self._run(lambda: self._commit_sync(identity))

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""

        normalized = _validate_run_id(run_id)
        return await self._run(lambda: self._get_head_sync(normalized))

    async def close(self) -> None:
        """Close the SQLite connection and release the dedicated worker."""

        with self._lifecycle_lock:
            if self._closed:
                return
            close_future = self._close_future
            owns_close = close_future is None
            if close_future is None:
                close_future = self._executor.submit(self._close_sync)
                self._close_future = close_future
        try:
            await _settle_future(close_future, cancel_if_queued=False)
        except sqlite3.Error as exc:
            raise RepositoryError("SQLite repository close failed") from exc
        finally:
            if owns_close:
                self._executor.shutdown(wait=True, cancel_futures=False)
                with self._lifecycle_lock:
                    self._closed = True

    async def __aenter__(self) -> SQLiteRunRepository:
        try:
            await self.initialize()
        except BaseException:
            with suppress(Exception, asyncio.CancelledError):
                await self.close()
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        await self.close()

    async def _run(self, operation: Callable[[], _T]) -> _T:
        with self._lifecycle_lock:
            if self._close_future is not None or self._closed:
                raise RepositoryError("SQLite repository is closed")
            future = self._executor.submit(operation)
        try:
            return await _settle_future(future)
        except RepositoryError:
            raise
        except sqlite3.Error as exc:
            raise RepositoryError("SQLite repository operation failed") from exc

    def _initialize_sync(self) -> None:
        if self._initialized:
            return
        connection = self._get_connection()
        connection.execute(_CREATE_HEADS)
        connection.execute(_CREATE_LEDGER)
        connection.execute(_CREATE_HISTORY)
        self._initialized = True

    def _commit_sync(self, identity: CommitIdentity) -> None:
        self._initialize_sync()
        connection = self._get_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            head = _read_head_manifest(connection, identity.run_id)
            existing = _read_ledger(connection, identity.run_id, identity.checkpoint_id)
            if existing is not None:
                _accept_existing(existing, head, identity)
                connection.commit()
                return

            actual_revision = None if head is None else head.revision
            if actual_revision != identity.expected_revision:
                raise RevisionConflict(
                    identity.run_id,
                    identity.expected_revision,
                    actual_revision,
                )
            _validate_new_base(head, identity)

            core = encode_core(identity)
            chunks = encode_history_change(identity.commit)
            generation, first_index, total_chunks = _next_history_manifest(
                head,
                identity,
                len(chunks),
            )
            _validate_encoded_history(head, identity, chunks)

            connection.executemany(
                """
                INSERT INTO jharness_v2_history_chunks (
                    run_id,
                    history_generation,
                    chunk_index,
                    message_count,
                    chunk_payload,
                    chunk_digest
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        identity.run_id,
                        generation,
                        first_index + offset,
                        chunk.message_count,
                        chunk.payload,
                        chunk.digest,
                    )
                    for offset, chunk in enumerate(chunks)
                ),
            )
            connection.execute(
                """
                INSERT INTO jharness_v2_checkpoint_ledger (
                    run_id, checkpoint_id, revision, checkpoint_digest
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    identity.run_id,
                    identity.checkpoint_id,
                    identity.revision,
                    identity.digest,
                ),
            )
            _write_head(
                connection,
                identity,
                core.payload,
                core.digest,
                generation,
                total_chunks,
                actual_revision,
            )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise

    def _get_head_sync(self, run_id: str) -> Checkpoint | None:
        self._initialize_sync()
        connection = self._get_connection()
        head = _read_complete_head(connection, run_id)
        if head is None:
            return None
        rows = connection.execute(
            """
            SELECT chunk_index, chunk_payload, chunk_digest, message_count
            FROM jharness_v2_history_chunks
            WHERE run_id = ?
              AND history_generation = ?
              AND chunk_index < ?
            ORDER BY chunk_index
            """,
            (run_id, head.history_generation, head.history_chunk_count),
        ).fetchall()
        if len(rows) != head.history_chunk_count:
            raise RepositoryError("stored SQLite checkpoint history is incomplete")
        chunks: list[tuple[bytes, bytes, int]] = []
        message_count = 0
        for expected_index, row in enumerate(rows):
            if _stored_nonnegative_int(row[0], "history chunk index") != expected_index:
                raise RepositoryError("stored SQLite history chunk order is invalid")
            payload = _stored_bytes(row[1], "history chunk payload")
            digest = _stored_digest(row[2], "history chunk digest")
            count = _stored_positive_int(row[3], "history chunk message count")
            if count > HISTORY_CHUNK_SIZE:
                raise RepositoryError("stored SQLite history chunk message count is invalid")
            message_count += count
            chunks.append((payload, digest, count))
        if message_count != head.history_message_count:
            raise RepositoryError("stored SQLite checkpoint history count is inconsistent")
        checkpoint, core = reconstruct_checkpoint(
            core_payload=head.core_payload,
            core_digest=head.core_digest,
            chunks=chunks,
            expected_checkpoint_digest=head.checkpoint_digest,
        )
        if (
            checkpoint.id != head.checkpoint_id
            or checkpoint.snapshot.context.run_id != run_id
            or checkpoint.snapshot.revision != head.revision
            or core.parent_checkpoint_id != head.parent_checkpoint_id
            or core.history_digest != head.history_digest
        ):
            raise RepositoryError("stored SQLite run head is inconsistent")
        return checkpoint

    def _get_connection(self) -> sqlite3.Connection:
        connection = self._connection
        if connection is not None:
            return connection
        connection = sqlite3.connect(
            self._database,
            timeout=self._timeout,
            isolation_level=None,
            uri=self._uri,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)}")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return connection

    def _close_sync(self) -> None:
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            finally:
                self._connection = None


def _read_head_manifest(connection: sqlite3.Connection, run_id: str) -> _Head | None:
    row = connection.execute(
        """
        SELECT revision,
               checkpoint_id,
               parent_checkpoint_id,
               checkpoint_digest,
               checkpoint_core_digest,
               history_generation,
               history_chunk_count,
               history_message_count,
               history_digest
        FROM jharness_v2_run_heads
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    head = _Head(
        run_id=run_id,
        revision=_stored_nonnegative_int(row[0], "revision"),
        checkpoint_id=_stored_text(row[1], "checkpoint id"),
        parent_checkpoint_id=_stored_optional_text(row[2], "parent checkpoint id"),
        checkpoint_digest=_stored_digest(row[3], "checkpoint digest"),
        core_digest=_stored_digest(row[4], "checkpoint core digest"),
        history_generation=_stored_nonnegative_int(row[5], "history generation"),
        history_chunk_count=_stored_positive_int(row[6], "history chunk count"),
        history_message_count=_stored_positive_int(row[7], "history message count"),
        history_digest=_stored_digest(row[8], "history digest"),
    )
    _validate_head_parent(head, "SQLite")
    _validate_head_ledger(connection, head)
    return head


def _read_complete_head(
    connection: sqlite3.Connection,
    run_id: str,
) -> _CompleteHead | None:
    row = connection.execute(
        """
        SELECT revision,
               checkpoint_id,
               parent_checkpoint_id,
               checkpoint_digest,
               checkpoint_core_digest,
               history_generation,
               history_chunk_count,
               history_message_count,
               history_digest,
               checkpoint_core
        FROM jharness_v2_run_heads
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    head = _CompleteHead(
        run_id=run_id,
        revision=_stored_nonnegative_int(row[0], "revision"),
        checkpoint_id=_stored_text(row[1], "checkpoint id"),
        parent_checkpoint_id=_stored_optional_text(row[2], "parent checkpoint id"),
        checkpoint_digest=_stored_digest(row[3], "checkpoint digest"),
        core_digest=_stored_digest(row[4], "checkpoint core digest"),
        history_generation=_stored_nonnegative_int(row[5], "history generation"),
        history_chunk_count=_stored_positive_int(row[6], "history chunk count"),
        history_message_count=_stored_positive_int(row[7], "history message count"),
        history_digest=_stored_digest(row[8], "history digest"),
        core_payload=_stored_bytes(row[9], "checkpoint core"),
    )
    _validate_head_parent(head, "SQLite")
    core = decode_core(head.core_payload, head.core_digest)
    if (
        core.checkpoint_id != head.checkpoint_id
        or core.parent_checkpoint_id != head.parent_checkpoint_id
        or core.revision != head.revision
        or core.history_count != head.history_message_count
        or core.history_digest != head.history_digest
    ):
        raise RepositoryError("stored SQLite run head core is inconsistent")
    _validate_head_ledger(connection, head)
    return head


def _validate_head_parent(head: _Head, backend: str) -> None:
    if (head.revision == 0) != (head.parent_checkpoint_id is None):
        raise RepositoryError(f"stored {backend} run head parent is inconsistent")


def _validate_head_ledger(connection: sqlite3.Connection, head: _Head) -> None:
    current_ledger = _read_ledger(connection, head.run_id, head.checkpoint_id)
    if current_ledger is None:
        raise RepositoryError("stored SQLite run head has no checkpoint ledger entry")
    if current_ledger != (head.revision, head.checkpoint_digest):
        raise RepositoryError("stored SQLite run head and checkpoint ledger differ")


def _read_ledger(
    connection: sqlite3.Connection,
    run_id: str,
    checkpoint_id: str,
) -> tuple[int, bytes] | None:
    row = connection.execute(
        """
        SELECT revision, checkpoint_digest
        FROM jharness_v2_checkpoint_ledger
        WHERE run_id = ? AND checkpoint_id = ?
        """,
        (run_id, checkpoint_id),
    ).fetchone()
    if row is None:
        return None
    return (
        _stored_nonnegative_int(row[0], "ledger revision"),
        _stored_digest(row[1], "ledger checkpoint digest"),
    )


def _accept_existing(
    existing: tuple[int, bytes],
    head: _Head | None,
    identity: CommitIdentity,
) -> None:
    revision, digest = existing
    if digest != identity.digest:
        raise RepositoryError(
            f"checkpoint id {identity.checkpoint_id!r} was reused with new content "
            f"in run {identity.run_id!r}"
        )
    if revision != identity.revision:
        raise RepositoryError("stored SQLite checkpoint ledger is inconsistent")
    if head is None or head.revision < revision:
        raise RepositoryError("stored SQLite checkpoint ledger is orphaned")
    if head.revision == revision and (
        head.checkpoint_id != identity.checkpoint_id or head.checkpoint_digest != digest
    ):
        raise RepositoryError("stored SQLite checkpoint ledger is orphaned")


def _validate_new_base(head: _Head | None, identity: CommitIdentity) -> None:
    if head is None:
        if (
            identity.parent_checkpoint_id is not None
            or identity.base_history_count is not None
            or identity.base_history_digest is not None
        ):
            raise RepositoryError("first durable commit has an invalid history base")
        return
    if identity.parent_checkpoint_id != head.checkpoint_id:
        raise RepositoryError("parent checkpoint does not match the authoritative head")
    if (
        identity.base_history_count != head.history_message_count
        or identity.base_history_digest != head.history_digest
    ):
        raise RepositoryError("history change base does not match the authoritative head")


def _next_history_manifest(
    head: _Head | None,
    identity: CommitIdentity,
    added_chunks: int,
) -> tuple[int, int, int]:
    change = identity.commit.history
    if head is None or isinstance(change, HistoryReplace):
        return identity.revision, 0, added_chunks
    if isinstance(change, HistoryAppend):
        return (
            head.history_generation,
            head.history_chunk_count,
            head.history_chunk_count + added_chunks,
        )
    if not isinstance(change, HistoryUnchanged):
        raise RepositoryError("advanced durable commit has an invalid history mutation")
    return head.history_generation, head.history_chunk_count, head.history_chunk_count


def _validate_encoded_history(
    head: _Head | None,
    identity: CommitIdentity,
    chunks: Sequence[EncodedHistoryChunk],
) -> None:
    added_messages = sum(chunk.message_count for chunk in chunks)
    change = identity.commit.history
    if head is None or isinstance(change, HistoryReplace):
        if not chunks or added_messages != identity.history_count:
            raise RepositoryError("encoded replacement history is inconsistent")
    elif isinstance(change, HistoryAppend):
        if not chunks or head.history_message_count + added_messages != identity.history_count:
            raise RepositoryError("encoded appended history is inconsistent")
    elif chunks or head.history_message_count != identity.history_count:
        raise RepositoryError("encoded unchanged history is inconsistent")


def _write_head(
    connection: sqlite3.Connection,
    identity: CommitIdentity,
    core_payload: bytes,
    core_digest: bytes,
    generation: int,
    chunk_count: int,
    actual_revision: int | None,
) -> None:
    values: tuple[object, ...] = (
        identity.revision,
        identity.checkpoint_id,
        identity.parent_checkpoint_id,
        identity.digest,
        core_payload,
        core_digest,
        generation,
        chunk_count,
        identity.history_count,
        identity.history_digest,
    )
    if actual_revision is None:
        connection.execute(
            """
            INSERT INTO jharness_v2_run_heads (
                run_id,
                revision,
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint_digest,
                checkpoint_core,
                checkpoint_core_digest,
                history_generation,
                history_chunk_count,
                history_message_count,
                history_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (identity.run_id, *values),
        )
        return
    updated = connection.execute(
        """
        UPDATE jharness_v2_run_heads
        SET revision = ?,
            checkpoint_id = ?,
            parent_checkpoint_id = ?,
            checkpoint_digest = ?,
            checkpoint_core = ?,
            checkpoint_core_digest = ?,
            history_generation = ?,
            history_chunk_count = ?,
            history_message_count = ?,
            history_digest = ?
        WHERE run_id = ? AND revision = ? AND checkpoint_id = ?
        """,
        (
            *values,
            identity.run_id,
            identity.expected_revision,
            identity.parent_checkpoint_id,
        ),
    )
    if updated.rowcount != 1:
        raise RepositoryError("SQLite run head changed inside its write transaction")


async def _settle_future(
    future: Future[_T],
    *,
    cancel_if_queued: bool = True,
) -> _T:
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        if cancel_if_queued and future.cancel():
            raise
    while True:
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            continue


def _stored_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return value


def _stored_digest(value: object, label: str) -> bytes:
    digest = _stored_bytes(value, label)
    if len(digest) != 32:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return digest


def _stored_text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return value


def _stored_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _stored_text(value, label)


def _stored_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return value


def _stored_positive_int(value: object, label: str) -> int:
    result = _stored_nonnegative_int(value, label)
    if result == 0:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return result


def _validate_run_id(run_id: str) -> str:
    if not isinstance(cast(object, run_id), str):
        raise TypeError("run_id must be a string")
    normalized = str.__str__(run_id)
    if not normalized:
        raise ValueError("run_id must not be empty")
    return normalized
