"""SQLite-backed checkpoint repository."""

from __future__ import annotations

import asyncio
import math
import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from hashlib import sha256
from threading import Lock
from typing import TypeVar

from jharness.kernel import Checkpoint, RepositoryError, RevisionConflict

from ._codec import EncodedCheckpoint, decode_checkpoint, encode_checkpoint

_T = TypeVar("_T")

_CREATE_HEADS = """
CREATE TABLE IF NOT EXISTS jharness_v1_run_heads (
    run_id TEXT NOT NULL PRIMARY KEY,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    checkpoint_id TEXT NOT NULL,
    checkpoint_digest BLOB NOT NULL CHECK (length(checkpoint_digest) = 32),
    checkpoint_payload BLOB NOT NULL
)
"""

_CREATE_CHECKPOINT_IDS = """
CREATE TABLE IF NOT EXISTS jharness_v1_checkpoint_ids (
    checkpoint_id TEXT NOT NULL PRIMARY KEY,
    run_id TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision >= 0),
    checkpoint_digest BLOB NOT NULL CHECK (length(checkpoint_digest) = 32)
)
"""


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
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jharness-sqlite",
        )
        self._lifecycle_lock = Lock()
        self._close_future: Future[None] | None = None
        self._closed = False

    async def initialize(self) -> None:
        """Create the repository schema if needed."""
        await self._run(self._initialize_sync)

    async def commit(self, checkpoint: Checkpoint) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""
        encoded = encode_checkpoint(checkpoint)
        await self._run(lambda: self._commit_sync(encoded))

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""
        run_id = _validate_run_id(run_id)
        return await self._run(lambda: self._get_head_sync(run_id))

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
        connection.execute(_CREATE_CHECKPOINT_IDS)
        self._initialized = True

    def _commit_sync(self, checkpoint: EncodedCheckpoint) -> None:
        self._initialize_sync()
        connection = self._get_connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            current_head = _read_validated_head(connection, checkpoint.run_id)
            actual_revision = None if current_head is None else current_head.snapshot.revision
            existing = connection.execute(
                """
                SELECT run_id, revision, checkpoint_digest
                FROM jharness_v1_checkpoint_ids
                WHERE checkpoint_id = ?
                """,
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if _accept_existing_checkpoint(existing, current_head, checkpoint):
                connection.commit()
                return

            if actual_revision != checkpoint.expected_revision:
                raise RevisionConflict(
                    checkpoint.run_id,
                    checkpoint.expected_revision,
                    actual_revision,
                )

            connection.execute(
                """
                INSERT INTO jharness_v1_checkpoint_ids (
                    checkpoint_id,
                    run_id,
                    revision,
                    checkpoint_digest
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.revision,
                    checkpoint.digest,
                ),
            )
            if actual_revision is None:
                connection.execute(
                    """
                    INSERT INTO jharness_v1_run_heads (
                        run_id,
                        revision,
                        checkpoint_id,
                        checkpoint_digest,
                        checkpoint_payload
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint.run_id,
                        checkpoint.revision,
                        checkpoint.checkpoint_id,
                        checkpoint.digest,
                        checkpoint.payload,
                    ),
                )
            else:
                updated = connection.execute(
                    """
                    UPDATE jharness_v1_run_heads
                    SET revision = ?,
                        checkpoint_id = ?,
                        checkpoint_digest = ?,
                        checkpoint_payload = ?
                    WHERE run_id = ? AND revision = ?
                    """,
                    (
                        checkpoint.revision,
                        checkpoint.checkpoint_id,
                        checkpoint.digest,
                        checkpoint.payload,
                        checkpoint.run_id,
                        checkpoint.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise RepositoryError("SQLite run head changed inside its write transaction")
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise

    def _get_head_sync(self, run_id: str) -> Checkpoint | None:
        self._initialize_sync()
        return _read_validated_head(self._get_connection(), run_id)

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
            connection.close()
            self._connection = None


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
            # The worker had not started this operation, so no write can
            # appear after cancellation escapes.
            raise
    while True:
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # The worker already owns the operation. Let it reach a known
            # outcome instead of exposing an ambiguous transaction result.
            continue


def _stored_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return value


def _stored_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepositoryError(f"stored SQLite {label} is invalid")
    return value


def _stored_revision(value: object) -> int:
    if type(value) is not int or value < 0:
        raise RepositoryError("stored SQLite revision is invalid")
    return value


def _read_validated_head(
    connection: sqlite3.Connection,
    run_id: str,
) -> Checkpoint | None:
    row = connection.execute(
        """
        SELECT checkpoint_id, revision, checkpoint_digest, checkpoint_payload
        FROM jharness_v1_run_heads
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    checkpoint_id = _stored_text(row[0], "checkpoint id")
    revision = _stored_revision(row[1])
    digest = _stored_bytes(row[2], "checkpoint digest")
    payload = _stored_bytes(row[3], "checkpoint payload")
    if len(digest) != 32 or sha256(payload).digest() != digest:
        raise RepositoryError("stored SQLite run head has an invalid checkpoint digest")

    ledger = connection.execute(
        """
        SELECT run_id, revision, checkpoint_digest
        FROM jharness_v1_checkpoint_ids
        WHERE checkpoint_id = ?
        """,
        (checkpoint_id,),
    ).fetchone()
    if ledger is None:
        raise RepositoryError("stored SQLite run head has no checkpoint ledger entry")
    if (
        _stored_text(ledger[0], "run id") != run_id
        or _stored_revision(ledger[1]) != revision
        or _stored_bytes(ledger[2], "checkpoint digest") != digest
    ):
        raise RepositoryError("stored SQLite run head and checkpoint ledger differ")

    checkpoint = decode_checkpoint(payload)
    if (
        checkpoint.id != checkpoint_id
        or checkpoint.snapshot.context.run_id != run_id
        or checkpoint.snapshot.revision != revision
    ):
        raise RepositoryError("stored SQLite run head is inconsistent")
    return checkpoint


def _accept_existing_checkpoint(
    row: tuple[object, ...] | None,
    current_head: Checkpoint | None,
    checkpoint: EncodedCheckpoint,
) -> bool:
    if row is None:
        return False
    stored_run_id = _stored_text(row[0], "run id")
    stored_revision = _stored_revision(row[1])
    stored_digest = _stored_bytes(row[2], "checkpoint digest")
    if stored_digest != checkpoint.digest:
        raise RepositoryError(
            f"checkpoint id {checkpoint.checkpoint_id!r} was reused with new content"
        )
    if stored_run_id != checkpoint.run_id or stored_revision != checkpoint.revision:
        raise RepositoryError("stored SQLite checkpoint ledger is inconsistent")
    if current_head is None or current_head.snapshot.revision < checkpoint.revision:
        raise RepositoryError("stored SQLite checkpoint ledger is orphaned")
    if (
        current_head.snapshot.revision == checkpoint.revision
        and current_head.id != checkpoint.checkpoint_id
    ):
        raise RepositoryError("stored SQLite checkpoint ledger is orphaned")
    return True


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("run_id must be a string")
    run_id = str.__str__(run_id)
    if not run_id:
        raise ValueError("run_id must not be empty")
    return run_id
