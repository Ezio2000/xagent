"""MySQL-backed incremental durable-commit repository."""

from __future__ import annotations

import asyncio
import re
import struct
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import import_module
from threading import Lock
from time import sleep
from typing import Protocol, TypeVar, cast, final

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
_MYSQL_RETRYABLE_CODES = frozenset({1062, 1205, 1213})
_MYSQL_TRANSPORT_CODES = frozenset({1158, 1159, 1160, 1161, 2002, 2003, 2006, 2013, 2055})
_TABLE_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


class _CommitOutcomeUnknown(Exception):
    """Signal that MySQL may have committed without returning its response."""


class _Cursor(Protocol):
    def execute(self, query: str, args: tuple[object, ...] = ()) -> int: ...

    def fetchone(self) -> tuple[object, ...] | None: ...

    def close(self) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def begin(self) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class _PyMySQL(Protocol):
    MySQLError: type[Exception]
    err: _PyMySQLErrors

    def connect(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        charset: str,
        autocommit: bool,
        connect_timeout: int,
        read_timeout: int,
        write_timeout: int,
        ssl_ca: str | None = None,
        ssl_cert: str | None = None,
        ssl_key: str | None = None,
        ssl_key_password: str | None = None,
        ssl_verify_cert: bool | None = None,
        ssl_verify_identity: bool | None = None,
    ) -> _Connection: ...


class _PyMySQLErrors(Protocol):
    InternalError: type[Exception]
    InterfaceError: type[Exception]


@final
@dataclass(frozen=True, slots=True)
class MySQLTLS:
    """Explicit certificate settings for a verified MySQL TLS connection."""

    ca: str
    cert: str | None = None
    key: str | None = None
    key_password: str | None = field(default=None, repr=False)
    verify_identity: bool = True

    def __post_init__(self) -> None:
        ca = _non_empty_string(self.ca, "ca")
        cert = _optional_non_empty_string(self.cert, "cert")
        key = _optional_non_empty_string(self.key, "key")
        password = _optional_string(self.key_password, "key_password")
        if (cert is None) != (key is None):
            raise ValueError("cert and key must be provided together")
        if password is not None and key is None:
            raise ValueError("key_password requires cert and key")
        if type(self.verify_identity) is not bool:
            raise TypeError("verify_identity must be a boolean")
        object.__setattr__(self, "ca", ca)
        object.__setattr__(self, "cert", cert)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "key_password", password)


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


class MySQLRunRepository:
    """Multi-run CAS repository backed by MySQL/InnoDB.

    Blocking PyMySQL calls run in a bounded, repository-owned executor.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "jharness",
        tls: MySQLTLS | None = None,
        table_prefix: str = "jharness",
        connect_timeout: int = 10,
        read_timeout: int = 30,
        write_timeout: int = 30,
        max_workers: int = 4,
    ) -> None:
        self._host = _non_empty_string(host, "host")
        self._port = _positive_integer(port, "port", maximum=65535)
        self._user = _non_empty_string(user, "user")
        self._password = _string(password, "password")
        self._database = _non_empty_string(database, "database")
        if tls is not None and not isinstance(cast(object, tls), MySQLTLS):
            raise TypeError("tls must be a MySQLTLS instance or None")
        self._tls = tls
        self._table_prefix = _validate_table_prefix(table_prefix)
        self._connect_timeout = _positive_integer(connect_timeout, "connect_timeout")
        self._read_timeout = _positive_integer(read_timeout, "read_timeout")
        self._write_timeout = _positive_integer(write_timeout, "write_timeout")
        worker_count = _positive_integer(max_workers, "max_workers")

        self._heads_table = f"{self._table_prefix}_v2_run_heads"
        self._ledger_table = f"{self._table_prefix}_v2_checkpoint_ledger"
        self._history_table = f"{self._table_prefix}_v2_history_chunks"
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="jharness-mysql",
        )
        self._initialize_lock = Lock()
        self._lifecycle_lock = Lock()
        self._initialized = False
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def initialize(self) -> None:
        """Connect to MySQL and create the v2 InnoDB schema if needed."""

        await self._run(self._initialize_sync, "initialization")

    async def commit(self, commit: DurableCommit) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""

        identity = commit_identity(commit)
        await self._run(lambda: self._commit_sync(identity), "commit")

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""

        normalized = _non_empty_string(run_id, "run_id")
        return await self._run(lambda: self._get_head_sync(normalized), "read")

    async def close(self) -> None:
        """Wait for submitted work and release the driver executor."""

        with self._lifecycle_lock:
            close_task = self._close_task
            if close_task is None:
                close_task = asyncio.create_task(self._shutdown_executor())
                self._close_task = close_task
        await _settle_task(close_task)

    async def __aenter__(self) -> MySQLRunRepository:
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

    async def _run(self, operation: Callable[[], _T], label: str) -> _T:
        with self._lifecycle_lock:
            if self._close_task is not None or self._closed:
                raise RepositoryError("MySQL repository is closed")
            future = self._executor.submit(operation)
        try:
            return await _settle_future(future)
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"MySQL repository {label} failed") from exc

    async def _shutdown_executor(self) -> None:
        await asyncio.to_thread(self._executor.shutdown, True)
        with self._lifecycle_lock:
            self._closed = True

    def _initialize_sync(self) -> None:
        with self._initialize_lock:
            if self._initialized:
                return
            connection: _Connection | None = None
            cursor: _Cursor | None = None
            try:
                connection = self._connect(autocommit=True)
                cursor = connection.cursor()
                cursor.execute(self._create_heads_sql())
                cursor.execute(self._create_ledger_sql())
                cursor.execute(self._create_history_sql())
            finally:
                _best_effort_close(cursor)
                _best_effort_close(connection)
            self._initialized = True

    def _commit_sync(self, identity: CommitIdentity) -> None:
        self._initialize_sync()
        ordinary_retries = 0
        settlement_failures = 0
        outcome_unknown = False
        while True:
            try:
                self._commit_once(identity)
                return
            except _CommitOutcomeUnknown:
                outcome_unknown = True
            except Exception as exc:
                if outcome_unknown:
                    if not _is_mysql_settlement_retry(exc):
                        raise
                elif ordinary_retries >= 2 or _mysql_error_code(exc) not in _MYSQL_RETRYABLE_CODES:
                    raise
                else:
                    ordinary_retries += 1
                    continue
            sleep(min(0.01 * (2 ** min(settlement_failures, 7)), 1.0))
            settlement_failures += 1

    def _commit_once(self, identity: CommitIdentity) -> None:
        connection: _Connection | None = None
        cursor: _Cursor | None = None
        try:
            connection = self._connect(autocommit=False)
            cursor = connection.cursor()
            connection.begin()
            head = self._lock_head(cursor, identity.run_id)
            existing = self._select_ledger(
                cursor,
                identity.run_id,
                identity.checkpoint_id,
                lock=True,
            )
            if existing is not None:
                _accept_existing(existing, head, identity, "MySQL")
                _commit_connection(connection)
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
            self._insert_chunks(cursor, identity.run_id, generation, first_index, chunks)
            self._insert_ledger(cursor, identity)
            self._write_head(
                cursor,
                identity,
                core.payload,
                core.digest,
                generation,
                total_chunks,
                actual_revision,
            )
            _commit_connection(connection)
        except BaseException:
            if connection is not None:
                _best_effort(connection.rollback)
            raise
        finally:
            _best_effort_close(cursor)
            _best_effort_close(connection)

    def _lock_head(self, cursor: _Cursor, run_id: str) -> _Head | None:
        cursor.execute(
            f"""
            SELECT run_id,
                   revision,
                   checkpoint_id,
                   parent_checkpoint_id,
                   checkpoint_digest,
                   checkpoint_core_digest,
                   history_generation,
                   history_chunk_count,
                   history_message_count,
                   history_digest
            FROM `{self._heads_table}`
            WHERE run_key = %s
            FOR UPDATE
            """,
            (_identifier_key(run_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        head = _decode_head_manifest(row, run_id)
        current = self._select_ledger(cursor, run_id, head.checkpoint_id, lock=True)
        if current is None:
            raise RepositoryError("stored MySQL run head has no checkpoint ledger entry")
        if current != (head.revision, head.checkpoint_digest):
            raise RepositoryError("stored MySQL run head and checkpoint ledger differ")
        return head

    def _select_ledger(
        self,
        cursor: _Cursor,
        run_id: str,
        checkpoint_id: str,
        *,
        lock: bool,
    ) -> tuple[int, bytes] | None:
        suffix = " FOR UPDATE" if lock else ""
        cursor.execute(
            f"""
            SELECT run_id, checkpoint_id, revision, checkpoint_digest
            FROM `{self._ledger_table}`
            WHERE run_key = %s AND checkpoint_key = %s{suffix}
            """,
            (_identifier_key(run_id), _identifier_key(checkpoint_id)),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        stored_run_id = _row_text(row, 0, "run id")
        stored_checkpoint_id = _row_text(row, 1, "checkpoint id")
        if stored_run_id != run_id:
            raise RepositoryError("stored MySQL run id hash collision")
        if stored_checkpoint_id != checkpoint_id:
            raise RepositoryError("stored MySQL checkpoint id hash collision")
        return _row_revision(row, 2), _row_digest(row, 3, "checkpoint digest")

    def _insert_chunks(
        self,
        cursor: _Cursor,
        run_id: str,
        generation: int,
        first_index: int,
        chunks: Sequence[EncodedHistoryChunk],
    ) -> None:
        for offset, chunk in enumerate(chunks):
            cursor.execute(
                f"""
                INSERT INTO `{self._history_table}` (
                    run_key,
                    history_generation,
                    chunk_index,
                    message_count,
                    chunk_payload,
                    chunk_digest
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    _identifier_key(run_id),
                    generation,
                    first_index + offset,
                    chunk.message_count,
                    chunk.payload,
                    chunk.digest,
                ),
            )

    def _insert_ledger(self, cursor: _Cursor, identity: CommitIdentity) -> None:
        cursor.execute(
            f"""
            INSERT INTO `{self._ledger_table}` (
                run_key,
                checkpoint_key,
                run_id,
                checkpoint_id,
                revision,
                checkpoint_digest
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                _identifier_key(identity.run_id),
                _identifier_key(identity.checkpoint_id),
                identity.run_id,
                identity.checkpoint_id,
                identity.revision,
                identity.digest,
            ),
        )

    def _write_head(
        self,
        cursor: _Cursor,
        identity: CommitIdentity,
        core_payload: bytes,
        core_digest: bytes,
        generation: int,
        chunk_count: int,
        actual_revision: int | None,
    ) -> None:
        values: tuple[object, ...] = (
            identity.run_id,
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
            cursor.execute(
                f"""
                INSERT INTO `{self._heads_table}` (
                    run_key,
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
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (_identifier_key(identity.run_id), *values),
            )
            return
        updated = cursor.execute(
            f"""
            UPDATE `{self._heads_table}`
            SET run_id = %s,
                revision = %s,
                checkpoint_id = %s,
                parent_checkpoint_id = %s,
                checkpoint_digest = %s,
                checkpoint_core = %s,
                checkpoint_core_digest = %s,
                history_generation = %s,
                history_chunk_count = %s,
                history_message_count = %s,
                history_digest = %s
            WHERE run_key = %s AND revision = %s AND checkpoint_id = %s
            """,
            (
                *values,
                _identifier_key(identity.run_id),
                identity.expected_revision,
                identity.parent_checkpoint_id,
            ),
        )
        if updated != 1:
            raise RepositoryError("MySQL run head changed inside its write transaction")

    def _get_head_sync(self, run_id: str) -> Checkpoint | None:
        self._initialize_sync()
        connection: _Connection | None = None
        cursor: _Cursor | None = None
        try:
            connection = self._connect(autocommit=True)
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT run_id,
                       revision,
                       checkpoint_id,
                       parent_checkpoint_id,
                       checkpoint_digest,
                       checkpoint_core_digest,
                       history_generation,
                       history_chunk_count,
                       history_message_count,
                       history_digest,
                       checkpoint_core
                FROM `{self._heads_table}`
                WHERE run_key = %s
                """,
                (_identifier_key(run_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            head = _decode_complete_head(row, run_id)
            current = self._select_ledger(cursor, run_id, head.checkpoint_id, lock=False)
            if current is None:
                raise RepositoryError("stored MySQL run head has no checkpoint ledger entry")
            if current != (head.revision, head.checkpoint_digest):
                raise RepositoryError("stored MySQL run head and checkpoint ledger differ")
            chunks, message_count = self._read_history(cursor, run_id, head)
        finally:
            _best_effort_close(cursor)
            _best_effort_close(connection)
        if len(chunks) != head.history_chunk_count:
            raise RepositoryError("stored MySQL checkpoint history is incomplete")
        if message_count != head.history_message_count:
            raise RepositoryError("stored MySQL checkpoint history count is inconsistent")
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
            raise RepositoryError("stored MySQL run head is inconsistent")
        return checkpoint

    def _read_history(
        self,
        cursor: _Cursor,
        run_id: str,
        head: _Head,
    ) -> tuple[list[tuple[bytes, bytes, int]], int]:
        cursor.execute(
            f"""
            SELECT chunk_index, chunk_payload, chunk_digest, message_count
            FROM `{self._history_table}`
            WHERE run_key = %s
              AND history_generation = %s
              AND chunk_index < %s
            ORDER BY chunk_index
            """,
            (_identifier_key(run_id), head.history_generation, head.history_chunk_count),
        )
        chunks: list[tuple[bytes, bytes, int]] = []
        message_count = 0
        while (row := cursor.fetchone()) is not None:
            if _row_revision(row, 0) != len(chunks):
                raise RepositoryError("stored MySQL history chunk order is invalid")
            payload = _row_bytes(row, 1, "history chunk payload")
            digest = _row_digest(row, 2, "history chunk digest")
            count = _row_positive_integer(row, 3, "history chunk message count")
            if count > HISTORY_CHUNK_SIZE:
                raise RepositoryError("stored MySQL history chunk message count is invalid")
            message_count += count
            chunks.append((payload, digest, count))
        return chunks, message_count

    def _connect(self, *, autocommit: bool) -> _Connection:
        driver = _load_pymysql()
        tls = self._tls
        if tls is None:
            return driver.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                charset="utf8mb4",
                autocommit=autocommit,
                connect_timeout=self._connect_timeout,
                read_timeout=self._read_timeout,
                write_timeout=self._write_timeout,
            )
        return driver.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset="utf8mb4",
            autocommit=autocommit,
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
            write_timeout=self._write_timeout,
            ssl_ca=tls.ca,
            ssl_cert=tls.cert,
            ssl_key=tls.key,
            ssl_key_password=tls.key_password,
            ssl_verify_cert=True,
            ssl_verify_identity=tls.verify_identity,
        )

    def _create_heads_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS `{self._heads_table}` (
            run_key BINARY(32) NOT NULL PRIMARY KEY,
            run_id LONGTEXT NOT NULL,
            revision BIGINT UNSIGNED NOT NULL,
            checkpoint_id LONGTEXT NOT NULL,
            parent_checkpoint_id LONGTEXT NULL,
            checkpoint_digest BINARY(32) NOT NULL,
            checkpoint_core LONGBLOB NOT NULL,
            checkpoint_core_digest BINARY(32) NOT NULL,
            history_generation BIGINT UNSIGNED NOT NULL,
            history_chunk_count BIGINT UNSIGNED NOT NULL,
            history_message_count BIGINT UNSIGNED NOT NULL,
            history_digest BINARY(32) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        """

    def _create_ledger_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS `{self._ledger_table}` (
            run_key BINARY(32) NOT NULL,
            checkpoint_key BINARY(32) NOT NULL,
            run_id LONGTEXT NOT NULL,
            checkpoint_id LONGTEXT NOT NULL,
            revision BIGINT UNSIGNED NOT NULL,
            checkpoint_digest BINARY(32) NOT NULL,
            PRIMARY KEY (run_key, checkpoint_key),
            UNIQUE KEY run_revision (run_key, revision)
        ) ENGINE=InnoDB DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        """

    def _create_history_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS `{self._history_table}` (
            run_key BINARY(32) NOT NULL,
            history_generation BIGINT UNSIGNED NOT NULL,
            chunk_index BIGINT UNSIGNED NOT NULL,
            message_count SMALLINT UNSIGNED NOT NULL,
            chunk_payload LONGBLOB NOT NULL,
            chunk_digest BINARY(32) NOT NULL,
            PRIMARY KEY (run_key, history_generation, chunk_index)
        ) ENGINE=InnoDB DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        """


def _decode_head_manifest(row: tuple[object, ...], requested_run_id: str) -> _Head:
    run_id = _row_text(row, 0, "run id")
    if run_id != requested_run_id:
        raise RepositoryError("stored MySQL run id hash collision")
    head = _Head(
        run_id=run_id,
        revision=_row_revision(row, 1),
        checkpoint_id=_row_text(row, 2, "checkpoint id"),
        parent_checkpoint_id=_row_optional_text(row, 3, "parent checkpoint id"),
        checkpoint_digest=_row_digest(row, 4, "checkpoint digest"),
        core_digest=_row_digest(row, 5, "checkpoint core digest"),
        history_generation=_row_revision(row, 6),
        history_chunk_count=_row_positive_integer(row, 7, "history chunk count"),
        history_message_count=_row_positive_integer(row, 8, "history message count"),
        history_digest=_row_digest(row, 9, "history digest"),
    )
    _validate_head_parent(head, "MySQL")
    return head


def _decode_complete_head(
    row: tuple[object, ...],
    requested_run_id: str,
) -> _CompleteHead:
    manifest = _decode_head_manifest(row[:10], requested_run_id)
    head = _CompleteHead(
        run_id=manifest.run_id,
        revision=manifest.revision,
        checkpoint_id=manifest.checkpoint_id,
        parent_checkpoint_id=manifest.parent_checkpoint_id,
        checkpoint_digest=manifest.checkpoint_digest,
        core_digest=manifest.core_digest,
        history_generation=manifest.history_generation,
        history_chunk_count=manifest.history_chunk_count,
        history_message_count=manifest.history_message_count,
        history_digest=manifest.history_digest,
        core_payload=_row_bytes(row, 10, "checkpoint core"),
    )
    core = decode_core(head.core_payload, head.core_digest)
    if (
        core.checkpoint_id != head.checkpoint_id
        or core.parent_checkpoint_id != head.parent_checkpoint_id
        or core.revision != head.revision
        or core.history_count != head.history_message_count
        or core.history_digest != head.history_digest
    ):
        raise RepositoryError("stored MySQL run head core is inconsistent")
    return head


def _validate_head_parent(head: _Head, backend: str) -> None:
    if (head.revision == 0) != (head.parent_checkpoint_id is None):
        raise RepositoryError(f"stored {backend} run head parent is inconsistent")


def _accept_existing(
    existing: tuple[int, bytes],
    head: _Head | None,
    identity: CommitIdentity,
    backend: str,
) -> None:
    revision, digest = existing
    if digest != identity.digest:
        raise RepositoryError(
            f"checkpoint id {identity.checkpoint_id!r} was reused with new content "
            f"in run {identity.run_id!r}"
        )
    if revision != identity.revision:
        raise RepositoryError(f"stored {backend} checkpoint ledger is inconsistent")
    if head is None or head.revision < revision:
        raise RepositoryError(f"stored {backend} checkpoint ledger is orphaned")
    if head.revision == revision and (
        head.checkpoint_id != identity.checkpoint_id or head.checkpoint_digest != digest
    ):
        raise RepositoryError(f"stored {backend} checkpoint ledger is orphaned")


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


async def _settle_future(future: Future[_T]) -> _T:
    wrapped = asyncio.wrap_future(future)
    while True:
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            if future.cancel():
                raise


async def _settle_task(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _load_pymysql() -> _PyMySQL:
    try:
        module = import_module("pymysql")
    except ModuleNotFoundError as exc:
        if exc.name != "pymysql":
            raise
        raise RepositoryError(
            "MySQL repository support is not installed; run: uv add 'jharness-repository[mysql]'"
        ) from exc
    return cast(_PyMySQL, module)


def _identifier_key(value: str) -> bytes:
    return sha256(value.encode("utf-8")).digest()


def _commit_connection(connection: _Connection) -> None:
    try:
        connection.commit()
    except Exception as exc:
        if _is_mysql_commit_outcome_unknown(exc):
            raise _CommitOutcomeUnknown from exc
        raise


def _best_effort(operation: Callable[[], object]) -> None:
    with suppress(Exception):
        operation()


def _best_effort_close(resource: _Cursor | _Connection | None) -> None:
    if resource is not None:
        _best_effort(resource.close)


def _row_text(row: tuple[object, ...], index: int, label: str) -> str:
    if index >= len(row) or type(row[index]) is not str or not row[index]:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return cast(str, row[index])


def _row_optional_text(row: tuple[object, ...], index: int, label: str) -> str | None:
    if index >= len(row):
        raise RepositoryError(f"stored MySQL {label} is invalid")
    if row[index] is None:
        return None
    return _row_text(row, index, label)


def _row_bytes(row: tuple[object, ...], index: int, label: str) -> bytes:
    if index >= len(row) or type(row[index]) is not bytes:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return cast(bytes, row[index])


def _row_digest(row: tuple[object, ...], index: int, label: str) -> bytes:
    digest = _row_bytes(row, index, label)
    if len(digest) != 32:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return digest


def _row_revision(row: tuple[object, ...], index: int) -> int:
    if index >= len(row) or type(row[index]) is not int or cast(int, row[index]) < 0:
        raise RepositoryError("stored MySQL revision is invalid")
    return cast(int, row[index])


def _row_positive_integer(row: tuple[object, ...], index: int, label: str) -> int:
    value = _row_revision(row, index)
    if value == 0:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return value


def _mysql_error_code(error: Exception) -> int | None:
    if error.args and type(error.args[0]) is int:
        return error.args[0]
    return None


def _is_mysql_transport_error(error: Exception) -> bool:
    driver = _load_pymysql()
    return (
        isinstance(error, (driver.err.InterfaceError, OSError))
        or _mysql_error_code(error) in _MYSQL_TRANSPORT_CODES
    )


def _is_mysql_commit_outcome_unknown(error: Exception) -> bool:
    driver = _load_pymysql()
    error_code = _mysql_error_code(error)
    if _is_mysql_transport_error(error) or error_code == 2014:
        return True
    if isinstance(error, driver.err.InternalError):
        return error_code is None
    return isinstance(error, (struct.error, IndexError, UnicodeDecodeError))


def _is_mysql_settlement_retry(error: Exception) -> bool:
    return isinstance(error, (_load_pymysql().MySQLError, OSError))


def _validate_table_prefix(value: str) -> str:
    value = _non_empty_string(value, "table_prefix")
    if _TABLE_PREFIX.fullmatch(value) is None:
        raise ValueError("table_prefix must contain only ASCII letters, digits, and underscores")
    suffixes = ("_v2_run_heads", "_v2_checkpoint_ledger", "_v2_history_chunks")
    if any(len(f"{value}{suffix}") > 64 for suffix in suffixes):
        raise ValueError("table_prefix is too long for a MySQL table name")
    return value


def _string(value: str, label: str) -> str:
    if not isinstance(cast(object, value), str):
        raise TypeError(f"{label} must be a string")
    return str.__str__(value)


def _non_empty_string(value: str, label: str) -> str:
    value = _string(value, label)
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _optional_string(value: str | None, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _optional_non_empty_string(value: str | None, label: str) -> str | None:
    return None if value is None else _non_empty_string(value, label)


def _positive_integer(value: int, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{label} must be greater than zero{suffix}")
    return value
