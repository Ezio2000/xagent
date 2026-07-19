"""MySQL-backed checkpoint repository."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from hashlib import sha256
from importlib import import_module
from threading import Lock
from time import sleep
from typing import Protocol, TypeVar, cast

from jharness.kernel import Checkpoint, RepositoryError, RevisionConflict

from ._codec import EncodedCheckpoint, decode_checkpoint, encode_checkpoint

_T = TypeVar("_T")
_MYSQL_RETRYABLE_CODES = frozenset({1062, 1205, 1213})
_MYSQL_TRANSPORT_CODES = frozenset(
    {
        1158,  # network read error
        1159,  # network read timeout
        1160,  # network write error
        1161,  # network write timeout
        2002,
        2003,
        2006,
        2013,
        2055,
    }
)
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
    ) -> _Connection: ...


class _PyMySQLErrors(Protocol):
    InterfaceError: type[Exception]


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
        self._table_prefix = _validate_table_prefix(table_prefix)
        self._connect_timeout = _positive_integer(connect_timeout, "connect_timeout")
        self._read_timeout = _positive_integer(read_timeout, "read_timeout")
        self._write_timeout = _positive_integer(write_timeout, "write_timeout")
        worker_count = _positive_integer(max_workers, "max_workers")

        self._heads_table = f"{self._table_prefix}_v1_run_heads"
        self._ids_table = f"{self._table_prefix}_v1_checkpoint_ids"
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
        """Connect to MySQL and create the InnoDB schema if needed."""
        await self._run(self._initialize_sync, "initialization")

    async def commit(self, checkpoint: Checkpoint) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""
        encoded = encode_checkpoint(checkpoint)
        await self._run(lambda: self._commit_sync(encoded), "commit")

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""
        run_id = _non_empty_string(run_id, "run_id")
        return await self._run(lambda: self._get_head_sync(run_id), "read")

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
                cursor.execute(self._create_ids_sql())
            finally:
                _best_effort_close(cursor)
                _best_effort_close(connection)
            self._initialized = True

    def _commit_sync(self, checkpoint: EncodedCheckpoint) -> None:
        self._initialize_sync()
        ordinary_retries = 0
        settlement_failures = 0
        outcome_unknown = False
        while True:
            try:
                self._commit_once(checkpoint)
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

            # Once COMMIT may have succeeded, returning a transport failure
            # would expose an ambiguous result. Each iteration opens a fresh
            # connection and replays the complete idempotent transaction.
            sleep(min(0.01 * (2 ** min(settlement_failures, 7)), 1.0))
            settlement_failures += 1

    def _commit_once(self, checkpoint: EncodedCheckpoint) -> None:
        connection: _Connection | None = None
        cursor: _Cursor | None = None
        try:
            connection = self._connect(autocommit=False)
            cursor = connection.cursor()
            connection.begin()
            if self._accept_existing(cursor, checkpoint):
                _commit_connection(connection)
                return
            head = self._lock_head(cursor, checkpoint.run_id)
            actual_revision = None if head is None else head.snapshot.revision
            if actual_revision != checkpoint.expected_revision:
                raise RevisionConflict(
                    checkpoint.run_id,
                    checkpoint.expected_revision,
                    actual_revision,
                )
            self._insert_id(cursor, checkpoint)
            self._write_head(cursor, checkpoint, actual_revision)
            _commit_connection(connection)
        except BaseException:
            if connection is not None:
                _best_effort(connection.rollback)
            raise
        finally:
            _best_effort_close(cursor)
            _best_effort_close(connection)

    def _accept_existing(self, cursor: _Cursor, checkpoint: EncodedCheckpoint) -> bool:
        cursor.execute(
            f"""
            SELECT checkpoint_id, run_id, revision, checkpoint_digest
            FROM `{self._ids_table}`
            WHERE checkpoint_key = %s
            FOR UPDATE
            """,
            (_identifier_key(checkpoint.checkpoint_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        checkpoint_id = _row_text(row, 0, "checkpoint id")
        run_id = _row_text(row, 1, "run id")
        revision = _row_revision(row, 2)
        digest = _row_bytes(row, 3, "checkpoint digest")
        if checkpoint_id == checkpoint.checkpoint_id and digest == checkpoint.digest:
            if run_id != checkpoint.run_id or revision != checkpoint.revision:
                raise RepositoryError("stored MySQL checkpoint ledger is inconsistent")
            head = self._lock_head(cursor, checkpoint.run_id)
            if head is None or head.snapshot.revision < checkpoint.revision:
                raise RepositoryError("stored MySQL checkpoint ledger is orphaned")
            if (
                head.snapshot.revision == checkpoint.revision
                and head.id != checkpoint.checkpoint_id
            ):
                raise RepositoryError("stored MySQL checkpoint ledger is orphaned")
            return True
        if checkpoint_id != checkpoint.checkpoint_id:
            raise RepositoryError("stored MySQL checkpoint id hash collision")
        raise RepositoryError(
            f"checkpoint id {checkpoint.checkpoint_id!r} was reused with new content"
        )

    def _lock_head(self, cursor: _Cursor, run_id: str) -> Checkpoint | None:
        cursor.execute(
            f"""
            SELECT run_id, checkpoint_id, revision, checkpoint_digest, checkpoint_payload
            FROM `{self._heads_table}`
            WHERE run_key = %s
            FOR UPDATE
            """,
            (_identifier_key(run_id),),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        checkpoint = _decode_head(row, run_id)
        digest = _row_bytes(row, 3, "checkpoint digest")
        cursor.execute(
            f"""
            SELECT checkpoint_id, run_id, revision, checkpoint_digest
            FROM `{self._ids_table}`
            WHERE checkpoint_key = %s
            FOR UPDATE
            """,
            (_identifier_key(checkpoint.id),),
        )
        _validate_head_ledger(cursor.fetchone(), checkpoint, digest)
        return checkpoint

    def _insert_id(self, cursor: _Cursor, checkpoint: EncodedCheckpoint) -> None:
        cursor.execute(
            f"""
            INSERT INTO `{self._ids_table}` (
                checkpoint_key,
                checkpoint_id,
                run_id,
                revision,
                checkpoint_digest
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                _identifier_key(checkpoint.checkpoint_id),
                checkpoint.checkpoint_id,
                checkpoint.run_id,
                checkpoint.revision,
                checkpoint.digest,
            ),
        )

    def _write_head(
        self,
        cursor: _Cursor,
        checkpoint: EncodedCheckpoint,
        actual_revision: int | None,
    ) -> None:
        values: tuple[object, ...] = (
            _identifier_key(checkpoint.run_id),
            checkpoint.run_id,
            checkpoint.revision,
            checkpoint.checkpoint_id,
            checkpoint.digest,
            checkpoint.payload,
        )
        if actual_revision is None:
            cursor.execute(
                f"""
                INSERT INTO `{self._heads_table}` (
                    run_key,
                    run_id,
                    revision,
                    checkpoint_id,
                    checkpoint_digest,
                    checkpoint_payload
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                values,
            )
            return
        updated = cursor.execute(
            f"""
            UPDATE `{self._heads_table}`
            SET run_id = %s,
                revision = %s,
                checkpoint_id = %s,
                checkpoint_digest = %s,
                checkpoint_payload = %s
            WHERE run_key = %s AND revision = %s
            """,
            (
                checkpoint.run_id,
                checkpoint.revision,
                checkpoint.checkpoint_id,
                checkpoint.digest,
                checkpoint.payload,
                _identifier_key(checkpoint.run_id),
                checkpoint.expected_revision,
            ),
        )
        if updated != 1:
            raise RepositoryError("MySQL run head changed inside its write transaction")

    def _get_head_sync(self, run_id: str) -> Checkpoint | None:
        self._initialize_sync()
        connection: _Connection | None = None
        cursor: _Cursor | None = None
        row: tuple[object, ...] | None = None
        ledger: tuple[object, ...] | None = None
        try:
            connection = self._connect(autocommit=True)
            cursor = connection.cursor()
            cursor.execute(
                f"""
                SELECT run_id, checkpoint_id, revision, checkpoint_digest, checkpoint_payload
                FROM `{self._heads_table}`
                WHERE run_key = %s
                """,
                (_identifier_key(run_id),),
            )
            row = cursor.fetchone()
            if row is not None:
                checkpoint_id = _row_text(row, 1, "checkpoint id")
                cursor.execute(
                    f"""
                    SELECT checkpoint_id, run_id, revision, checkpoint_digest
                    FROM `{self._ids_table}`
                    WHERE checkpoint_key = %s
                    """,
                    (_identifier_key(checkpoint_id),),
                )
                ledger = cursor.fetchone()
        finally:
            _best_effort_close(cursor)
            _best_effort_close(connection)
        if row is None:
            return None
        checkpoint = _decode_head(row, run_id)
        _validate_head_ledger(
            ledger,
            checkpoint,
            _row_bytes(row, 3, "checkpoint digest"),
        )
        return checkpoint

    def _connect(self, *, autocommit: bool) -> _Connection:
        return _load_pymysql().connect(
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

    def _create_heads_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS `{self._heads_table}` (
            run_key BINARY(32) NOT NULL PRIMARY KEY,
            run_id LONGTEXT NOT NULL,
            revision BIGINT UNSIGNED NOT NULL,
            checkpoint_id LONGTEXT NOT NULL,
            checkpoint_digest BINARY(32) NOT NULL,
            checkpoint_payload LONGBLOB NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        """

    def _create_ids_sql(self) -> str:
        return f"""
        CREATE TABLE IF NOT EXISTS `{self._ids_table}` (
            checkpoint_key BINARY(32) NOT NULL PRIMARY KEY,
            checkpoint_id LONGTEXT NOT NULL,
            run_id LONGTEXT NOT NULL,
            revision BIGINT UNSIGNED NOT NULL,
            checkpoint_digest BINARY(32) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        """


async def _settle_future(future: Future[_T]) -> _T:
    wrapped = asyncio.wrap_future(future)
    while True:
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            if future.cancel():
                # A worker that has not started cannot have crossed an atomic
                # boundary, so cancellation can escape with no later write.
                raise
            # A submitted transaction must reach a known outcome before this
            # call returns or raises.
            continue


async def _settle_task(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            continue


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
        if _is_mysql_transport_error(exc):
            raise _CommitOutcomeUnknown from exc
        raise


def _best_effort(operation: Callable[[], object]) -> None:
    with suppress(Exception):
        operation()


def _best_effort_close(resource: _Cursor | _Connection | None) -> None:
    if resource is not None:
        _best_effort(resource.close)


def _decode_head(row: tuple[object, ...], requested_run_id: str) -> Checkpoint:
    run_id = _row_text(row, 0, "run id")
    checkpoint_id = _row_text(row, 1, "checkpoint id")
    revision = _row_revision(row, 2)
    digest = _row_bytes(row, 3, "checkpoint digest")
    payload = _row_bytes(row, 4, "checkpoint payload")
    if run_id != requested_run_id:
        raise RepositoryError("stored MySQL run id hash collision")
    if len(digest) != 32 or sha256(payload).digest() != digest:
        raise RepositoryError("stored MySQL run head has an invalid checkpoint digest")
    checkpoint = decode_checkpoint(payload)
    if (
        checkpoint.id != checkpoint_id
        or checkpoint.snapshot.context.run_id != run_id
        or checkpoint.snapshot.revision != revision
    ):
        raise RepositoryError("stored MySQL run head is inconsistent")
    return checkpoint


def _validate_head_ledger(
    row: tuple[object, ...] | None,
    checkpoint: Checkpoint,
    head_digest: bytes,
) -> None:
    if row is None:
        raise RepositoryError("stored MySQL run head has no checkpoint ledger entry")
    if (
        _row_text(row, 0, "checkpoint id") != checkpoint.id
        or _row_text(row, 1, "run id") != checkpoint.snapshot.context.run_id
        or _row_revision(row, 2) != checkpoint.snapshot.revision
        or _row_bytes(row, 3, "checkpoint digest") != head_digest
    ):
        raise RepositoryError("stored MySQL run head and checkpoint ledger differ")


def _row_text(row: tuple[object, ...], index: int, label: str) -> str:
    if index >= len(row) or not isinstance(row[index], str) or not row[index]:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return str.__str__(cast(str, row[index]))


def _row_bytes(row: tuple[object, ...], index: int, label: str) -> bytes:
    if index >= len(row) or type(row[index]) is not bytes:
        raise RepositoryError(f"stored MySQL {label} is invalid")
    return cast(bytes, row[index])


def _row_revision(row: tuple[object, ...], index: int) -> int:
    if index >= len(row) or type(row[index]) is not int or cast(int, row[index]) < 0:
        raise RepositoryError("stored MySQL revision is invalid")
    return cast(int, row[index])


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


def _is_mysql_settlement_retry(error: Exception) -> bool:
    # After one COMMIT response was lost, a later driver error cannot prove
    # whether that earlier transaction committed. Keep replaying until the
    # idempotency ledger or a repository semantic error supplies an answer.
    return isinstance(error, (_load_pymysql().MySQLError, OSError))


def _validate_table_prefix(value: str) -> str:
    value = _non_empty_string(value, "table_prefix")
    if _TABLE_PREFIX.fullmatch(value) is None:
        raise ValueError("table_prefix must contain only ASCII letters, digits, and underscores")
    if len(f"{value}_v1_checkpoint_ids") > 64:
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


def _positive_integer(value: int, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{label} must be greater than zero{suffix}")
    return value
