from __future__ import annotations

import asyncio
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from threading import Event
from unittest.mock import Mock
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import pymysql
import pytest
from pymysql.protocol import MysqlPacket, OKPacketWrapper

from jharness.kernel import DurableCommit, RepositoryError, RevisionConflict
from jharness.repository._codec import (  # pyright: ignore[reportPrivateUsage]
    CommitIdentity,
    commit_identity,
    encode_core,
)
from jharness.repository.mysql import (
    MySQLRunRepository,
    MySQLTLS,
    _CommitOutcomeUnknown,  # pyright: ignore[reportPrivateUsage]
    _settle_future,  # pyright: ignore[reportPrivateUsage]
)
from tests.repository_backends.support import append_external, started

_MYSQL_URL_ENV = "JHARNESS_TEST_MYSQL_URL"
_COMMIT_RESPONSE_ERROR_CASES: tuple[
    tuple[type[Exception], tuple[object, ...]],
    ...,
] = (
    (pymysql.err.OperationalError, (2014, "commands out of sync")),
    (
        pymysql.err.InternalError,
        ("Packet sequence number wrong - got 2 expected 1",),
    ),
    (struct.error, ("truncated COMMIT response",)),
    (IndexError, ("empty COMMIT response",)),
    (
        UnicodeDecodeError,
        ("utf-8", b"\xff", 0, 1, "invalid COMMIT SQLSTATE"),
    ),
)


@dataclass(frozen=True)
class _MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


class _ScriptedCursor:
    def __init__(
        self,
        *rows: tuple[object, ...] | None,
        execute_error_at: int | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.rows = list(rows)
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False
        self._execute_error_at = execute_error_at
        self._close_error = close_error

    def execute(self, query: str, args: tuple[object, ...] = ()) -> int:
        self.queries.append((query, args))
        if self._execute_error_at == len(self.queries):
            raise pymysql.err.OperationalError(1146, "injected schema failure")
        return 1

    def fetchone(self) -> tuple[object, ...] | None:
        if not self.rows:
            return None
        return self.rows.pop(0)

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _ScriptedConnection:
    def __init__(
        self,
        cursor: _ScriptedCursor,
        *,
        commit_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self._cursor = cursor
        self._commit_error = commit_error
        self._cleanup_error = cleanup_error
        self.begun = False
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> _ScriptedCursor:
        return self._cursor

    def begin(self) -> None:
        self.begun = True

    def commit(self) -> None:
        self.committed = True
        if self._commit_error is not None:
            raise self._commit_error

    def rollback(self) -> None:
        self.rolled_back = True
        if self._cleanup_error is not None:
            raise self._cleanup_error

    def close(self) -> None:
        self.closed = True
        if self._cleanup_error is not None:
            raise self._cleanup_error


def _head_row(identity: CommitIdentity) -> tuple[object, ...]:
    core = encode_core(identity)
    return (
        identity.run_id,
        identity.revision,
        identity.checkpoint_id,
        identity.parent_checkpoint_id,
        identity.digest,
        core.digest,
        0,
        1,
        identity.history_count,
        identity.history_digest,
    )


def _ledger_row(identity: CommitIdentity) -> tuple[object, ...]:
    return (
        identity.run_id,
        identity.checkpoint_id,
        identity.revision,
        identity.digest,
    )


def _mysql_config() -> _MySQLConfig:
    raw_url = os.environ.get(_MYSQL_URL_ENV)
    if raw_url is None:
        pytest.skip(f"set {_MYSQL_URL_ENV} to run MySQL integration tests")
    parsed = urlsplit(raw_url)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise ValueError(f"{_MYSQL_URL_ENV} must use the mysql scheme")
    database = unquote(parsed.path.removeprefix("/"))
    if parsed.hostname is None or parsed.username is None or not database:
        raise ValueError(f"{_MYSQL_URL_ENV} must include host, user, and database")
    return _MySQLConfig(
        host=parsed.hostname,
        port=parsed.port or 3306,
        user=unquote(parsed.username),
        password="" if parsed.password is None else unquote(parsed.password),
        database=database,
    )


def _repository(table_prefix: str) -> MySQLRunRepository:
    config = _mysql_config()
    return MySQLRunRepository(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        table_prefix=table_prefix,
    )


def _drop_test_tables(table_prefix: str) -> None:
    config = _mysql_config()
    table_names = (
        f"{table_prefix}_v2_history_chunks",
        f"{table_prefix}_v2_checkpoint_ledger",
        f"{table_prefix}_v2_run_heads",
    )
    connection = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset="utf8mb4",
        autocommit=True,
    )
    try:
        with connection.cursor() as cursor:
            for table_name in table_names:
                cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            for table_name in table_names:
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (config.database, table_name),
                )
                assert cursor.fetchone() == (0,)
    finally:
        connection.close()


async def _capture_commit(
    repository: MySQLRunRepository,
    commit: DurableCommit,
) -> BaseException | None:
    try:
        await repository.commit(commit)
    except BaseException as exc:
        return exc
    return None


async def test_mysql_schema_uses_only_hashed_v2_tables() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_unit")
    try:
        heads = repository._create_heads_sql()  # pyright: ignore[reportPrivateUsage]
        ledger = repository._create_ledger_sql()  # pyright: ignore[reportPrivateUsage]
        history = repository._create_history_sql()  # pyright: ignore[reportPrivateUsage]
        assert "jharness_unit_v2_run_heads" in heads
        assert "run_key BINARY(32)" in heads
        assert "checkpoint_core LONGBLOB" in heads
        assert "jharness_unit_v2_checkpoint_ledger" in ledger
        assert "PRIMARY KEY (run_key, checkpoint_key)" in ledger
        assert "UNIQUE KEY run_revision (run_key, revision)" in ledger
        assert "jharness_unit_v2_history_chunks" in history
        assert "PRIMARY KEY (run_key, history_generation, chunk_index)" in history
        assert "_v1_" not in heads + ledger + history
    finally:
        await repository.close()


def test_mysql_real_packet_parser_exposes_commit_response_builtin_errors() -> None:
    truncated_ok = MysqlPacket(b"\x00\xfe\x00\x00\x00\x00\x00", "utf8")
    with pytest.raises(struct.error):
        OKPacketWrapper(truncated_ok)

    empty = MysqlPacket(b"", "utf8")
    with pytest.raises(IndexError):
        empty.is_error_packet()

    invalid_sqlstate = MysqlPacket(b"\xff\x01\x00#\xff\xff\xff\xff\xff", "utf8")
    with pytest.raises(UnicodeDecodeError):
        invalid_sqlstate.raise_for_error()


def test_mysql_rejects_unsafe_or_too_long_table_prefix() -> None:
    with pytest.raises(ValueError, match="table_prefix"):
        MySQLRunRepository(table_prefix="jharness; DROP TABLE runs")
    with pytest.raises(ValueError, match="too long"):
        MySQLRunRepository(table_prefix="j" * 60)


def test_mysql_tls_is_immutable_and_validates_certificate_pair() -> None:
    tls = MySQLTLS(
        ca="root-ca.pem",
        cert="client-cert.pem",
        key="client-key.pem",
        key_password="secret",
    )
    assert tls.verify_identity is True
    assert "secret" not in repr(tls)
    with pytest.raises(FrozenInstanceError):
        tls.ca = "other-ca.pem"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="ca"):
        MySQLTLS(ca="")
    with pytest.raises(ValueError, match="provided together"):
        MySQLTLS(ca="root-ca.pem", cert="client-cert.pem")
    with pytest.raises(ValueError, match="provided together"):
        MySQLTLS(ca="root-ca.pem", key="client-key.pem")
    with pytest.raises(ValueError, match="key_password"):
        MySQLTLS(ca="root-ca.pem", key_password="secret")
    with pytest.raises(TypeError, match="boolean"):
        MySQLTLS(ca="root-ca.pem", verify_identity=1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="MySQLTLS"):
        MySQLRunRepository(tls={})  # pyright: ignore[reportArgumentType]


async def test_mysql_tls_parameters_are_explicit_and_driver_stays_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _ScriptedConnection(_ScriptedCursor())
    connect = Mock(return_value=connection)
    loads = 0

    def load_driver() -> object:
        nonlocal loads
        loads += 1
        return type("Driver", (), {"connect": staticmethod(connect)})()

    monkeypatch.setattr("jharness.repository.mysql._load_pymysql", load_driver)
    tls = MySQLTLS(
        ca="root-ca.pem",
        cert="client-cert.pem",
        key="client-key.pem",
        key_password="secret",
        verify_identity=False,
    )
    secured = MySQLRunRepository(
        host="mysql.internal",
        port=3307,
        user="agent",
        password="password",
        database="runs",
        tls=tls,
        connect_timeout=11,
        read_timeout=12,
        write_timeout=13,
    )
    plaintext = MySQLRunRepository()
    assert loads == 0
    try:
        assert secured._connect(autocommit=True) is connection  # pyright: ignore[reportPrivateUsage]
        assert loads == 1
        connect.assert_called_once_with(
            host="mysql.internal",
            port=3307,
            user="agent",
            password="password",
            database="runs",
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=11,
            read_timeout=12,
            write_timeout=13,
            ssl_ca="root-ca.pem",
            ssl_cert="client-cert.pem",
            ssl_key="client-key.pem",
            ssl_key_password="secret",
            ssl_verify_cert=True,
            ssl_verify_identity=False,
        )

        connect.reset_mock()
        assert plaintext._connect(autocommit=False) is connection  # pyright: ignore[reportPrivateUsage]
        assert loads == 2
        options = connect.call_args.kwargs
        assert not any(name.startswith("ssl_") for name in options)
    finally:
        await secured.close()
        await plaintext.close()


async def test_mysql_driver_is_lazy_and_failed_enter_closes_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def missing_driver(name: str) -> object:
        imports.append(name)
        raise ModuleNotFoundError("No module named 'pymysql'", name="pymysql")

    monkeypatch.setattr("jharness.repository.mysql.import_module", missing_driver)
    repository = MySQLRunRepository(table_prefix="jharness_missing")
    with pytest.raises(RepositoryError, match=r"jharness-repository\[mysql\]"):
        async with repository:
            raise AssertionError("initialization unexpectedly succeeded")
    assert imports == ["pymysql"]
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("after-failure")
    await repository.close()


async def test_mysql_initialization_failure_closes_driver_resources() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_init_failure")
    cursor = _ScriptedCursor(execute_error_at=2)
    connection = _ScriptedConnection(cursor)

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    object.__setattr__(repository, "_connect", connect)

    with pytest.raises(RepositoryError, match="initialization"):
        async with repository:
            raise AssertionError("initialization unexpectedly succeeded")
    assert cursor.closed
    assert connection.closed
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("after-failure")


@pytest.mark.parametrize(
    "transport_error",
    [
        pymysql.err.OperationalError(2013, "lost response"),
        pymysql.err.OperationalError(2014, "commands out of sync"),
        pymysql.err.InternalError("Packet sequence number wrong - got 2 expected 1"),
        struct.error("truncated COMMIT response"),
        IndexError("empty COMMIT response"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid COMMIT SQLSTATE"),
        pymysql.err.InterfaceError(0, "connection closed"),
        OSError("connection reset"),
    ],
)
async def test_mysql_lost_commit_response_is_unknown_and_cleanup_is_best_effort(
    transport_error: Exception,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_unknown")
    identity = commit_identity(started("run-a", "cp-0"))
    cursor = _ScriptedCursor(None, None, close_error=RuntimeError("close failed"))
    connection = _ScriptedConnection(
        cursor,
        commit_error=transport_error,
        cleanup_error=RuntimeError("cleanup failed"),
    )

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    object.__setattr__(repository, "_connect", connect)
    try:
        with pytest.raises(_CommitOutcomeUnknown) as raised:
            repository._commit_once(identity)  # pyright: ignore[reportPrivateUsage]
        assert raised.value.__cause__ is transport_error
        assert connection.begun and connection.committed and connection.rolled_back
        assert cursor.closed and connection.closed
    finally:
        await repository.close()


async def test_mysql_server_err_packet_internal_error_fails_immediately() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_server_commit_error")
    identity = commit_identity(started("run-a", "cp-0"))
    packet = MysqlPacket(b"\xff\x3e\x00#HY000Packets out of order", encoding="utf8")
    with pytest.raises(pymysql.err.InternalError) as parsed:
        packet.raise_for_error()
    server_error = parsed.value
    assert server_error.args == (62, "Packets out of order")
    cursor = _ScriptedCursor(None, None)
    connection = _ScriptedConnection(cursor, commit_error=server_error)
    attempts = 0

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        nonlocal attempts
        assert autocommit is False
        attempts += 1
        return connection

    object.__setattr__(repository, "_initialize_sync", lambda: None)
    object.__setattr__(repository, "_connect", connect)
    try:
        with pytest.raises(pymysql.err.InternalError) as raised:
            repository._commit_sync(identity)  # pyright: ignore[reportPrivateUsage]
        assert raised.value is server_error
        assert attempts == 1
        assert connection.committed and connection.rolled_back and connection.closed
    finally:
        await repository.close()


@pytest.mark.parametrize("persisted", [False, True])
@pytest.mark.parametrize(
    ("error_type", "error_args"),
    _COMMIT_RESPONSE_ERROR_CASES,
)
async def test_mysql_commit_parse_failure_settles_new_or_existing_state(
    monkeypatch: pytest.MonkeyPatch,
    persisted: bool,
    error_type: type[Exception],
    error_args: tuple[object, ...],
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_commit_parse")
    identity = commit_identity(started("run-a", "cp-0"))
    first_cursor = _ScriptedCursor(None, None)
    first = _ScriptedConnection(
        first_cursor,
        commit_error=error_type(*error_args),
    )
    second_cursor = (
        _ScriptedCursor(
            _head_row(identity),
            _ledger_row(identity),
            _ledger_row(identity),
        )
        if persisted
        else _ScriptedCursor(None, None)
    )
    second = _ScriptedConnection(second_cursor)
    connections = [first, second]

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        assert autocommit is False
        return connections.pop(0)

    object.__setattr__(repository, "_initialize_sync", lambda: None)
    object.__setattr__(repository, "_connect", connect)

    def no_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("jharness.repository.mysql.sleep", no_sleep)
    try:
        repository._commit_sync(identity)  # pyright: ignore[reportPrivateUsage]
        assert connections == []
        assert first.committed and first.rolled_back and first.closed
        assert second.committed and second.closed
        second_inserted = any("INSERT INTO" in query for query, _ in second_cursor.queries)
        assert second_inserted is not persisted
    finally:
        await repository.close()


@pytest.mark.parametrize(
    ("error_type", "error_args"),
    _COMMIT_RESPONSE_ERROR_CASES,
)
async def test_mysql_same_parse_failure_in_transaction_body_fails_immediately(
    error_type: type[Exception],
    error_args: tuple[object, ...],
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_body_parse")
    identity = commit_identity(started("run-a", "cp-0"))
    body_error = error_type(*error_args)
    attempts = 0

    def commit_once(_: object) -> None:
        nonlocal attempts
        attempts += 1
        raise body_error

    object.__setattr__(repository, "_initialize_sync", lambda: None)
    object.__setattr__(repository, "_commit_once", commit_once)
    try:
        with pytest.raises(error_type) as raised:
            repository._commit_sync(identity)  # pyright: ignore[reportPrivateUsage]
        assert raised.value is body_error
        assert attempts == 1
    finally:
        await repository.close()


async def test_mysql_unknown_outcome_retries_until_idempotently_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_settlement")
    identity = commit_identity(started("run-a", "cp-0"))
    outcomes: list[Exception | None] = [
        _CommitOutcomeUnknown(),
        pymysql.err.OperationalError(2003, "cannot connect"),
        pymysql.err.OperationalError(1213, "deadlock"),
        None,
    ]
    attempts = 0

    def commit_once(_: object) -> None:
        nonlocal attempts
        attempts += 1
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    object.__setattr__(repository, "_initialize_sync", lambda: None)
    object.__setattr__(repository, "_commit_once", commit_once)

    def no_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("jharness.repository.mysql.sleep", no_sleep)
    try:
        repository._commit_sync(identity)  # pyright: ignore[reportPrivateUsage]
        assert attempts == 4
    finally:
        await repository.close()


async def test_mysql_exact_retry_does_not_encode_core_or_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_retry")
    identity = commit_identity(started("run-a", "cp-0"))
    cursor = _ScriptedCursor(
        _head_row(identity),
        _ledger_row(identity),
        _ledger_row(identity),
    )
    connection = _ScriptedConnection(cursor)

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    object.__setattr__(repository, "_connect", connect)

    def forbidden(*args: object) -> None:
        del args
        raise AssertionError("exact retry must not encode payloads")

    monkeypatch.setattr("jharness.repository.mysql.encode_core", forbidden)
    monkeypatch.setattr("jharness.repository.mysql.encode_history_change", forbidden)
    monkeypatch.setattr("jharness.repository.mysql.decode_core", forbidden)
    try:
        repository._commit_once(identity)  # pyright: ignore[reportPrivateUsage]
        assert connection.committed
        assert not any("INSERT INTO" in query for query, _ in cursor.queries)
    finally:
        await repository.close()


async def test_mysql_append_inserts_exactly_one_delta_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_delta")
    first = started("run-a", "cp-0")
    initial = commit_identity(first)
    appended = commit_identity(append_external(first.checkpoint, "cp-1"))
    cursor = _ScriptedCursor(_head_row(initial), _ledger_row(initial), None)
    connection = _ScriptedConnection(cursor)

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    def forbidden_decode(*args: object) -> None:
        del args
        raise AssertionError("commit path must not decode the previous core")

    object.__setattr__(repository, "_connect", connect)
    monkeypatch.setattr("jharness.repository.mysql.decode_core", forbidden_decode)
    try:
        repository._commit_once(appended)  # pyright: ignore[reportPrivateUsage]
        history_inserts = [
            query
            for query, _ in cursor.queries
            if "INSERT INTO `jharness_delta_v2_history_chunks`" in query
        ]
        assert len(history_inserts) == 1
        assert connection.committed
    finally:
        await repository.close()


async def test_mysql_commit_settles_worker_after_cancellation() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_cancel", max_workers=1)
    worker_started = Event()
    release = Event()

    def blocking_commit(_: object) -> None:
        worker_started.set()
        if not release.wait(2):
            raise AssertionError("test did not release MySQL commit worker")

    object.__setattr__(repository, "_commit_sync", blocking_commit)
    task = asyncio.create_task(repository.commit(started("run-a", "cp-0")))
    try:
        assert await asyncio.to_thread(worker_started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, 1)
    finally:
        release.set()
        await repository.close()


async def test_mysql_cancellation_removes_a_queued_worker() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    started_event = Event()
    release = Event()
    executed = Event()

    def occupy_worker() -> None:
        started_event.set()
        if not release.wait(2):
            raise AssertionError("test did not release occupied worker")

    running = executor.submit(occupy_worker)
    try:
        assert await asyncio.to_thread(started_event.wait, 1)
        queued = executor.submit(executed.set)
        settling = asyncio.create_task(_settle_future(queued))
        await asyncio.sleep(0)
        settling.cancel()
        with pytest.raises(asyncio.CancelledError):
            await settling
        assert queued.cancelled()
        assert not executed.is_set()
    finally:
        release.set()
        await asyncio.to_thread(running.result, 1)
        executor.shutdown(wait=True, cancel_futures=False)


@pytest.mark.skipif(
    _MYSQL_URL_ENV not in os.environ,
    reason=f"set {_MYSQL_URL_ENV} to run MySQL integration tests",
)
async def test_mysql_atomic_cas_per_run_ledger_and_reliable_cleanup() -> None:
    table_prefix = f"jharness_t_{uuid4().hex[:16]}"
    repository = _repository(table_prefix)
    peer = _repository(table_prefix)
    first = started("run-a", "shared-checkpoint")
    other = started("run-b", "shared-checkpoint")
    left = append_external(first.checkpoint, "checkpoint-left")
    right = append_external(first.checkpoint, "checkpoint-right")
    try:
        async with repository, peer:
            await repository.commit(first)
            await peer.commit(other)
            assert await peer.get_head("run-a") == first.checkpoint
            outcomes = await asyncio.gather(
                _capture_commit(repository, left),
                _capture_commit(peer, right),
            )
            assert outcomes.count(None) == 1
            assert sum(isinstance(item, RevisionConflict) for item in outcomes) == 1
            head = await repository.get_head("run-a")
            assert head in (left.checkpoint, right.checkpoint)
            await repository.commit(first)
            collision = append_external(first.checkpoint, first.checkpoint_id, text="changed")
            with pytest.raises(RepositoryError, match="reused"):
                await repository.commit(collision)
    finally:
        try:
            await repository.close()
        finally:
            try:
                await peer.close()
            finally:
                await asyncio.to_thread(_drop_test_tables, table_prefix)
