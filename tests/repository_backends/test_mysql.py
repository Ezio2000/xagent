from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from threading import Event
from urllib.parse import unquote, urlsplit
from uuid import uuid4

import pymysql
import pytest

from jharness.kernel import (
    Checkpoint,
    ControlFact,
    Limited,
    LimitedControl,
    LimitReason,
    Message,
    Planning,
    RepositoryError,
    RevisionConflict,
    RunContext,
    RunMetrics,
    RunSnapshot,
    StartedFact,
)
from jharness.repository._codec import (  # pyright: ignore[reportPrivateUsage]
    encode_checkpoint,
)
from jharness.repository.mysql import (
    MySQLRunRepository,
    _CommitOutcomeUnknown,  # pyright: ignore[reportPrivateUsage]
    _settle_future,  # pyright: ignore[reportPrivateUsage]
)

_MYSQL_URL_ENV = "JHARNESS_TEST_MYSQL_URL"


@dataclass(frozen=True)
class _MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


class _StoredIdCursor:
    def __init__(
        self,
        *rows: tuple[object, ...] | None,
        close_error: Exception | None = None,
    ) -> None:
        self.rows = list(rows)
        self.query = ""
        self.args: tuple[object, ...] = ()
        self.closed = False
        self._close_error = close_error

    def execute(self, query: str, args: tuple[object, ...] = ()) -> int:
        self.query = query
        self.args = args
        return 1

    def fetchone(self) -> tuple[object, ...] | None:
        if not self.rows:
            raise AssertionError("test cursor has no scripted row")
        return self.rows.pop(0)

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _ScriptedConnection:
    def __init__(
        self,
        cursor: _StoredIdCursor,
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

    def cursor(self) -> _StoredIdCursor:
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


def _started(run_id: str, checkpoint_id: str) -> Checkpoint:
    context = RunContext(run_id, 1.0)
    snapshot = RunSnapshot(
        0,
        context,
        (Message.user("hello"),),
        RunMetrics(),
        Planning(),
    )
    return Checkpoint(checkpoint_id, snapshot, StartedFact(1.0, ("user",)))


def _limited(previous: Checkpoint, checkpoint_id: str) -> Checkpoint:
    snapshot = RunSnapshot(
        1,
        previous.snapshot.context,
        previous.snapshot.history,
        previous.snapshot.metrics,
        Limited(LimitReason.DEADLINE),
    )
    return Checkpoint(
        checkpoint_id,
        snapshot,
        ControlFact(2.0, LimitedControl(LimitReason.DEADLINE)),
    )


def _head_row(checkpoint: Checkpoint) -> tuple[object, ...]:
    encoded = encode_checkpoint(checkpoint)
    return (
        encoded.run_id,
        encoded.checkpoint_id,
        encoded.revision,
        encoded.digest,
        encoded.payload,
    )


def _ledger_row(checkpoint: Checkpoint) -> tuple[object, ...]:
    encoded = encode_checkpoint(checkpoint)
    return (
        encoded.checkpoint_id,
        encoded.run_id,
        encoded.revision,
        encoded.digest,
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
        f"{table_prefix}_v1_checkpoint_ids",
        f"{table_prefix}_v1_run_heads",
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
    checkpoint: Checkpoint,
) -> BaseException | None:
    try:
        await repository.commit(checkpoint)
    except BaseException as exc:
        return exc
    return None


async def test_mysql_sql_uses_hashed_keys_and_checks_full_checkpoint_id() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_unit")
    first = _started("run-" + "x" * 2048, "checkpoint-" + "y" * 2048)
    encoded = encode_checkpoint(first)
    try:
        heads_sql = repository._create_heads_sql()  # pyright: ignore[reportPrivateUsage]
        ids_sql = repository._create_ids_sql()  # pyright: ignore[reportPrivateUsage]
        assert "run_key BINARY(32)" in heads_sql
        assert "jharness_unit_v1_run_heads" in heads_sql
        assert "run_id LONGTEXT" in heads_sql
        assert "checkpoint_key BINARY(32)" in ids_sql
        assert "jharness_unit_v1_checkpoint_ids" in ids_sql
        assert "checkpoint_id LONGTEXT" in ids_sql

        exact = _StoredIdCursor(
            _ledger_row(first),
            _head_row(first),
            _ledger_row(first),
        )
        assert repository._accept_existing(  # pyright: ignore[reportPrivateUsage]
            exact,
            encoded,
        )
        assert "FOR UPDATE" in exact.query
        checkpoint_key = exact.args[0]
        assert isinstance(checkpoint_key, bytes)
        assert len(checkpoint_key) == 32

        changed = _StoredIdCursor(
            (
                encoded.checkpoint_id,
                encoded.run_id,
                encoded.revision,
                b"x" * 32,
            )
        )
        with pytest.raises(RepositoryError, match="reused"):
            repository._accept_existing(  # pyright: ignore[reportPrivateUsage]
                changed,
                encoded,
            )
    finally:
        await repository.close()


def test_mysql_rejects_unsafe_table_prefix() -> None:
    with pytest.raises(ValueError, match="table_prefix"):
        MySQLRunRepository(table_prefix="jharness; DROP TABLE runs")


async def test_mysql_driver_is_loaded_only_when_the_backend_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def missing_driver(name: str) -> object:
        imports.append(name)
        raise ModuleNotFoundError("No module named 'pymysql'", name="pymysql")

    monkeypatch.setattr("jharness.repository.mysql.import_module", missing_driver)
    repository = MySQLRunRepository(table_prefix="jharness_missing_driver")
    try:
        with pytest.raises(RepositoryError, match=r"jharness-repository\[mysql\]"):
            async with repository:
                raise AssertionError("initialization unexpectedly succeeded")
        assert imports == ["pymysql"]
        with pytest.raises(RepositoryError, match="closed"):
            await repository.get_head("run-after-failed-enter")
    finally:
        await repository.close()


async def test_mysql_normalizes_string_subclasses_before_formatting_and_reads() -> None:
    class HostileString(str):
        def __str__(self) -> str:
            return "attacker-controlled-str"

        def __format__(self, format_spec: str) -> str:
            del format_spec
            return "bad`; DROP TABLE run_heads; --"

    repository = MySQLRunRepository(
        host=HostileString("db.example"),
        user=HostileString("safe-user"),
        password=HostileString("safe-password"),
        database=HostileString("safe-database"),
        table_prefix=HostileString("safe_prefix"),
    )
    captured: list[str] = []

    def capture_read(run_id: str) -> None:
        captured.append(run_id)

    object.__setattr__(repository, "_get_head_sync", capture_read)
    try:
        assert "`safe_prefix_v1_run_heads`" in repository._create_heads_sql()  # pyright: ignore[reportPrivateUsage]
        assert "attacker" not in repository._create_heads_sql()  # pyright: ignore[reportPrivateUsage]
        for name in ("_host", "_user", "_password", "_database", "_table_prefix"):
            assert type(vars(repository)[name]) is str

        assert await repository.get_head(HostileString("safe-run")) is None
        assert captured == ["safe-run"]
        assert type(captured[0]) is str
    finally:
        await repository.close()


@pytest.mark.parametrize(
    "transport_error",
    [
        pymysql.err.OperationalError(1158, "network read error"),
        pymysql.err.OperationalError(2006, "server gone"),
        pymysql.err.OperationalError(2013, "lost response"),
        pymysql.err.OperationalError(2055, "lost extended response"),
        pymysql.err.InterfaceError(0, "connection closed"),
        OSError("connection reset"),
    ],
    ids=(
        "packet-read",
        "server-gone",
        "server-lost",
        "server-lost-extended",
        "interface",
        "os-error",
    ),
)
async def test_mysql_commit_transport_failure_becomes_unknown_and_cleanup_is_best_effort(
    transport_error: Exception,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_unknown_commit")
    cursor = _StoredIdCursor(
        None,
        None,
        close_error=RuntimeError("cursor close failed"),
    )
    connection = _ScriptedConnection(
        cursor,
        commit_error=transport_error,
        cleanup_error=RuntimeError("connection cleanup failed"),
    )

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    object.__setattr__(repository, "_connect", connect)
    try:
        with pytest.raises(_CommitOutcomeUnknown) as raised:
            repository._commit_once(  # pyright: ignore[reportPrivateUsage]
                encode_checkpoint(_started("run-unknown", "checkpoint-unknown"))
            )
        assert raised.value.__cause__ is transport_error
        assert connection.begun
        assert connection.committed
        assert connection.rolled_back
        assert cursor.closed
        assert connection.closed
    finally:
        await repository.close()


async def test_mysql_unknown_commit_retries_transient_failures_until_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = MySQLRunRepository(table_prefix="jharness_settlement")
    checkpoint = encode_checkpoint(_started("run-settlement", "checkpoint-settlement"))
    outcomes: list[Exception | None] = [
        _CommitOutcomeUnknown(),
        pymysql.err.OperationalError(2003, "cannot connect"),
        pymysql.err.OperationalError(1213, "deadlock"),
        pymysql.err.InterfaceError(0, "closed"),
        OSError("reset"),
        pymysql.err.ProgrammingError(1146, "table temporarily unavailable"),
        None,
    ]
    attempts = 0

    def commit_once(_: object) -> None:
        nonlocal attempts
        attempts += 1
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    def no_sleep(_: float) -> None:
        pass

    object.__setattr__(repository, "_initialize_sync", lambda: None)
    object.__setattr__(repository, "_commit_once", commit_once)
    monkeypatch.setattr("jharness.repository.mysql.sleep", no_sleep)
    try:
        repository._commit_sync(checkpoint)  # pyright: ignore[reportPrivateUsage]
        assert attempts == 7
        assert not outcomes

        outcomes.extend([_CommitOutcomeUnknown(), RepositoryError("definitive corruption")])
        with pytest.raises(RepositoryError, match="definitive corruption"):
            repository._commit_sync(checkpoint)  # pyright: ignore[reportPrivateUsage]
        assert attempts == 9
    finally:
        await repository.close()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("digest", "digest"),
        ("payload", "payload"),
        ("ledger", "ledger"),
    ],
)
async def test_mysql_commit_validates_existing_head_before_revision_cas(
    corruption: str,
    message: str,
) -> None:
    repository = MySQLRunRepository(table_prefix=f"jharness_corrupt_{corruption}")
    first = _started("run-corrupt", "checkpoint-first")
    next_checkpoint = _limited(first, "checkpoint-next")
    head = list(_head_row(first))
    ledger = list(_ledger_row(first))
    if corruption == "digest":
        head[3] = b"x" * 32
    elif corruption == "payload":
        payload = b"not-json"
        digest = sha256(payload).digest()
        head[3] = digest
        head[4] = payload
        ledger[3] = digest
    else:
        ledger[3] = b"x" * 32

    cursor = _StoredIdCursor(None, tuple(head), tuple(ledger))
    connection = _ScriptedConnection(cursor)

    def connect(*, autocommit: bool) -> _ScriptedConnection:
        del autocommit
        return connection

    object.__setattr__(repository, "_connect", connect)
    try:
        with pytest.raises(RepositoryError, match=message):
            repository._commit_once(  # pyright: ignore[reportPrivateUsage]
                encode_checkpoint(next_checkpoint)
            )
        assert connection.begun
        assert connection.rolled_back
        assert not connection.committed
    finally:
        await repository.close()


async def test_mysql_exact_retry_rejects_same_revision_with_another_head() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_strict_retry")
    retried = _started("run-strict", "checkpoint-retried")
    other = _started("run-strict", "checkpoint-other")
    cursor = _StoredIdCursor(
        _ledger_row(retried),
        _head_row(other),
        _ledger_row(other),
    )
    try:
        with pytest.raises(RepositoryError, match="orphaned"):
            repository._accept_existing(  # pyright: ignore[reportPrivateUsage]
                cursor,
                encode_checkpoint(retried),
            )
    finally:
        await repository.close()


async def test_mysql_commit_settles_worker_after_cancellation() -> None:
    repository = MySQLRunRepository(table_prefix="jharness_cancel", max_workers=1)
    started = Event()
    release = Event()

    def blocking_commit(checkpoint: object) -> None:
        del checkpoint
        started.set()
        if not release.wait(2):
            raise AssertionError("test did not release MySQL commit worker")

    object.__setattr__(repository, "_commit_sync", blocking_commit)
    commit = asyncio.create_task(repository.commit(_started("run-cancel", "checkpoint-cancel")))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        commit.cancel()
        await asyncio.sleep(0)
        assert not commit.done()
        release.set()
        await asyncio.wait_for(commit, 1)
    finally:
        release.set()
        await repository.close()


async def test_mysql_cancellation_removes_a_queued_worker() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    started = Event()
    release = Event()
    executed = Event()

    def occupy_worker() -> None:
        started.set()
        if not release.wait(2):
            raise AssertionError("test did not release occupied worker")

    running = executor.submit(occupy_worker)
    try:
        assert await asyncio.to_thread(started.wait, 1)
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
async def test_mysql_repository_atomic_cas_and_global_id_ledger() -> None:
    table_prefix = f"jharness_t_{uuid4().hex[:16]}"
    repository = _repository(table_prefix)
    peer = _repository(table_prefix)
    run_id = "run-长-" + "r" * 2048
    first = _started(run_id, "checkpoint-零-" + "c" * 2048)
    left = _limited(first, "checkpoint-left")
    right = _limited(first, "checkpoint-right")

    try:
        async with repository, peer:
            assert await repository.get_head(run_id) is None
            await repository.commit(first)
            assert await peer.get_head(run_id) == first

            outcomes = await asyncio.gather(
                _capture_commit(repository, left),
                _capture_commit(peer, right),
            )
            assert outcomes.count(None) == 1
            failures = [outcome for outcome in outcomes if outcome is not None]
            assert len(failures) == 1
            assert isinstance(failures[0], RevisionConflict)

            head = await repository.get_head(run_id)
            assert head in (left, right)
            await repository.commit(first)

            assert head is not None
            collision = Checkpoint(first.id, head.snapshot, head.fact)
            with pytest.raises(RepositoryError, match="reused"):
                await repository.commit(collision)

            other_run = _started("other-run", first.id)
            with pytest.raises(RepositoryError, match="reused"):
                await peer.commit(other_run)

            stale = Checkpoint("checkpoint-stale", head.snapshot, head.fact)
            with pytest.raises(RevisionConflict) as raised:
                await peer.commit(stale)
            assert raised.value.expected_revision == 0
            assert raised.value.actual_revision == 1
    finally:
        await asyncio.to_thread(_drop_test_tables, table_prefix)
