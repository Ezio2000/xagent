from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

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
from jharness.repository.sqlite import SQLiteRunRepository


class _HostileString(str):
    def __str__(self) -> str:
        return "hostile"


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


def _limited(previous: Checkpoint, checkpoint_id: str, *, revision: int = 1) -> Checkpoint:
    snapshot = RunSnapshot(
        revision,
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


async def _capture_commit(
    repository: SQLiteRunRepository,
    checkpoint: Checkpoint,
) -> BaseException | None:
    try:
        await repository.commit(checkpoint)
    except BaseException as exc:
        return exc
    return None


def _corrupt_head(database: Path, corruption: str) -> None:
    with closing(sqlite3.connect(database)) as connection:
        if corruption == "checkpoint_id":
            connection.execute(
                """
                UPDATE jharness_v1_run_heads
                SET checkpoint_id = 'missing-checkpoint'
                WHERE run_id = 'run-a'
                """
            )
        elif corruption == "revision":
            connection.execute(
                """
                UPDATE jharness_v1_run_heads
                SET revision = 9
                WHERE run_id = 'run-a'
                """
            )
        elif corruption == "digest":
            connection.execute(
                """
                UPDATE jharness_v1_run_heads
                SET checkpoint_digest = ?
                WHERE run_id = 'run-a'
                """,
                (bytes(32),),
            )
        elif corruption == "payload":
            payload = b"{"
            digest = sha256(payload).digest()
            connection.execute(
                """
                UPDATE jharness_v1_run_heads
                SET checkpoint_digest = ?, checkpoint_payload = ?
                WHERE run_id = 'run-a'
                """,
                (digest, payload),
            )
            connection.execute(
                """
                UPDATE jharness_v1_checkpoint_ids
                SET checkpoint_digest = ?
                WHERE checkpoint_id = 'cp-0'
                """,
                (digest,),
            )
        elif corruption == "ledger":
            connection.execute(
                """
                UPDATE jharness_v1_checkpoint_ids
                SET run_id = 'run-b'
                WHERE checkpoint_id = 'cp-0'
                """
            )
        elif corruption == "decoded_checkpoint":
            other = connection.execute(
                """
                SELECT checkpoint_digest, checkpoint_payload
                FROM jharness_v1_run_heads
                WHERE run_id = 'run-b'
                """
            ).fetchone()
            assert other is not None
            connection.execute(
                """
                UPDATE jharness_v1_run_heads
                SET checkpoint_digest = ?, checkpoint_payload = ?
                WHERE run_id = 'run-a'
                """,
                other,
            )
            connection.execute(
                """
                UPDATE jharness_v1_checkpoint_ids
                SET checkpoint_digest = ?
                WHERE checkpoint_id = 'cp-0'
                """,
                (other[0],),
            )
        else:
            raise AssertionError(f"unknown corruption: {corruption}")
        connection.commit()


def _head_row(database: Path) -> tuple[object, ...]:
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            """
            SELECT checkpoint_id, revision, checkpoint_digest, checkpoint_payload
            FROM jharness_v1_run_heads
            WHERE run_id = 'run-a'
            """
        ).fetchone()
    assert row is not None
    return row


async def test_sqlite_context_closes_after_initialization_failure(tmp_path: Path) -> None:
    repository = SQLiteRunRepository(tmp_path / "runs.sqlite3")
    failure = sqlite3.OperationalError("injected initialization failure")

    def fail_initialization() -> None:
        raise failure

    object.__setattr__(repository, "_initialize_sync", fail_initialization)
    try:
        with pytest.raises(RepositoryError, match="operation failed") as raised:
            async with repository:
                raise AssertionError("initialization unexpectedly succeeded")
        assert raised.value.__cause__ is failure
        with pytest.raises(RepositoryError, match="closed"):
            await repository.get_head("run-after-failed-enter")
    finally:
        await repository.close()


async def test_sqlite_repository_is_lazy_and_persists_across_instances(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = _started("run-a", "cp-0")
    advanced = _limited(first, "cp-1")
    repository = SQLiteRunRepository(database)
    assert not database.exists()

    async with repository:
        assert database.exists()
        assert await repository.get_head("run-a") is None
        await repository.commit(first)
        await repository.commit(advanced)

    async with SQLiteRunRepository(database) as reopened:
        head = await reopened.get_head("run-a")
        assert head == advanced
        assert head is not advanced
        await reopened.commit(first)


async def test_sqlite_repository_normalizes_string_subclasses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "runs.sqlite3"
    first = _started("run-a", "cp-0")

    async with SQLiteRunRepository(_HostileString(str(database))) as repository:
        await repository.commit(first)
        assert await repository.get_head(_HostileString("run-a")) == first

    assert database.exists()
    assert not (tmp_path / "hostile").exists()


async def test_sqlite_repository_checks_exact_retry_before_revision_cas(
    tmp_path: Path,
) -> None:
    first = _started("run-a", "cp-0")
    advanced = _limited(first, "cp-1")
    async with SQLiteRunRepository(tmp_path / "runs.sqlite3") as repository:
        await repository.commit(first)
        await repository.commit(advanced)

        await repository.commit(first)

        collision = Checkpoint(first.id, advanced.snapshot, advanced.fact)
        with pytest.raises(RepositoryError, match="reused"):
            await repository.commit(collision)
        with pytest.raises(RevisionConflict) as raised:
            await repository.commit(_limited(first, "cp-stale"))
        assert raised.value.expected_revision == 0
        assert raised.value.actual_revision == 1


async def test_sqlite_exact_retry_rejects_same_revision_with_another_head(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    current = _started("run-a", "cp-current")
    retried = _started("run-a", "cp-retried")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(current)

    encoded = encode_checkpoint(retried)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            INSERT INTO jharness_v1_checkpoint_ids (
                checkpoint_id, run_id, revision, checkpoint_digest
            ) VALUES (?, ?, ?, ?)
            """,
            (
                encoded.checkpoint_id,
                encoded.run_id,
                encoded.revision,
                encoded.digest,
            ),
        )
        connection.commit()

    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="orphaned"):
            await repository.commit(retried)
        assert await repository.get_head("run-a") == current


async def test_sqlite_repository_checkpoint_ids_are_global(tmp_path: Path) -> None:
    async with SQLiteRunRepository(tmp_path / "runs.sqlite3") as repository:
        await repository.commit(_started("run-a", "shared-id"))
        with pytest.raises(RepositoryError, match="reused"):
            await repository.commit(_started("run-b", "shared-id"))


async def test_sqlite_repository_rejects_an_orphaned_idempotency_entry(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    first = _started("run-a", "cp-0")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)

    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM jharness_v1_run_heads WHERE run_id = ?", ("run-a",))
        connection.commit()

    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="orphaned"):
            await repository.commit(first)


async def test_sqlite_repository_cross_checks_head_and_idempotency_ledger(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    first = _started("run-a", "cp-0")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)

    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "DELETE FROM jharness_v1_checkpoint_ids WHERE checkpoint_id = ?",
            (first.id,),
        )
        connection.commit()

    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="no checkpoint ledger"):
            await repository.get_head("run-a")


@pytest.mark.parametrize("commit_kind", ["exact_retry", "new_revision"])
@pytest.mark.parametrize(
    "corruption",
    [
        "checkpoint_id",
        "revision",
        "digest",
        "payload",
        "ledger",
        "decoded_checkpoint",
    ],
)
async def test_sqlite_commit_rejects_a_corrupt_current_head(
    tmp_path: Path,
    commit_kind: str,
    corruption: str,
) -> None:
    database = tmp_path / "runs.sqlite3"
    first = _started("run-a", "cp-0")
    other = _started("run-b", "cp-other")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)
        await repository.commit(other)

    _corrupt_head(database, corruption)
    corrupt_row = _head_row(database)
    attempted = first if commit_kind == "exact_retry" else _limited(first, "cp-1")
    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError) as raised:
            await repository.commit(attempted)
        assert not isinstance(raised.value, RevisionConflict)

    assert _head_row(database) == corrupt_row


async def test_sqlite_repository_serializes_concurrent_instances(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    left = SQLiteRunRepository(database)
    right = SQLiteRunRepository(database)
    async with left, right:
        outcomes = await asyncio.gather(
            _capture_commit(left, _started("run-a", "cp-left")),
            _capture_commit(right, _started("run-a", "cp-right")),
        )

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RevisionConflict) for outcome in outcomes) == 1


async def test_sqlite_repository_accepts_concurrent_exact_retries(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    checkpoint = _started("run-a", "cp-0")
    left = SQLiteRunRepository(database)
    right = SQLiteRunRepository(database)
    async with left, right:
        await asyncio.gather(left.commit(checkpoint), right.commit(checkpoint))
        assert await left.get_head("run-a") == checkpoint


async def test_sqlite_repository_settles_a_submitted_commit_after_cancellation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    repository = SQLiteRunRepository(database, timeout=2.0)
    await repository.initialize()
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        commit = asyncio.create_task(repository.commit(_started("run-a", "cp-0")))
        await asyncio.sleep(0.05)
        commit.cancel()
        await asyncio.sleep(0)
        assert not commit.done()
        blocker.rollback()
        await commit
        assert await repository.get_head("run-a") == _started("run-a", "cp-0")
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()
        await repository.close()


async def test_sqlite_repository_cancels_a_commit_that_has_not_started(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    repository = SQLiteRunRepository(database, timeout=2.0)
    await repository.initialize()
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        running = asyncio.create_task(repository.commit(_started("run-a", "cp-a")))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(repository.commit(_started("run-b", "cp-b")))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued

        blocker.rollback()
        await running
        assert await repository.get_head("run-a") == _started("run-a", "cp-a")
        assert await repository.get_head("run-b") is None
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()
        await repository.close()


async def test_sqlite_repository_rejects_nonconsecutive_first_commit(tmp_path: Path) -> None:
    first = _started("run-a", "cp-0")
    async with SQLiteRunRepository(tmp_path / "runs.sqlite3") as repository:
        with pytest.raises(RevisionConflict) as raised:
            await repository.commit(_limited(first, "cp-1"))
        assert raised.value.actual_revision is None


async def test_sqlite_repository_keeps_an_in_memory_database_for_its_lifetime() -> None:
    first = _started("run-a", "cp-0")
    async with SQLiteRunRepository(":memory:") as repository:
        await repository.commit(first)
        assert await repository.get_head("run-a") == first


async def test_sqlite_repository_rejects_operations_after_close(tmp_path: Path) -> None:
    repository = SQLiteRunRepository(tmp_path / "runs.sqlite3")
    await repository.close()
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("run-a")
