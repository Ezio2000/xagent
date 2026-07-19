from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import pytest

from jharness.kernel import Message, RepositoryError, RevisionConflict
from jharness.kernel.wire import encode_message as wire_encode_message
from jharness.repository import _codec
from jharness.repository.sqlite import SQLiteRunRepository
from tests.repository_backends.support import (
    append_external,
    limited,
    replace_history,
    started,
)


async def _capture_commit(
    repository: SQLiteRunRepository,
    commit: object,
) -> BaseException | None:
    try:
        await repository.commit(commit)  # type: ignore[arg-type]
    except BaseException as exc:
        return exc
    return None


async def test_sqlite_context_closes_after_initialization_failure(tmp_path: Path) -> None:
    repository = SQLiteRunRepository(tmp_path / "runs.sqlite3")
    failure = sqlite3.OperationalError("injected initialization failure")

    def fail_initialization() -> None:
        raise failure

    object.__setattr__(repository, "_initialize_sync", fail_initialization)
    with pytest.raises(RepositoryError, match="operation failed") as raised:
        async with repository:
            raise AssertionError("initialization unexpectedly succeeded")
    assert raised.value.__cause__ is failure
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("run-after-failed-enter")
    await repository.close()


async def test_sqlite_is_lazy_and_persists_split_v2_state(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = started("run-a", "cp-0")
    advanced = append_external(first.checkpoint, "cp-1")
    repository = SQLiteRunRepository(database)
    assert not database.exists()

    async with repository:
        assert database.exists()
        assert await repository.get_head("run-a") is None
        await repository.commit(first)
        await repository.commit(advanced)

    async with SQLiteRunRepository(database) as reopened:
        head = await reopened.get_head("run-a")
        assert head == advanced.checkpoint
        assert head is not advanced.checkpoint
        await reopened.commit(first)

    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert tables == {
        "jharness_v2_run_heads",
        "jharness_v2_checkpoint_ledger",
        "jharness_v2_history_chunks",
    }


async def test_sqlite_idempotency_is_scoped_per_run_and_precedes_cas(
    tmp_path: Path,
) -> None:
    first = started("run-a", "shared-id")
    other = started("run-b", "shared-id")
    advanced = append_external(first.checkpoint, "cp-1")
    async with SQLiteRunRepository(tmp_path / "runs.sqlite3") as repository:
        await repository.commit(first)
        await repository.commit(other)
        await repository.commit(advanced)
        await repository.commit(first)

        collision = append_external(first.checkpoint, first.checkpoint_id, text="changed")
        with pytest.raises(RepositoryError, match="reused"):
            await repository.commit(collision)
        with pytest.raises(RevisionConflict) as raised:
            await repository.commit(append_external(first.checkpoint, "stale"))
        assert raised.value.expected_revision == 0
        assert raised.value.actual_revision == 1


async def test_sqlite_append_unchanged_and_retry_encode_only_the_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = wire_encode_message

    def counting(message: Message) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(message)

    monkeypatch.setattr(_codec, "encode_message", counting)

    def forbidden_decode(*args: object) -> None:
        del args
        raise AssertionError("commit path must not decode the previous core")

    monkeypatch.setattr("jharness.repository.sqlite.decode_core", forbidden_decode)
    first = started("run-a", "cp-0")
    appended = append_external(first.checkpoint, "cp-1")
    unchanged = limited(appended.checkpoint, "cp-2")
    async with SQLiteRunRepository(tmp_path / "runs.sqlite3") as repository:
        await repository.commit(first)
        assert calls == 1
        await repository.commit(appended)
        assert calls == 2
        await repository.commit(unchanged)
        assert calls == 2
        await repository.commit(appended)
        assert calls == 2


async def test_sqlite_rewrite_writes_only_new_generation_chunks(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = started("run-a", "cp-0")
    current = first.checkpoint
    commits = [first]
    for index in range(70):
        commit = append_external(current, f"cp-{index + 1}", text=str(index))
        commits.append(commit)
        current = commit.checkpoint
    replacement_messages = tuple(Message.user(str(index)) for index in range(65))
    replacement = replace_history(
        current,
        "cp-rewrite",
        messages=replacement_messages,
    )
    async with SQLiteRunRepository(database) as repository:
        for commit in commits:
            await repository.commit(commit)
        await repository.commit(replacement)
        assert await repository.get_head("run-a") == replacement.checkpoint

    with closing(sqlite3.connect(database)) as connection:
        generations = connection.execute(
            """
            SELECT history_generation, COUNT(*), SUM(message_count)
            FROM jharness_v2_history_chunks
            WHERE run_id = 'run-a'
            GROUP BY history_generation
            ORDER BY history_generation
            """
        ).fetchall()
    assert generations == [(0, 71, 71), (replacement.revision, 2, 65)]


async def test_sqlite_rejects_orphaned_ledger_and_corrupt_core(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = started("run-a", "cp-0")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)

    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DELETE FROM jharness_v2_run_heads WHERE run_id = 'run-a'")
        connection.commit()
    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="orphaned"):
            await repository.commit(first)

    database = tmp_path / "corrupt.sqlite3"
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)
    payload = b"{"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            UPDATE jharness_v2_run_heads
            SET checkpoint_core = ?, checkpoint_core_digest = ?
            WHERE run_id = 'run-a'
            """,
            (payload, sha256(payload).digest()),
        )
        connection.commit()
    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="core"):
            await repository.get_head("run-a")
        advanced = append_external(first.checkpoint, "cp-1")
        await repository.commit(advanced)
        assert await repository.get_head("run-a") == advanced.checkpoint


async def test_sqlite_detects_corrupt_history_when_materialized(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    first = started("run-a", "cp-0")
    async with SQLiteRunRepository(database) as repository:
        await repository.commit(first)
    payload = b"[]"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            """
            UPDATE jharness_v2_history_chunks
            SET chunk_payload = ?, chunk_digest = ?
            WHERE run_id = 'run-a' AND history_generation = 0 AND chunk_index = 0
            """,
            (payload, sha256(payload).digest()),
        )
        connection.commit()
    async with SQLiteRunRepository(database) as repository:
        with pytest.raises(RepositoryError, match="history chunk"):
            await repository.get_head("run-a")


async def test_sqlite_serializes_concurrent_instances(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    left = SQLiteRunRepository(database)
    right = SQLiteRunRepository(database)
    async with left, right:
        outcomes = await asyncio.gather(
            _capture_commit(left, started("run-a", "cp-left")),
            _capture_commit(right, started("run-a", "cp-right")),
        )
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RevisionConflict) for outcome in outcomes) == 1


async def test_sqlite_accepts_concurrent_exact_retries(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    commit = started("run-a", "cp-0")
    left = SQLiteRunRepository(database)
    right = SQLiteRunRepository(database)
    async with left, right:
        await asyncio.gather(left.commit(commit), right.commit(commit))
        assert await left.get_head("run-a") == commit.checkpoint


async def test_sqlite_settles_submitted_commit_after_cancellation(
    tmp_path: Path,
) -> None:
    database = tmp_path / "runs.sqlite3"
    repository = SQLiteRunRepository(database, timeout=2.0)
    await repository.initialize()
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        task = asyncio.create_task(repository.commit(started("run-a", "cp-0")))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        blocker.rollback()
        await task
        assert await repository.get_head("run-a") == started("run-a", "cp-0").checkpoint
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()
        await repository.close()


async def test_sqlite_cancels_a_commit_that_has_not_started(tmp_path: Path) -> None:
    database = tmp_path / "runs.sqlite3"
    repository = SQLiteRunRepository(database, timeout=2.0)
    await repository.initialize()
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        running = asyncio.create_task(repository.commit(started("run-a", "cp-a")))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(repository.commit(started("run-b", "cp-b")))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        blocker.rollback()
        await running
        assert await repository.get_head("run-a") is not None
        assert await repository.get_head("run-b") is None
    finally:
        if blocker.in_transaction:
            blocker.rollback()
        blocker.close()
        await repository.close()


async def test_sqlite_memory_database_and_closed_lifecycle() -> None:
    first = started("run-a", "cp-0")
    repository = SQLiteRunRepository(":memory:")
    async with repository:
        await repository.commit(first)
        assert await repository.get_head("run-a") == first.checkpoint
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("run-a")
