from __future__ import annotations

import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from weakref import ref

import pytest

from jharness.kernel import Checkpoint, DurableCommit, RepositoryError, RevisionConflict
from jharness.repository.memory import MemoryRunRepository
from tests.repository_backends.support import append_external, limited, started


async def test_memory_supports_multiple_runs_and_per_run_checkpoint_ids() -> None:
    first = started("run-a", "shared-id")
    other = started("run-b", "shared-id")
    advanced = append_external(first.checkpoint, "cp-a1")
    repository = MemoryRunRepository()

    await repository.commit(first)
    await repository.commit(other)
    await repository.commit(advanced)

    first_head = await repository.get_head("run-a")
    other_head = await repository.get_head("run-b")
    assert first_head == advanced.checkpoint
    assert first_head is not advanced.checkpoint
    assert first_head is not None
    assert first_head.snapshot.history is advanced.checkpoint.snapshot.history
    assert other_head == other.checkpoint
    assert await repository.get_head("missing") is None


async def test_memory_checks_exact_retry_before_revision_cas() -> None:
    first = started("run-a", "cp-0")
    advanced = append_external(first.checkpoint, "cp-1")
    repository = MemoryRunRepository()
    await repository.commit(first)
    await repository.commit(advanced)

    await repository.commit(first)

    collision = append_external(first.checkpoint, first.checkpoint_id, text="changed")
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(collision)
    stale = append_external(first.checkpoint, "cp-stale")
    with pytest.raises(RevisionConflict) as raised:
        await repository.commit(stale)
    assert raised.value.expected_revision == 0
    assert raised.value.actual_revision == 1


async def test_memory_rejects_parent_or_history_base_mismatch() -> None:
    first = started("run-a", "cp-0")
    advanced = append_external(first.checkpoint, "cp-1")
    repository = MemoryRunRepository()
    await repository.commit(first)

    object.__setattr__(advanced, "parent_checkpoint_id", "another-parent")
    with pytest.raises(RepositoryError, match="parent"):
        await repository.commit(advanced)

    advanced = append_external(first.checkpoint, "cp-2")
    object.__setattr__(advanced.history, "base_digest", bytes(32))
    with pytest.raises(RepositoryError, match="history change base"):
        await repository.commit(advanced)


async def test_memory_never_calls_incremental_or_json_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("memory repository must not serialize")

    monkeypatch.setattr("jharness.repository._codec.encode_core", forbidden)
    monkeypatch.setattr("jharness.repository._codec.encode_history_change", forbidden)
    monkeypatch.setattr("jharness.repository._codec.json.dumps", forbidden)
    first = started("run-a", "cp-0")
    advanced = limited(first.checkpoint, "cp-1")
    repository = MemoryRunRepository()

    await repository.commit(first)
    await repository.commit(advanced)
    await repository.commit(first)
    assert await repository.get_head("run-a") == advanced.checkpoint


async def test_memory_does_not_retain_checkpoint_envelope() -> None:
    class WeakCheckpoint(Checkpoint):
        pass

    base = started("run-a", "cp-0")
    weak_checkpoint = WeakCheckpoint(
        base.checkpoint.id,
        base.checkpoint.snapshot,
        base.checkpoint.fact,
    )
    checkpoint_ref = ref(weak_checkpoint)
    commit = DurableCommit(weak_checkpoint, None, base.history)
    repository = MemoryRunRepository()
    await repository.commit(commit)

    del commit
    del weak_checkpoint
    gc.collect()

    assert checkpoint_ref() is None
    assert await repository.get_head("run-a") == base.checkpoint


async def test_memory_rolls_back_a_partially_mutated_head_mapping() -> None:
    class FailAfterSet(dict[str, object]):
        def __setitem__(self, key: str, value: object) -> None:
            super().__setitem__(key, value)
            raise RuntimeError("injected head write failure")

    repository = MemoryRunRepository()
    repository._heads = FailAfterSet()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match="injected"):
        await repository.commit(started("run-a", "cp-0"))

    assert await repository.get_head("run-a") is None
    assert repository._ledger == {}  # pyright: ignore[reportPrivateUsage]


def test_memory_serializes_concurrent_first_writers() -> None:
    repository = MemoryRunRepository()
    commits = (started("run-a", "cp-left"), started("run-a", "cp-right"))

    def commit(value: DurableCommit) -> BaseException | None:
        try:
            asyncio.run(repository.commit(value))
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(commit, commits))

    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RevisionConflict) for outcome in outcomes) == 1


async def test_memory_normalizes_hostile_string_identifiers() -> None:
    class HostileString(str):
        def __hash__(self) -> int:
            raise AssertionError("repository must not hash an untrusted str subclass")

        def __str__(self) -> str:
            return "redirected"

    repository = MemoryRunRepository()
    commit = started(HostileString("run-a"), HostileString("cp-0"))
    await repository.commit(commit)
    assert await repository.get_head(HostileString("run-a")) == commit.checkpoint


@pytest.mark.parametrize("run_id", ["", 1, None])
async def test_memory_validates_head_run_id(run_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        await MemoryRunRepository().get_head(run_id)  # type: ignore[arg-type]
