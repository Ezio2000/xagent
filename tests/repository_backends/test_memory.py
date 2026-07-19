from __future__ import annotations

import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor
from weakref import ref

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
    decode_checkpoint,
)
from jharness.repository.memory import MemoryRunRepository


def _started(
    run_id: str,
    checkpoint_id: str,
    *,
    metadata: dict[str, object] | None = None,
) -> Checkpoint:
    context = RunContext(run_id, 1.0, metadata={} if metadata is None else metadata)
    snapshot = RunSnapshot(0, context, (Message.user("hello"),), RunMetrics(), Planning())
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


async def test_memory_repository_supports_multiple_runs_and_head_reads() -> None:
    first = _started("run-a", "cp-a0")
    other = _started("run-b", "cp-b0")
    advanced = _limited(first, "cp-a1")
    repository = MemoryRunRepository()

    await repository.commit(first)
    await repository.commit(other)
    await repository.commit(advanced)

    first_head = await repository.get_head("run-a")
    other_head = await repository.get_head("run-b")
    assert first_head == advanced
    assert first_head is not advanced
    assert other_head == other
    assert await repository.get_head("missing") is None


async def test_memory_repository_checks_exact_retry_before_revision_cas() -> None:
    first = _started("run-a", "cp-0")
    advanced = _limited(first, "cp-1")
    repository = MemoryRunRepository()
    await repository.commit(first)
    await repository.commit(advanced)

    await repository.commit(first)

    collision = Checkpoint(first.id, advanced.snapshot, advanced.fact)
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(collision)
    stale = _limited(first, "cp-stale")
    with pytest.raises(RevisionConflict) as raised:
        await repository.commit(stale)
    assert raised.value.expected_revision == 0
    assert raised.value.actual_revision == 1


async def test_memory_repository_checkpoint_ids_are_global() -> None:
    repository = MemoryRunRepository()
    await repository.commit(_started("run-a", "shared-id"))

    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(_started("run-b", "shared-id"))


async def test_memory_repository_canonical_encoding_is_order_stable_and_type_strict() -> None:
    repository = MemoryRunRepository()
    original = _started(
        "run-a",
        "cp-0",
        metadata={"nested": {"second": 2, "first": 1}, "values": [False, 1, 1.5]},
    )
    reordered = _started(
        "run-a",
        "cp-0",
        metadata={"values": [False, 1, 1.5], "nested": {"first": 1, "second": 2}},
    )
    changed_type = _started(
        "run-a",
        "cp-0",
        metadata={"nested": {"first": 1, "second": 2}, "values": [False, 1.0, 1.5]},
    )

    await repository.commit(original)
    await repository.commit(reordered)
    with pytest.raises(RepositoryError, match="reused"):
        await repository.commit(changed_type)


async def test_memory_repository_does_not_retain_checkpoint_objects() -> None:
    class WeakCheckpoint(Checkpoint):
        pass

    base = _started("run-a", "cp-0")
    checkpoint = WeakCheckpoint(base.id, base.snapshot, base.fact)
    checkpoint_ref = ref(checkpoint)
    repository = MemoryRunRepository()
    await repository.commit(checkpoint)

    del checkpoint
    gc.collect()

    assert checkpoint_ref() is None
    assert await repository.get_head("run-a") == base


async def test_memory_repository_rejects_nonconsecutive_first_commit() -> None:
    first = _started("run-a", "cp-0")
    with pytest.raises(RevisionConflict) as raised:
        await MemoryRunRepository().commit(_limited(first, "cp-1"))
    assert raised.value.actual_revision is None


def test_checkpoint_decoder_wraps_excessive_nesting_as_repository_error() -> None:
    payload = b"[" * 2_000 + b"]" * 2_000

    with pytest.raises(RepositoryError, match="payload is invalid"):
        decode_checkpoint(payload)


async def test_memory_repository_normalizes_string_subclass_identifiers() -> None:
    class HostileString(str):
        def __hash__(self) -> int:
            raise AssertionError("repository must not hash an untrusted str subclass")

        def __str__(self) -> str:
            return "redirected-by-hostile-str"

    checkpoint = _started(HostileString("run-a"), HostileString("cp-0"))
    repository = MemoryRunRepository()

    await repository.commit(checkpoint)

    assert await repository.get_head(HostileString("run-a")) == checkpoint


async def test_memory_repository_rolls_back_a_partially_mutated_mapping() -> None:
    class FailAfterSet(dict[str, object]):
        def __setitem__(self, key: str, value: object) -> None:
            super().__setitem__(key, value)
            raise RuntimeError("injected head write failure")

    repository = MemoryRunRepository()
    repository._heads = FailAfterSet()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(RuntimeError, match="injected"):
        await repository.commit(_started("run-a", "cp-0"))

    assert await repository.get_head("run-a") is None
    assert repository._by_id == {}  # pyright: ignore[reportPrivateUsage]


def test_memory_repository_serializes_concurrent_first_writers() -> None:
    repository = MemoryRunRepository()
    checkpoints = (_started("run-a", "cp-left"), _started("run-a", "cp-right"))

    def commit(checkpoint: Checkpoint) -> BaseException | None:
        try:
            asyncio.run(repository.commit(checkpoint))
        except BaseException as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(commit, checkpoints))

    assert sum(outcome is None for outcome in outcomes) == 1
    conflicts = tuple(outcome for outcome in outcomes if isinstance(outcome, RevisionConflict))
    assert len(conflicts) == 1


@pytest.mark.parametrize("run_id", ["", 1, None])
async def test_memory_repository_validates_head_run_id(run_id: object) -> None:
    repository = MemoryRunRepository()
    with pytest.raises((TypeError, ValueError)):
        await repository.get_head(run_id)  # type: ignore[arg-type]
