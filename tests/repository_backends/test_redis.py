from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
import redis.asyncio as redis_asyncio
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError as RedisResponseError

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
from jharness.repository.redis import (
    _COMMIT_SCRIPT,  # pyright: ignore[reportPrivateUsage]
    RedisRunRepository,
    _handle_commit_result,  # pyright: ignore[reportPrivateUsage]
)

_REDIS_URL_ENV = "JHARNESS_TEST_REDIS_URL"


class _BlockingRedisClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def ping(self) -> bool:
        return True

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | bytes | int,
    ) -> object:
        del script, numkeys, keys_and_args
        self.started.set()
        await self.release.wait()
        return [b"committed"]

    async def hmget(self, name: str, keys: list[str]) -> list[object]:
        del name
        return [None] * len(keys)

    async def aclose(self) -> None:
        self.closed = True


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


def _limited(
    previous: Checkpoint,
    checkpoint_id: str,
    *,
    revision: int = 1,
) -> Checkpoint:
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


def _repository(key_prefix: str) -> RedisRunRepository:
    raw_url = os.environ.get(_REDIS_URL_ENV)
    if raw_url is None:
        pytest.skip(f"set {_REDIS_URL_ENV} to run Redis integration tests")
    return RedisRunRepository(raw_url, key_prefix=key_prefix)


async def _capture_commit(
    repository: RedisRunRepository,
    checkpoint: Checkpoint,
) -> BaseException | None:
    try:
        await repository.commit(checkpoint)
    except BaseException as exc:
        return exc
    return None


def test_redis_commit_result_parser_preserves_repository_semantics() -> None:
    encoded = encode_checkpoint(_started("run-unit", "checkpoint-unit"))
    _handle_commit_result([b"committed"], encoded)
    _handle_commit_result([b"idempotent"], encoded)

    with pytest.raises(RevisionConflict) as raised:
        _handle_commit_result([b"revision_conflict", b"7"], encoded)
    assert raised.value.expected_revision is None
    assert raised.value.actual_revision == 7

    with pytest.raises(RepositoryError, match="reused"):
        _handle_commit_result([b"id_reused"], encoded)
    with pytest.raises(RepositoryError, match="invalid"):
        _handle_commit_result([b"ledger_corrupt"], encoded)
    with pytest.raises(RepositoryError, match="unknown"):
        _handle_commit_result([b"future_status"], encoded)


async def test_redis_uses_one_versioned_hash_with_bounded_field_names() -> None:
    repository = RedisRunRepository(key_prefix="unit-prefix")
    try:
        state_key = repository._state_key()  # pyright: ignore[reportPrivateUsage]
        checkpoint_prefix = repository._checkpoint_field_prefix(  # pyright: ignore[reportPrivateUsage]
            "checkpoint-" + "x" * 4096
        )
        head_prefix = repository._head_field_prefix(  # pyright: ignore[reportPrivateUsage]
            "run-" + "y" * 4096
        )
        assert state_key.startswith("jharness:{")
        assert state_key.endswith("}:v1:state")
        assert checkpoint_prefix.startswith("c:")
        assert head_prefix.startswith("r:")
        assert len(checkpoint_prefix.split(":")[1]) == 64
        assert len(head_prefix.split(":")[1]) == 64
        assert _COMMIT_SCRIPT.count("'HSET'") == 1
    finally:
        await repository.close()


async def test_redis_driver_is_loaded_only_when_the_backend_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []

    def missing_driver(name: str) -> object:
        imports.append(name)
        raise ModuleNotFoundError("No module named 'redis'", name="redis")

    monkeypatch.setattr("jharness.repository.redis.import_module", missing_driver)
    repository = RedisRunRepository(key_prefix="missing-driver-prefix")
    try:
        with pytest.raises(RepositoryError, match=r"jharness-repository\[redis\]"):
            await repository.initialize()
        assert imports == ["redis.asyncio"]
    finally:
        await repository.close()


async def test_redis_commit_retries_transport_errors_until_settled() -> None:
    class RetryingRedisClient(_BlockingRedisClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def eval(
            self,
            script: str,
            numkeys: int,
            *keys_and_args: str | bytes | int,
        ) -> object:
            del script, numkeys, keys_and_args
            self.attempts += 1
            if self.attempts == 1:
                raise RedisConnectionError("injected lost response")
            if self.attempts == 2:
                raise RedisResponseError("BUSY Redis is busy running a script")
            if self.attempts == 3:
                raise RedisConnectionError("injected reconnect failure")
            return [b"committed"]

    repository = RedisRunRepository(key_prefix="settlement-prefix")
    client = RetryingRedisClient()
    object.__setattr__(repository, "_client", client)
    try:
        await repository.commit(_started("run-settlement", "checkpoint-settlement"))
        assert client.attempts == 4
    finally:
        await repository.close()


async def test_redis_commit_settles_eval_after_cancellation() -> None:
    repository = RedisRunRepository(key_prefix="cancel-prefix")
    client = _BlockingRedisClient()
    object.__setattr__(repository, "_client", client)
    commit = asyncio.create_task(repository.commit(_started("run-cancel", "checkpoint-cancel")))
    try:
        await asyncio.wait_for(client.started.wait(), 1)
        commit.cancel()
        await asyncio.sleep(0)
        assert not commit.done()
        client.release.set()
        await asyncio.wait_for(commit, 1)
    finally:
        client.release.set()
        await repository.close()
    assert client.closed


@pytest.mark.skipif(
    _REDIS_URL_ENV not in os.environ,
    reason=f"set {_REDIS_URL_ENV} to run Redis integration tests",
)
async def test_redis_repository_atomic_cas_and_global_id_ledger() -> None:
    raw_url = os.environ[_REDIS_URL_ENV]
    key_prefix = f"jharness-test-{uuid4().hex}"
    repository = _repository(key_prefix)
    peer = _repository(key_prefix)
    run_id = "run-长-" + "r" * 2048
    first = _started(run_id, "checkpoint-零-" + "c" * 2048)
    left = _limited(first, "checkpoint-left")
    right = _limited(first, "checkpoint-right")
    state_key = repository._state_key()  # pyright: ignore[reportPrivateUsage]
    raw_client = redis_asyncio.from_url(raw_url, decode_responses=False)
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

            head_prefix = repository._head_field_prefix(run_id)  # pyright: ignore[reportPrivateUsage]
            await raw_client.hset(
                state_key,
                f"{head_prefix}checkpoint_payload",
                b"corrupt-payload",
            )
            next_checkpoint = _limited(
                head,
                "checkpoint-after-corruption",
                revision=2,
            )
            with pytest.raises(RepositoryError, match="invalid"):
                await repository.commit(next_checkpoint)
            next_prefix = repository._checkpoint_field_prefix(  # pyright: ignore[reportPrivateUsage]
                next_checkpoint.id
            )
            assert not await raw_client.hexists(
                state_key,
                f"{next_prefix}checkpoint_id",
            )
    finally:
        try:
            await raw_client.delete(state_key)
            assert not await raw_client.exists(state_key)
        finally:
            await raw_client.aclose()
