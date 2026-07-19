from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
import redis.asyncio as redis_asyncio
from redis.exceptions import ClusterError as RedisClusterError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import InvalidResponse as RedisInvalidResponse
from redis.exceptions import RedisClusterException, RedisError
from redis.exceptions import ResponseError as RedisResponseError

from jharness.kernel import (
    Checkpoint,
    ConversationInsertFact,
    DurableCommit,
    HistoryAppend,
    Message,
    Planning,
    RepositoryError,
    RevisionConflict,
    RunHistory,
    RunSnapshot,
)
from jharness.repository._codec import (  # pyright: ignore[reportPrivateUsage]
    EncodedHistoryChunk,
    commit_identity,
    encode_core,
    encode_history_change,
)
from jharness.repository.redis import (
    _COMMIT_SCRIPT,  # pyright: ignore[reportPrivateUsage]
    _PROBE_SCRIPT,  # pyright: ignore[reportPrivateUsage]
    RedisRunRepository,
    _checkpoint_field_prefix,  # pyright: ignore[reportPrivateUsage]
    _handle_commit_result,  # pyright: ignore[reportPrivateUsage]
    _handle_probe_result,  # pyright: ignore[reportPrivateUsage]
    _history_chunk_prefix,  # pyright: ignore[reportPrivateUsage]
)
from tests.repository_backends.support import append_external, started

_REDIS_URL_ENV = "JHARNESS_TEST_REDIS_URL"
_REDIS_CLUSTER_URL_ENV = "JHARNESS_TEST_REDIS_CLUSTER_URL"


class _ScriptedRedisClient:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.eval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def ping(self) -> bool:
        return True

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | bytes | int,
    ) -> object:
        assert numkeys == 3
        self.eval_calls.append((script, tuple(keys_and_args)))
        if not self.outcomes:
            raise AssertionError("test client has no scripted outcome")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def hmget(self, name: str, keys: list[str]) -> list[object]:
        del name
        return [None] * len(keys)

    async def aclose(self) -> None:
        self.closed = True


class _BlockingRedisClient(_ScriptedRedisClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = 0

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | bytes | int,
    ) -> object:
        del numkeys, keys_and_args
        self.attempts += 1
        if script == _PROBE_SCRIPT:
            return [b"new"]
        self.started.set()
        await self.release.wait()
        return [b"committed"]


class _RecordingRedisFactory:
    def __init__(self, client: _ScriptedRedisClient) -> None:
        self.client = client
        self.calls: list[tuple[str, bool, float, float, int]] = []

    def from_url(
        self,
        url: str,
        *,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        health_check_interval: int,
    ) -> _ScriptedRedisClient:
        self.calls.append(
            (
                url,
                decode_responses,
                socket_connect_timeout,
                socket_timeout,
                health_check_interval,
            )
        )
        return self.client


def _repository(key_prefix: str) -> RedisRunRepository:
    raw_url = os.environ.get(_REDIS_URL_ENV)
    if raw_url is None:
        pytest.skip(f"set {_REDIS_URL_ENV} to run Redis integration tests")
    return RedisRunRepository(raw_url, key_prefix=key_prefix)


async def _capture_commit(
    repository: RedisRunRepository,
    commit: DurableCommit,
) -> BaseException | None:
    try:
        await repository.commit(commit)
    except BaseException as exc:
        return exc
    return None


def _append_many(previous: Checkpoint, checkpoint_id: str, count: int) -> DurableCommit:
    messages = tuple(Message.external(f"message-{index}") for index in range(count))
    before = previous.snapshot.history
    history = RunHistory((*before, *messages))
    snapshot = RunSnapshot(
        previous.snapshot.revision + 1,
        previous.snapshot.context,
        history,
        previous.snapshot.metrics,
        Planning(),
    )
    checkpoint = Checkpoint(
        checkpoint_id,
        snapshot,
        ConversationInsertFact(2.0 + previous.snapshot.revision, "fault-injection"),
    )
    return DurableCommit(
        checkpoint,
        previous.id,
        HistoryAppend(len(before), before.digest, messages),
    )


def test_redis_result_parsers_preserve_repository_semantics() -> None:
    identity = commit_identity(started("run-a", "cp-0"))
    assert _handle_probe_result([b"idempotent"], identity) is True
    assert _handle_probe_result([b"new"], identity) is False
    _handle_commit_result([b"committed"], identity)
    _handle_commit_result([b"idempotent"], identity)
    with pytest.raises(RevisionConflict) as raised:
        _handle_probe_result([b"revision_conflict", b"7"], identity)
    assert raised.value.actual_revision == 7
    with pytest.raises(RepositoryError, match="reused"):
        _handle_commit_result([b"id_reused"], identity)
    with pytest.raises(RepositoryError, match="invalid"):
        _handle_commit_result([b"head_corrupt"], identity)
    with pytest.raises(RepositoryError, match="unknown"):
        _handle_commit_result([b"future_status"], identity)


async def test_redis_uses_three_v2_keys_per_run_in_one_cluster_slot() -> None:
    repository = RedisRunRepository(key_prefix="unit-prefix")
    try:
        first = repository._run_keys("run-a")  # pyright: ignore[reportPrivateUsage]
        second = repository._run_keys("run-b")  # pyright: ignore[reportPrivateUsage]
        assert len(first) == 3
        assert all(key.startswith("jharness:v2:{") for key in first)
        assert tuple(key.rsplit(":", 1)[1] for key in first) == (
            "head",
            "ledger",
            "history",
        )
        first_tags = {key.split("{", 1)[1].split("}", 1)[0] for key in first}
        second_tags = {key.split("{", 1)[1].split("}", 1)[0] for key in second}
        assert len(first_tags) == 1
        assert len(second_tags) == 1
        assert first_tags != second_tags
        prefix = _checkpoint_field_prefix("checkpoint-" + "x" * 4096)
        assert prefix.startswith("c:") and len(prefix.split(":")[1]) == 40
        assert "_v1_" not in _PROBE_SCRIPT + _COMMIT_SCRIPT
        assert "'checkpoint_core'," not in _PROBE_SCRIPT
    finally:
        await repository.close()


@pytest.mark.parametrize("cluster", [False, True])
async def test_redis_driver_is_lazy_and_failed_enter_closes_repository(
    monkeypatch: pytest.MonkeyPatch,
    cluster: bool,
) -> None:
    imports: list[str] = []

    def missing_driver(name: str) -> object:
        imports.append(name)
        raise ModuleNotFoundError("No module named 'redis'", name="redis")

    monkeypatch.setattr("jharness.repository.redis.import_module", missing_driver)
    repository = RedisRunRepository(key_prefix="missing-driver", cluster=cluster)
    with pytest.raises(RepositoryError, match=r"jharness-repository\[redis\]"):
        async with repository:
            raise AssertionError("initialization unexpectedly succeeded")
    assert imports == ["redis.asyncio"]
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("after-failure")
    await repository.close()


@pytest.mark.parametrize("cluster", [None, 0, 1, "yes"])
def test_redis_cluster_flag_requires_an_actual_boolean(cluster: object) -> None:
    with pytest.raises(TypeError, match="cluster must be a boolean"):
        RedisRunRepository(cluster=cluster)  # pyright: ignore[reportArgumentType]


async def test_redis_selects_factory_lazily_and_forwards_connection_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_client = _ScriptedRedisClient()
    cluster_client = _ScriptedRedisClient()
    standalone = _RecordingRedisFactory(standalone_client)
    cluster = _RecordingRedisFactory(cluster_client)
    module = SimpleNamespace(from_url=standalone.from_url, RedisCluster=cluster)
    exceptions = SimpleNamespace(
        RedisError=RedisError,
        RedisClusterException=RedisClusterException,
        ResponseError=RedisResponseError,
    )
    imports: list[str] = []

    def import_driver(name: str) -> object:
        imports.append(name)
        if name == "redis.asyncio":
            return module
        if name == "redis.exceptions":
            return exceptions
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("jharness.repository.redis.import_module", import_driver)
    direct = RedisRunRepository(
        "redis://standalone.internal:6379/4",
        socket_connect_timeout=1.5,
        socket_timeout=2.5,
        health_check_interval=7,
    )
    clustered = RedisRunRepository(
        "redis://cluster.internal:6379",
        cluster=True,
        socket_connect_timeout=3.5,
        socket_timeout=4.5,
        health_check_interval=11,
    )
    assert imports == []
    try:
        await direct.initialize()
        assert standalone.calls == [("redis://standalone.internal:6379/4", False, 1.5, 2.5, 7)]
        assert cluster.calls == []
        assert imports == ["redis.asyncio", "redis.exceptions"]

        await clustered.initialize()
        assert cluster.calls == [("redis://cluster.internal:6379", False, 3.5, 4.5, 11)]
        assert imports == [
            "redis.asyncio",
            "redis.exceptions",
            "redis.asyncio",
            "redis.exceptions",
        ]
    finally:
        await direct.close()
        await clustered.close()
    assert standalone_client.closed
    assert cluster_client.closed


async def test_redis_initialization_failure_closes_created_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient(_ScriptedRedisClient):
        async def ping(self) -> bool:
            raise RedisConnectionError("ping failed")

    class FailingModule:
        def from_url(self, *args: object, **kwargs: object) -> FailingClient:
            del args, kwargs
            return client

    client = FailingClient()
    module = FailingModule()
    exceptions = SimpleNamespace(
        RedisError=RedisError,
        RedisClusterException=RedisClusterException,
        ResponseError=RedisResponseError,
    )
    runtime = SimpleNamespace(standalone=module, cluster=module, exceptions=exceptions)
    monkeypatch.setattr("jharness.repository.redis._load_redis", lambda: runtime)
    repository = RedisRunRepository(key_prefix="init-failure")

    with pytest.raises(RepositoryError, match="initialization"):
        async with repository:
            raise AssertionError("initialization unexpectedly succeeded")
    assert client.closed
    with pytest.raises(RepositoryError, match="closed"):
        await repository.get_head("after-failure")


async def test_redis_exact_retry_stops_after_probe_without_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RedisRunRepository(key_prefix="exact-retry")
    client = _ScriptedRedisClient([b"idempotent"])
    object.__setattr__(repository, "_client", client)

    def forbidden(_: object) -> None:
        raise AssertionError("exact retry must not encode payloads")

    monkeypatch.setattr("jharness.repository.redis.encode_core", forbidden)
    monkeypatch.setattr("jharness.repository.redis.encode_history_change", forbidden)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert len(client.eval_calls) == 1
        assert client.eval_calls[0][0] == _PROBE_SCRIPT
    finally:
        await repository.close()


async def test_redis_prewrite_probe_transport_failure_returns_promptly() -> None:
    client = _ScriptedRedisClient(RedisConnectionError("probe unavailable"))
    repository = RedisRunRepository(key_prefix="probe-failure")
    object.__setattr__(repository, "_client", client)
    try:
        with pytest.raises(RepositoryError, match="commit failed") as raised:
            await repository.commit(started("run-a", "cp-0"))
        assert isinstance(raised.value.__cause__, RedisConnectionError)
        assert len(client.eval_calls) == 1
        assert client.eval_calls[0][0] == _PROBE_SCRIPT
    finally:
        await repository.close()


async def test_redis_prewrite_invalid_response_returns_promptly() -> None:
    client = _ScriptedRedisClient(RedisInvalidResponse("malformed probe response"))
    repository = RedisRunRepository(key_prefix="prewrite-invalid-response")
    object.__setattr__(repository, "_client", client)
    try:
        with pytest.raises(RepositoryError, match="commit failed") as raised:
            await repository.commit(started("run-a", "cp-0"))
        assert isinstance(raised.value.__cause__, RedisInvalidResponse)
        assert tuple(call[0] for call in client.eval_calls) == (_PROBE_SCRIPT,)
    finally:
        await repository.close()


async def test_redis_prewrite_probe_response_failure_returns_promptly() -> None:
    client = _ScriptedRedisClient(RedisResponseError("MOVED 1 redis.internal:6379"))
    repository = RedisRunRepository(key_prefix="probe-response-failure")
    object.__setattr__(repository, "_client", client)
    try:
        with pytest.raises(RepositoryError, match="commit failed") as raised:
            await repository.commit(started("run-a", "cp-0"))
        assert isinstance(raised.value.__cause__, RedisResponseError)
        assert len(client.eval_calls) == 1
        assert client.eval_calls[0][0] == _PROBE_SCRIPT
    finally:
        await repository.close()


@pytest.mark.parametrize("cluster", [False, True])
async def test_redis_write_invalid_response_is_settled(
    monkeypatch: pytest.MonkeyPatch,
    cluster: bool,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisInvalidResponse("malformed write response"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(
        key_prefix="write-invalid-response",
        cluster=cluster,
    )
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


async def test_redis_unknown_probe_invalid_response_keeps_settling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisConnectionError("lost commit response"),
        RedisInvalidResponse("malformed settlement response"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="probe-invalid-response")
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


@pytest.mark.parametrize("failure_type", [RedisClusterError, RedisClusterException])
async def test_redis_prewrite_cluster_failure_returns_promptly(
    failure_type: type[Exception],
) -> None:
    client = _ScriptedRedisClient(failure_type("cluster routing unavailable"))
    repository = RedisRunRepository(key_prefix="prewrite-cluster-failure", cluster=True)
    object.__setattr__(repository, "_client", client)
    try:
        with pytest.raises(RepositoryError, match="commit failed") as raised:
            await repository.commit(started("run-a", "cp-0"))
        assert isinstance(raised.value.__cause__, failure_type)
        assert tuple(call[0] for call in client.eval_calls) == (_PROBE_SCRIPT,)
    finally:
        await repository.close()


async def test_redis_unknown_outcome_retries_both_cluster_failure_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisConnectionError("lost commit response"),
        RedisClusterError("cluster topology unavailable"),
        RedisClusterException("cluster routing unavailable"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="cluster-settlement", cluster=True)
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
            _PROBE_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


@pytest.mark.parametrize("failure_type", [RedisClusterError, RedisClusterException])
async def test_redis_cluster_write_failure_is_settled_after_initial_probe_new(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        failure_type("cluster routing unavailable"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="cluster-write-settlement", cluster=True)
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


@pytest.mark.parametrize("failure_type", [RedisClusterError, RedisClusterException])
async def test_redis_cluster_write_failure_after_cleared_unknown_is_settled(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisConnectionError("lost commit response"),
        [b"new"],
        failure_type("cluster routing unavailable"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="cleared-cluster-settlement", cluster=True)
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


async def test_redis_postwrite_ambiguous_result_keeps_probing_until_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisConnectionError("lost commit response"),
        RedisConnectionError("settlement probe unavailable"),
        RedisResponseError("BUSY Redis is busy"),
        RedisResponseError("MOVED 1 redis.internal:6379"),
        RedisResponseError("ERR arbitrary response while settling"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="settlement")
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert len(client.eval_calls) == 7
        assert client.eval_calls[0][0] == _PROBE_SCRIPT
        assert client.eval_calls[1][0] == _COMMIT_SCRIPT
        assert all(call[0] == _PROBE_SCRIPT for call in client.eval_calls[2:])
    finally:
        await repository.close()


async def test_redis_commit_response_error_is_probed_when_write_may_have_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisResponseError("ERR synthetic script runtime failure"),
        RedisResponseError("ASK 1 redis.internal:6379"),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="commit-response-settlement")
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


async def test_redis_commit_known_no_write_response_fails_without_settlement() -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisResponseError("MOVED 1 redis.internal:6379"),
    )
    repository = RedisRunRepository(key_prefix="commit-moved")
    object.__setattr__(repository, "_client", client)
    try:
        with pytest.raises(RepositoryError, match="commit failed") as raised:
            await repository.commit(started("run-a", "cp-0"))
        assert isinstance(raised.value.__cause__, RedisResponseError)
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
        )
    finally:
        await repository.close()


@pytest.mark.parametrize(
    "response",
    [
        "READONLY replica cannot accept writes",
        "MOVED 1 redis.internal:6379",
    ],
)
async def test_redis_cluster_write_response_is_settled(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> None:
    client = _ScriptedRedisClient(
        [b"new"],
        RedisResponseError(response),
        [b"idempotent"],
    )
    repository = RedisRunRepository(key_prefix="cluster-response-settlement", cluster=True)
    object.__setattr__(repository, "_client", client)
    monkeypatch.setattr("jharness.repository.redis._SETTLEMENT_INITIAL_DELAY", 0)
    try:
        await repository.commit(started("run-a", "cp-0"))
        assert tuple(call[0] for call in client.eval_calls) == (
            _PROBE_SCRIPT,
            _COMMIT_SCRIPT,
            _PROBE_SCRIPT,
        )
    finally:
        await repository.close()


def test_redis_commit_script_has_a_complete_preflight_write_barrier() -> None:
    first_write = _COMMIT_SCRIPT.index("redis.call('HSET'")
    mutation_phase = _COMMIT_SCRIPT[first_write:]
    assert _COMMIT_SCRIPT.count("for offset = 1, new_chunks do") == 3
    assert _COMMIT_SCRIPT.index("if message_count ~= ARGV[12]") < first_write
    assert _COMMIT_SCRIPT.rindex("return {'state_corrupt'}", 0, first_write) < first_write
    assert [line.strip() for line in mutation_phase.splitlines() if "return" in line] == [
        "return {'committed'}"
    ]


async def test_redis_commit_settles_eval_after_cancellation() -> None:
    repository = RedisRunRepository(key_prefix="cancel")
    client = _BlockingRedisClient()
    object.__setattr__(repository, "_client", client)
    task = asyncio.create_task(repository.commit(started("run-a", "cp-0")))
    try:
        await asyncio.wait_for(client.started.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        client.release.set()
        await asyncio.wait_for(task, 1)
    finally:
        client.release.set()
        await repository.close()
    assert client.closed


@pytest.mark.skipif(
    _REDIS_URL_ENV not in os.environ,
    reason=f"set {_REDIS_URL_ENV} to run Redis integration tests",
)
async def test_redis_atomic_cas_per_run_ledger_and_reliable_cleanup() -> None:
    raw_url = os.environ[_REDIS_URL_ENV]
    key_prefix = f"jharness-test-{uuid4().hex}"
    repository = _repository(key_prefix)
    peer = _repository(key_prefix)
    first = started("run-a", "shared-checkpoint")
    other = started("run-b", "shared-checkpoint")
    left = append_external(first.checkpoint, "checkpoint-left")
    right = append_external(first.checkpoint, "checkpoint-right")
    raw_client = redis_asyncio.from_url(raw_url, decode_responses=False)
    keys = (
        *repository._run_keys("run-a"),  # pyright: ignore[reportPrivateUsage]
        *repository._run_keys("run-b"),  # pyright: ignore[reportPrivateUsage]
    )
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

            assert head is not None
            fault = _append_many(head, "checkpoint-fault-injection", 65)
            fault_identity = commit_identity(fault)
            fault_core = encode_core(fault_identity)
            fault_chunks = encode_history_change(fault)
            assert len(fault_chunks) == 2
            last = fault_chunks[-1]
            invalid_chunks = (
                fault_chunks[0],
                EncodedHistoryChunk(0, last.payload, last.digest),
            )
            fault_head_key, fault_ledger_key, fault_history_key = repository._run_keys(  # pyright: ignore[reportPrivateUsage]
                "run-a"
            )
            generation_raw, first_index_raw = await raw_client.hmget(
                fault_head_key,
                ["history_generation", "history_chunk_count"],
            )
            assert isinstance(generation_raw, bytes)
            assert isinstance(first_index_raw, bytes)
            generation = int(generation_raw)
            first_index = int(first_index_raw)
            result = await repository._eval_commit(  # pyright: ignore[reportPrivateUsage, reportArgumentType]
                raw_client,  # pyright: ignore[reportArgumentType]
                fault_identity,
                fault_core.payload,
                invalid_chunks,
            )
            assert result == [b"invalid_arguments"]
            fault_prefix = _checkpoint_field_prefix(fault.checkpoint_id)
            assert not await raw_client.hexists(
                fault_ledger_key,
                f"{fault_prefix}checkpoint_id",
            )
            fault_fields: list[str] = []
            for index in range(len(fault_chunks)):
                chunk_prefix = _history_chunk_prefix(generation, first_index + index)
                fault_fields.extend((f"{chunk_prefix}payload", f"{chunk_prefix}count"))
            assert await raw_client.hmget(fault_history_key, fault_fields) == [None] * len(
                fault_fields
            )
            assert await repository.get_head("run-a") == head

            collision = append_external(first.checkpoint, first.checkpoint_id, text="changed")
            with pytest.raises(RepositoryError, match="reused"):
                await repository.commit(collision)

            assert head is not None
            next_commit = append_external(head, "checkpoint-after-corruption")
            head_key, ledger_key, _ = repository._run_keys(  # pyright: ignore[reportPrivateUsage]
                "run-a"
            )
            await raw_client.hset(head_key, "checkpoint_core", b"corrupt-core")
            with pytest.raises(RepositoryError, match="invalid"):
                await repository.get_head("run-a")
            await repository.commit(next_commit)
            assert await repository.get_head("run-a") == next_commit.checkpoint
            next_prefix = _checkpoint_field_prefix(next_commit.checkpoint_id)
            assert await raw_client.hexists(
                ledger_key,
                f"{next_prefix}checkpoint_id",
            )
    finally:
        try:
            await repository.close()
        finally:
            try:
                await peer.close()
            finally:
                try:
                    await raw_client.delete(*keys)
                    for key in keys:
                        assert not await raw_client.exists(key)
                finally:
                    await raw_client.aclose()


@pytest.mark.skipif(
    _REDIS_CLUSTER_URL_ENV not in os.environ,
    reason=f"set {_REDIS_CLUSTER_URL_ENV} to run Redis Cluster integration tests",
)
async def test_redis_cluster_routes_distinct_run_slots_and_reliably_cleans_up() -> None:
    raw_url = os.environ[_REDIS_CLUSTER_URL_ENV]
    key_prefix = f"jharness-cluster-test-{uuid4().hex}"
    repository = RedisRunRepository(raw_url, key_prefix=key_prefix, cluster=True)
    raw_client = redis_asyncio.RedisCluster.from_url(raw_url, decode_responses=False)
    run_keys: tuple[tuple[str, str, str], ...] = ()
    try:
        await raw_client.ping()  # pyright: ignore[reportUnknownMemberType]
        runs_by_node: dict[str, str] = {}
        for index in range(256):
            run_id = f"cluster-run-{index}"
            head_key = repository._run_keys(run_id)[0]  # pyright: ignore[reportPrivateUsage]
            node = raw_client.get_node_from_key(head_key)
            assert node is not None
            runs_by_node.setdefault(node.name, run_id)
            if len(runs_by_node) == 3:
                break
        assert len(runs_by_node) >= 2

        run_ids = tuple(runs_by_node.values())
        run_keys = tuple(
            repository._run_keys(run_id)  # pyright: ignore[reportPrivateUsage]
            for run_id in run_ids
        )
        commits = tuple(
            started(run_id, f"cluster-checkpoint-{index}") for index, run_id in enumerate(run_ids)
        )
        async with repository:
            for commit in commits:
                await repository.commit(commit)
                assert await repository.get_head(commit.run_id) == commit.checkpoint
                await repository.commit(commit)
            for keys in run_keys:
                assert await raw_client.exists(*keys) == len(keys)
    finally:
        try:
            await repository.close()
        finally:
            try:
                for keys in run_keys:
                    await raw_client.delete(*keys)
                    assert await raw_client.exists(*keys) == 0
            finally:
                await raw_client.aclose()
