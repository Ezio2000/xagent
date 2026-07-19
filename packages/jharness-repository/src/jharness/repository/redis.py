"""Redis-backed incremental durable-commit repository."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha1, sha256
from importlib import import_module
from typing import Protocol, TypeVar, cast

from jharness.kernel import (
    Checkpoint,
    DurableCommit,
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
_LEDGER_FIELDS = ("checkpoint_id", "revision", "checkpoint_digest")
_HEAD_FIELDS = (
    "run_id",
    "revision",
    "checkpoint_id",
    "parent_checkpoint_id",
    "checkpoint_digest",
    "checkpoint_core",
    "checkpoint_core_sha1",
    "history_generation",
    "history_chunk_count",
    "history_message_count",
    "history_digest",
    "history_chain_sha1",
)
_READ_CHUNK_BATCH = 128
_SETTLEMENT_INITIAL_DELAY = 0.05
_SETTLEMENT_MAX_DELAY = 1.0
_CHAIN_DOMAIN = b"jharness.repository.redis.history.v2"
_EMPTY_CHAIN_SHA1 = sha1(_CHAIN_DOMAIN, usedforsecurity=False).hexdigest()

_LUA_HELPERS = r"""
local function canonical_nonnegative(value)
    if not value or not string.match(value, '^%d+$') then
        return false
    end
    return value == '0' or string.sub(value, 1, 1) ~= '0'
end

local function canonical_positive(value)
    return canonical_nonnegative(value) and value ~= '0'
end

local function decimal_gte(left, right)
    if string.len(left) ~= string.len(right) then
        return string.len(left) > string.len(right)
    end
    return left >= right
end

local function decimal_add(left, right)
    local carry = 0
    local output = ''
    local li = string.len(left)
    local ri = string.len(right)
    while li > 0 or ri > 0 or carry > 0 do
        local ld = 0
        local rd = 0
        if li > 0 then
            ld = string.byte(left, li) - 48
            li = li - 1
        end
        if ri > 0 then
            rd = string.byte(right, ri) - 48
            ri = ri - 1
        end
        local total = ld + rd + carry
        output = string.char(48 + (total % 10)) .. output
        carry = math.floor(total / 10)
    end
    return output
end

local function chain_step(chain, count, payload)
    return redis.sha1hex(
        chain .. string.char(0) .. count .. string.char(0) .. redis.sha1hex(payload)
    )
end

local function hash_key_or_missing(key)
    local result = redis.call('TYPE', key)
    local kind = result
    if type(result) == 'table' then
        kind = result['ok']
    end
    return kind == 'none' or kind == 'hash'
end

local function read_head(head_key, ledger_key)
    local head = redis.call('HMGET', head_key,
        'run_id',
        'revision',
        'checkpoint_id',
        'parent_checkpoint_id',
        'checkpoint_digest',
        'checkpoint_core_sha1',
        'history_generation',
        'history_chunk_count',
        'history_message_count',
        'history_digest',
        'history_chain_sha1'
    )
    local any = false
    for index = 1, 11 do
        if head[index] then
            any = true
        end
    end
    if not any then
        return nil, nil
    end
    for index = 1, 11 do
        if not head[index] then
            return nil, 'head_corrupt'
        end
    end
    if head[1] == '' or head[3] == ''
        or not canonical_nonnegative(head[2])
        or not canonical_nonnegative(head[7])
        or not canonical_positive(head[8])
        or not canonical_positive(head[9])
        or string.len(head[5]) ~= 32
        or string.len(head[10]) ~= 32
        or not string.match(head[6], '^[0-9a-f]+$') or string.len(head[6]) ~= 40
        or not string.match(head[11], '^[0-9a-f]+$') or string.len(head[11]) ~= 40
        or (head[2] == '0' and head[4] ~= '')
        or (head[2] ~= '0' and head[4] == '') then
        return nil, 'head_corrupt'
    end
    local prefix = 'c:' .. redis.sha1hex(head[3]) .. ':'
    local linked = redis.call('HMGET', ledger_key,
        prefix .. 'checkpoint_id',
        prefix .. 'revision',
        prefix .. 'checkpoint_digest'
    )
    if not linked[1] or not linked[2] or not linked[3]
        or linked[1] ~= head[3] or linked[2] ~= head[2] or linked[3] ~= head[5] then
        return nil, 'head_corrupt'
    end
    return head, nil
end

local function read_ledger(ledger_key, prefix)
    local stored = redis.call('HMGET', ledger_key,
        prefix .. 'checkpoint_id',
        prefix .. 'revision',
        prefix .. 'checkpoint_digest'
    )
    local any = stored[1] or stored[2] or stored[3]
    if not any then
        return nil, nil
    end
    if not stored[1] or not stored[2] or not stored[3]
        or stored[1] == '' or not canonical_nonnegative(stored[2])
        or string.len(stored[3]) ~= 32 then
        return nil, 'ledger_corrupt'
    end
    return stored, nil
end
"""

_PROBE_SCRIPT = (
    _LUA_HELPERS
    + r"""
if not hash_key_or_missing(KEYS[1])
    or not hash_key_or_missing(KEYS[2])
    or not hash_key_or_missing(KEYS[3]) then
    return {'state_corrupt'}
end

local head, head_error = read_head(KEYS[1], KEYS[2])
if head_error then
    return {head_error}
end
if not head then
    if redis.call('EXISTS', KEYS[2]) ~= 0 or redis.call('EXISTS', KEYS[3]) ~= 0 then
        return {'state_corrupt'}
    end
end
if head and head[1] ~= ARGV[1] then
    return {'run_hash_collision'}
end

local stored, ledger_error = read_ledger(KEYS[2], ARGV[2])
if ledger_error then
    return {ledger_error}
end
if stored then
    if stored[1] ~= ARGV[3] then
        return {'id_hash_collision'}
    end
    if stored[2] ~= ARGV[4] or stored[3] ~= ARGV[6] then
        return {'id_reused'}
    end
    if not head or not decimal_gte(head[2], stored[2]) then
        return {'ledger_corrupt'}
    end
    if head[2] == stored[2] and (head[3] ~= stored[1] or head[5] ~= stored[3]) then
        return {'ledger_corrupt'}
    end
    return {'idempotent'}
end

local actual = '-1'
if head then
    actual = head[2]
end
if actual ~= ARGV[5] then
    return {'revision_conflict', actual}
end
return {'new'}
"""
)

_COMMIT_SCRIPT = (
    _LUA_HELPERS
    + r"""
if #ARGV < 14 then
    return {'invalid_arguments'}
end

local kind = ARGV[9]
if ARGV[1] == '' or ARGV[3] == '' or ARGV[8] == ''
    or ARGV[2] ~= 'c:' .. redis.sha1hex(ARGV[3]) .. ':'
    or not canonical_nonnegative(ARGV[4])
    or (ARGV[5] ~= '-1' and not canonical_nonnegative(ARGV[5]))
    or string.len(ARGV[7]) ~= 32
    or not canonical_positive(ARGV[12])
    or string.len(ARGV[13]) ~= 32
    or not canonical_nonnegative(ARGV[14])
    or (kind ~= 'initial' and kind ~= 'append'
        and kind ~= 'replace' and kind ~= 'unchanged') then
    return {'invalid_arguments'}
end
if ARGV[10] == '-1' then
    if ARGV[11] ~= '' then
        return {'invalid_arguments'}
    end
elseif not canonical_nonnegative(ARGV[10]) or string.len(ARGV[11]) ~= 32 then
    return {'invalid_arguments'}
end

local new_chunks = tonumber(ARGV[14])
if not new_chunks or new_chunks < 0 or new_chunks > 9007199254740991
    or #ARGV ~= 14 + (new_chunks * 2) then
    return {'invalid_arguments'}
end
if (kind == 'unchanged' and new_chunks ~= 0)
    or (kind ~= 'unchanged' and new_chunks == 0) then
    return {'invalid_history_change'}
end

local argument = 15
for offset = 1, new_chunks do
    local count = ARGV[argument]
    local payload = ARGV[argument + 1]
    argument = argument + 2
    if not canonical_positive(count) or decimal_gte(count, '65')
        or not payload or payload == '' then
        return {'invalid_arguments'}
    end
end

if not hash_key_or_missing(KEYS[1])
    or not hash_key_or_missing(KEYS[2])
    or not hash_key_or_missing(KEYS[3]) then
    return {'state_corrupt'}
end

local head, head_error = read_head(KEYS[1], KEYS[2])
if head_error then
    return {head_error}
end
if not head then
    if redis.call('EXISTS', KEYS[2]) ~= 0 or redis.call('EXISTS', KEYS[3]) ~= 0 then
        return {'state_corrupt'}
    end
elseif head[1] ~= ARGV[1] then
    return {'run_hash_collision'}
end

local stored, ledger_error = read_ledger(KEYS[2], ARGV[2])
if ledger_error then
    return {ledger_error}
end
if stored then
    if stored[1] ~= ARGV[3] then
        return {'id_hash_collision'}
    end
    if stored[2] ~= ARGV[4] or stored[3] ~= ARGV[7] then
        return {'id_reused'}
    end
    if not head or not decimal_gte(head[2], stored[2]) then
        return {'ledger_corrupt'}
    end
    if head[2] == stored[2] and (head[3] ~= stored[1] or head[5] ~= stored[3]) then
        return {'ledger_corrupt'}
    end
    return {'idempotent'}
end

local actual = '-1'
if head then
    actual = head[2]
end
if actual ~= ARGV[5] then
    return {'revision_conflict', actual}
end

local generation = ARGV[4]
local first_index = '0'
local total_chunks = ARGV[14]
local message_count = '0'
local chain = redis.sha1hex('jharness.repository.redis.history.v2')

if not head then
    if ARGV[4] ~= '0' or kind ~= 'initial'
        or ARGV[6] ~= '' or ARGV[10] ~= '-1' then
        return {'invalid_history_base'}
    end
else
    if ARGV[4] ~= decimal_add(head[2], '1') then
        return {'invalid_arguments'}
    end
    if ARGV[6] ~= head[3] then
        return {'parent_mismatch'}
    end
    if ARGV[10] ~= head[9] or ARGV[11] ~= head[10] then
        return {'invalid_history_base'}
    end
    local anchor = 'g:' .. head[7] .. ':c:0:'
    local anchor_count = redis.call('HGET', KEYS[3], anchor .. 'count')
    if not anchor_count or not canonical_positive(anchor_count)
        or decimal_gte(anchor_count, '65')
        or not decimal_gte(head[9], anchor_count)
        or redis.call('HEXISTS', KEYS[3], anchor .. 'payload') == 0 then
        return {'state_corrupt'}
    end
    if kind == 'append' then
        generation = head[7]
        first_index = head[8]
        total_chunks = decimal_add(head[8], ARGV[14])
        message_count = head[9]
        chain = head[11]
    elseif kind == 'unchanged' then
        if ARGV[12] ~= head[9] or ARGV[13] ~= head[10] then
            return {'invalid_history_change'}
        end
        generation = head[7]
        first_index = head[8]
        total_chunks = head[8]
        message_count = head[9]
        chain = head[11]
    elseif kind ~= 'replace' then
        return {'invalid_history_change'}
    end
end

argument = 15
local index = first_index
for offset = 1, new_chunks do
    local count = ARGV[argument]
    local payload = ARGV[argument + 1]
    argument = argument + 2
    local payload_field = 'g:' .. generation .. ':c:' .. index .. ':payload'
    local count_field = 'g:' .. generation .. ':c:' .. index .. ':count'
    if redis.call('HEXISTS', KEYS[3], payload_field) ~= 0
        or redis.call('HEXISTS', KEYS[3], count_field) ~= 0 then
        return {'state_corrupt'}
    end
    message_count = decimal_add(message_count, count)
    chain = chain_step(chain, count, payload)
    index = decimal_add(index, '1')
end

if message_count ~= ARGV[12] then
    return {'invalid_history_change'}
end

argument = 15
index = first_index
for offset = 1, new_chunks do
    local count = ARGV[argument]
    local payload = ARGV[argument + 1]
    argument = argument + 2
    local prefix = 'g:' .. generation .. ':c:' .. index .. ':'
    redis.call('HSET', KEYS[3],
        prefix .. 'payload', payload,
        prefix .. 'count', count
    )
    index = decimal_add(index, '1')
end

redis.call('HSET', KEYS[2],
    ARGV[2] .. 'checkpoint_id', ARGV[3],
    ARGV[2] .. 'revision', ARGV[4],
    ARGV[2] .. 'checkpoint_digest', ARGV[7]
)
redis.call('HSET', KEYS[1],
    'run_id', ARGV[1],
    'revision', ARGV[4],
    'checkpoint_id', ARGV[3],
    'parent_checkpoint_id', ARGV[6],
    'checkpoint_digest', ARGV[7],
    'checkpoint_core', ARGV[8],
    'checkpoint_core_sha1', redis.sha1hex(ARGV[8]),
    'history_generation', generation,
    'history_chunk_count', total_chunks,
    'history_message_count', ARGV[12],
    'history_digest', ARGV[13],
    'history_chain_sha1', chain
)
return {'committed'}
"""
)


class _RedisClient(Protocol):
    async def ping(self) -> object: ...

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: str | bytes | int,
    ) -> object: ...

    async def hmget(self, name: str, keys: list[str]) -> list[object]: ...

    async def aclose(self) -> None: ...


class _RedisFactory(Protocol):
    def from_url(
        self,
        url: str,
        *,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        health_check_interval: int,
    ) -> _RedisClient: ...


class _RedisModule(_RedisFactory, Protocol):
    RedisCluster: _RedisFactory


class _RedisExceptions(Protocol):
    RedisError: type[Exception]
    RedisClusterException: type[Exception]
    ResponseError: type[Exception]


@dataclass(frozen=True, slots=True)
class _RedisRuntime:
    standalone: _RedisFactory
    cluster: _RedisFactory
    exceptions: _RedisExceptions


@dataclass(frozen=True, slots=True)
class _RedisHead:
    run_id: str
    revision: int
    checkpoint_id: str
    parent_checkpoint_id: str | None
    checkpoint_digest: bytes
    core_payload: bytes
    history_generation: int
    history_chunk_count: int
    history_message_count: int
    history_digest: bytes
    history_chain_sha1: str


class RedisRunRepository:
    """Multi-run CAS repository using three cluster-local keys per run."""

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        key_prefix: str = "jharness",
        cluster: bool = False,
        socket_connect_timeout: float = 10.0,
        socket_timeout: float = 30.0,
        health_check_interval: int = 30,
    ) -> None:
        self._url = _non_empty_string(url, "url")
        prefix = _non_empty_string(key_prefix, "key_prefix")
        if type(cluster) is not bool:
            raise TypeError("cluster must be a boolean")
        self._namespace = sha256(prefix.encode("utf-8")).hexdigest()
        self._cluster = cluster
        self._socket_connect_timeout = _positive_number(
            socket_connect_timeout,
            "socket_connect_timeout",
        )
        self._socket_timeout = _positive_number(socket_timeout, "socket_timeout")
        self._health_check_interval = _nonnegative_integer(
            health_check_interval,
            "health_check_interval",
        )
        self._client: _RedisClient | None = None
        self._initialize_lock = asyncio.Lock()
        self._operations: set[asyncio.Task[object]] = set()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False

    async def initialize(self) -> None:
        """Connect to Redis and verify that the selected database responds."""

        if self._closing or self._closed:
            raise RepositoryError("Redis repository is closed")
        try:
            await self._initialize_client()
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError("Redis repository initialization failed") from exc

    async def commit(self, commit: DurableCommit) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""

        identity = commit_identity(commit)
        await self._run(self._commit_identity(identity), "commit")

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""

        normalized = _non_empty_string(run_id, "run_id")
        return await self._run(self._get_head(normalized), "read")

    async def close(self) -> None:
        """Wait for accepted operations and close the Redis connection pool."""

        close_task = self._close_task
        if close_task is None:
            self._closing = True
            close_task = asyncio.create_task(self._close_after(tuple(self._operations)))
            self._close_task = close_task
        try:
            await _settle_task(close_task)
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError("Redis repository close failed") from exc

    async def __aenter__(self) -> RedisRunRepository:
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

    async def _run(self, operation: Coroutine[object, object, _T], label: str) -> _T:
        if self._closing or self._closed:
            operation.close()
            raise RepositoryError("Redis repository is closed")
        task = asyncio.create_task(operation)
        tracked = cast(asyncio.Task[object], task)
        self._operations.add(tracked)
        try:
            return await _settle_task(task)
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"Redis repository {label} failed") from exc
        finally:
            self._operations.discard(tracked)

    async def _initialize_client(self) -> _RedisClient:
        client = self._client
        if client is not None:
            return client
        async with self._initialize_lock:
            client = self._client
            if client is not None:
                return client
            runtime = _load_redis()
            factory = runtime.cluster if self._cluster else runtime.standalone
            client = factory.from_url(
                self._url,
                decode_responses=False,
                socket_connect_timeout=self._socket_connect_timeout,
                socket_timeout=self._socket_timeout,
                health_check_interval=self._health_check_interval,
            )
            try:
                await client.ping()
            except BaseException:
                with suppress(Exception):
                    await client.aclose()
                raise
            self._client = client
            return client

    async def _commit_identity(self, identity: CommitIdentity) -> None:
        client = await self._initialize_client()
        exceptions = _load_redis().exceptions
        delay = _SETTLEMENT_INITIAL_DELAY
        outcome_unknown = False
        prepared: tuple[bytes, tuple[EncodedHistoryChunk, ...]] | None = None
        while True:
            try:
                probe = await self._eval_probe(client, identity)
                if _handle_probe_result(probe, identity):
                    return
                if outcome_unknown:
                    outcome_unknown = False
                    delay = _SETTLEMENT_INITIAL_DELAY
            except Exception as exc:
                if not outcome_unknown or not _is_redis_settlement_error(
                    exc,
                    exceptions,
                ):
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, _SETTLEMENT_MAX_DELAY)
                continue

            if prepared is None:
                core = encode_core(identity)
                prepared = (core.payload, encode_history_change(identity.commit))
            try:
                result = await self._eval_commit(client, identity, *prepared)
                _handle_commit_result(result, identity)
                return
            except Exception as exc:
                outcome_unknown = _commit_outcome_is_unknown(
                    exc,
                    cluster=self._cluster,
                    outcome_unknown=outcome_unknown,
                    exceptions=exceptions,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _SETTLEMENT_MAX_DELAY)

    async def _eval_probe(self, client: _RedisClient, identity: CommitIdentity) -> object:
        head_key, ledger_key, history_key = self._run_keys(identity.run_id)
        return await client.eval(
            _PROBE_SCRIPT,
            3,
            head_key,
            ledger_key,
            history_key,
            identity.run_id,
            _checkpoint_field_prefix(identity.checkpoint_id),
            identity.checkpoint_id,
            str(identity.revision),
            _revision_text(identity.expected_revision),
            identity.digest,
        )

    async def _eval_commit(
        self,
        client: _RedisClient,
        identity: CommitIdentity,
        core_payload: bytes,
        chunks: tuple[EncodedHistoryChunk, ...],
    ) -> object:
        head_key, ledger_key, history_key = self._run_keys(identity.run_id)
        arguments: list[str | bytes | int] = [
            identity.run_id,
            _checkpoint_field_prefix(identity.checkpoint_id),
            identity.checkpoint_id,
            str(identity.revision),
            _revision_text(identity.expected_revision),
            "" if identity.parent_checkpoint_id is None else identity.parent_checkpoint_id,
            identity.digest,
            core_payload,
            identity.commit.history.kind,
            _revision_text(identity.base_history_count),
            b"" if identity.base_history_digest is None else identity.base_history_digest,
            str(identity.history_count),
            identity.history_digest,
            str(len(chunks)),
        ]
        for chunk in chunks:
            arguments.extend((str(chunk.message_count), chunk.payload))
        return await client.eval(
            _COMMIT_SCRIPT,
            3,
            head_key,
            ledger_key,
            history_key,
            *arguments,
        )

    async def _get_head(self, run_id: str) -> Checkpoint | None:
        client = await self._initialize_client()
        head_key, ledger_key, history_key = self._run_keys(run_id)
        values = _hash_values(
            await client.hmget(head_key, list(_HEAD_FIELDS)),
            len(_HEAD_FIELDS),
            "run head",
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RepositoryError("stored Redis run head is incomplete")
        head = _decode_redis_head(values, run_id)
        core = decode_core(
            head.core_payload,
            sha256(head.core_payload).digest(),
        )
        if (
            core.checkpoint_id != head.checkpoint_id
            or core.parent_checkpoint_id != head.parent_checkpoint_id
            or core.revision != head.revision
            or core.history_count != head.history_message_count
            or core.history_digest != head.history_digest
        ):
            raise RepositoryError("stored Redis run head core is inconsistent")

        await _validate_redis_ledger(client, ledger_key, head)
        chunks, message_count, chain = await _read_redis_history(
            client,
            history_key,
            head,
        )
        if message_count != head.history_message_count or chain != head.history_chain_sha1:
            raise RepositoryError("stored Redis checkpoint history manifest is inconsistent")
        checkpoint, decoded_core = reconstruct_checkpoint(
            core_payload=head.core_payload,
            core_digest=sha256(head.core_payload).digest(),
            chunks=chunks,
            expected_checkpoint_digest=head.checkpoint_digest,
        )
        if (
            checkpoint.id != head.checkpoint_id
            or checkpoint.snapshot.context.run_id != run_id
            or checkpoint.snapshot.revision != head.revision
            or decoded_core.history_digest != head.history_digest
        ):
            raise RepositoryError("stored Redis run head is inconsistent")
        return checkpoint

    async def _close_after(self, operations: tuple[asyncio.Task[object], ...]) -> None:
        for operation in operations:
            with suppress(Exception, asyncio.CancelledError):
                await _settle_task(operation)
        async with self._initialize_lock:
            client = self._client
            self._client = None
            try:
                if client is not None:
                    await client.aclose()
            finally:
                self._closed = True

    def _run_keys(self, run_id: str) -> tuple[str, str, str]:
        run_hash = sha256(run_id.encode("utf-8")).hexdigest()
        base = f"jharness:v2:{{{self._namespace}:{run_hash}}}"
        return f"{base}:head", f"{base}:ledger", f"{base}:history"


async def _settle_task(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()


def _load_redis() -> _RedisRuntime:
    try:
        client = import_module("redis.asyncio")
        exceptions = import_module("redis.exceptions")
    except ModuleNotFoundError as exc:
        if exc.name != "redis" and not (exc.name or "").startswith("redis."):
            raise
        raise RepositoryError(
            "Redis repository support is not installed; run: uv add 'jharness-repository[redis]'"
        ) from exc
    module = cast(_RedisModule, client)
    return _RedisRuntime(
        module,
        module.RedisCluster,
        cast(_RedisExceptions, exceptions),
    )


def _handle_probe_result(result: object, identity: CommitIdentity) -> bool:
    status, items = _result_status(result)
    if status == "idempotent":
        return True
    if status == "new":
        return False
    _raise_redis_status(status, items, identity)
    raise AssertionError("unreachable")


def _handle_commit_result(result: object, identity: CommitIdentity) -> None:
    status, items = _result_status(result)
    if status in {"committed", "idempotent"}:
        return
    _raise_redis_status(status, items, identity)


def _result_status(result: object) -> tuple[str, tuple[object, ...]]:
    if not isinstance(result, (list, tuple)) or not result:
        raise RepositoryError("Redis script returned an invalid result")
    items = tuple(cast(list[object] | tuple[object, ...], result))
    return _redis_utf8(items[0], "script status"), items


def _raise_redis_status(
    status: str,
    items: tuple[object, ...],
    identity: CommitIdentity,
) -> None:
    if status == "revision_conflict":
        if len(items) != 2:
            raise RepositoryError("Redis script returned an invalid conflict")
        actual_text = _redis_utf8(items[1], "actual revision")
        actual = None if actual_text == "-1" else _revision_from_text(actual_text)
        raise RevisionConflict(identity.run_id, identity.expected_revision, actual)
    if status == "id_reused":
        raise RepositoryError(
            f"checkpoint id {identity.checkpoint_id!r} was reused with new content "
            f"in run {identity.run_id!r}"
        )
    if status == "id_hash_collision":
        raise RepositoryError("stored Redis checkpoint id hash collision")
    if status == "run_hash_collision":
        raise RepositoryError("stored Redis run id hash collision")
    if status == "parent_mismatch":
        raise RepositoryError("parent checkpoint does not match the authoritative head")
    if status == "invalid_history_base":
        raise RepositoryError("history change base does not match the authoritative head")
    if status in {
        "head_corrupt",
        "ledger_corrupt",
        "state_corrupt",
        "invalid_arguments",
        "invalid_history_change",
    }:
        raise RepositoryError("stored Redis repository data is invalid")
    raise RepositoryError("Redis script returned an unknown status")


def _decode_redis_head(values: tuple[object, ...], requested_run_id: str) -> _RedisHead:
    run_id = _redis_utf8(values[0], "run id")
    if run_id != requested_run_id:
        raise RepositoryError("stored Redis run id hash collision")
    revision = _redis_revision(_redis_bytes(values[1], "revision"))
    checkpoint_id = _redis_utf8(values[2], "checkpoint id")
    parent_bytes = _redis_bytes(values[3], "parent checkpoint id")
    parent = None if not parent_bytes else _redis_utf8(parent_bytes, "parent checkpoint id")
    checkpoint_digest = _redis_digest(values[4], "checkpoint digest")
    core_payload = _redis_bytes(values[5], "checkpoint core")
    core_sha1 = _redis_utf8(values[6], "checkpoint core SHA-1")
    if sha1(core_payload, usedforsecurity=False).hexdigest() != core_sha1:
        raise RepositoryError("stored Redis checkpoint core has an invalid digest")
    generation = _redis_revision(_redis_bytes(values[7], "history generation"))
    chunk_count = _redis_revision(_redis_bytes(values[8], "history chunk count"))
    message_count = _redis_revision(_redis_bytes(values[9], "history message count"))
    if chunk_count <= 0 or message_count <= 0:
        raise RepositoryError("stored Redis history manifest is invalid")
    history_digest = _redis_digest(values[10], "history digest")
    chain = _redis_utf8(values[11], "history chain SHA-1")
    if len(chain) != 40 or any(character not in "0123456789abcdef" for character in chain):
        raise RepositoryError("stored Redis history chain SHA-1 is invalid")
    if (revision == 0) != (parent is None):
        raise RepositoryError("stored Redis parent checkpoint id is invalid")
    return _RedisHead(
        run_id,
        revision,
        checkpoint_id,
        parent,
        checkpoint_digest,
        core_payload,
        generation,
        chunk_count,
        message_count,
        history_digest,
        chain,
    )


async def _validate_redis_ledger(
    client: _RedisClient,
    ledger_key: str,
    head: _RedisHead,
) -> None:
    prefix = _checkpoint_field_prefix(head.checkpoint_id)
    ledger = _hash_values(
        await client.hmget(ledger_key, _prefixed_fields(prefix, _LEDGER_FIELDS)),
        len(_LEDGER_FIELDS),
        "checkpoint ledger",
    )
    if any(value is None for value in ledger):
        raise RepositoryError("stored Redis run head has no complete checkpoint ledger entry")
    if (
        _redis_utf8(ledger[0], "checkpoint id") != head.checkpoint_id
        or _redis_revision(_redis_bytes(ledger[1], "revision")) != head.revision
        or _redis_digest(ledger[2], "checkpoint digest") != head.checkpoint_digest
    ):
        raise RepositoryError("stored Redis run head and checkpoint ledger differ")


async def _read_redis_history(
    client: _RedisClient,
    history_key: str,
    head: _RedisHead,
) -> tuple[list[tuple[bytes, bytes, int]], int, str]:
    chunks: list[tuple[bytes, bytes, int]] = []
    message_count = 0
    chain = _EMPTY_CHAIN_SHA1
    for start in range(0, head.history_chunk_count, _READ_CHUNK_BATCH):
        stop = min(head.history_chunk_count, start + _READ_CHUNK_BATCH)
        fields = _history_fields(head.history_generation, start, stop)
        stored = _hash_values(
            await client.hmget(history_key, fields),
            len(fields),
            "history chunks",
        )
        if any(value is None for value in stored):
            raise RepositoryError("stored Redis checkpoint history is incomplete")
        for offset in range(0, len(stored), 2):
            payload = _redis_bytes(stored[offset], "history chunk payload")
            count = _redis_revision(_redis_bytes(stored[offset + 1], "history chunk count"))
            if count <= 0 or count > HISTORY_CHUNK_SIZE:
                raise RepositoryError("stored Redis history chunk count is invalid")
            message_count += count
            chain = _history_chain_step(chain, count, payload)
            chunks.append((payload, sha256(payload).digest(), count))
    return chunks, message_count, chain


def _history_fields(generation: int, start: int, stop: int) -> list[str]:
    fields: list[str] = []
    for index in range(start, stop):
        prefix = _history_chunk_prefix(generation, index)
        fields.extend((f"{prefix}payload", f"{prefix}count"))
    return fields


def _redis_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise RepositoryError(f"stored Redis {label} is invalid")
    return value


def _redis_digest(value: object, label: str) -> bytes:
    digest = _redis_bytes(value, label)
    if len(digest) != 32:
        raise RepositoryError(f"stored Redis {label} is invalid")
    return digest


def _hash_values(result: object, expected: int, label: str) -> tuple[object, ...]:
    if not isinstance(result, (list, tuple)):
        raise RepositoryError(f"stored Redis {label} returned an invalid field set")
    values = tuple(cast(list[object] | tuple[object, ...], result))
    if len(values) != expected:
        raise RepositoryError(f"stored Redis {label} returned an invalid field set")
    return values


def _prefixed_fields(prefix: str, names: tuple[str, ...]) -> list[str]:
    return [f"{prefix}{name}" for name in names]


def _redis_utf8(value: object, label: str) -> str:
    raw = _redis_bytes(value, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepositoryError(f"stored Redis {label} is invalid") from exc
    if not text:
        raise RepositoryError(f"stored Redis {label} is invalid")
    return text


def _redis_revision(value: bytes) -> int:
    try:
        text = value.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RepositoryError("stored Redis revision is invalid") from exc
    return _revision_from_text(text)


def _revision_from_text(value: str) -> int:
    try:
        revision = int(value)
    except ValueError as exc:
        raise RepositoryError("stored Redis revision is invalid") from exc
    if revision < 0 or str(revision) != value:
        raise RepositoryError("stored Redis revision is invalid")
    return revision


def _revision_text(revision: int | None) -> str:
    return "-1" if revision is None else str(revision)


def _checkpoint_field_prefix(checkpoint_id: str) -> str:
    return f"c:{sha1(checkpoint_id.encode('utf-8'), usedforsecurity=False).hexdigest()}:"


def _history_chunk_prefix(generation: int, index: int) -> str:
    return f"g:{generation}:c:{index}:"


def _history_chain_step(chain: str, count: int, payload: bytes) -> str:
    payload_sha1 = sha1(payload, usedforsecurity=False).hexdigest()
    value = b"\0".join(
        (
            chain.encode("ascii"),
            str(count).encode("ascii"),
            payload_sha1.encode("ascii"),
        )
    )
    return sha1(value, usedforsecurity=False).hexdigest()


def _redis_response_proves_no_write(error: Exception) -> bool:
    status = str(error).lstrip().upper()
    return status.startswith(
        (
            "ASK ",
            "BUSY",
            "CLUSTERDOWN",
            "CROSSSLOT",
            "LOADING",
            "MASTERDOWN",
            "MOVED ",
            "READONLY",
            "TRYAGAIN",
        )
    )


def _is_redis_settlement_error(
    error: Exception,
    exceptions: _RedisExceptions,
) -> bool:
    return isinstance(
        error,
        (
            exceptions.RedisError,
            exceptions.RedisClusterException,
        ),
    )


def _commit_outcome_is_unknown(
    error: Exception,
    *,
    cluster: bool,
    outcome_unknown: bool,
    exceptions: _RedisExceptions,
) -> bool:
    if isinstance(error, exceptions.ResponseError):
        if not cluster and not outcome_unknown and _redis_response_proves_no_write(error):
            raise error
        return True
    if isinstance(error, (exceptions.RedisError, exceptions.RedisClusterException)):
        return True
    raise error


def _non_empty_string(value: str, label: str) -> str:
    if not isinstance(cast(object, value), str):
        raise TypeError(f"{label} must be a string")
    normalized = str.__str__(value)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    return normalized


def _positive_number(value: float, label: str) -> float:
    if isinstance(cast(object, value), bool) or not isinstance(cast(object, value), (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise ValueError(f"{label} must be finite and greater than zero")
    return number


def _nonnegative_integer(value: int, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value
