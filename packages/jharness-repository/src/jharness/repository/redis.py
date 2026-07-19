"""Redis-backed checkpoint repository."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Coroutine
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha1, sha256
from importlib import import_module
from typing import Protocol, TypeVar, cast

from jharness.kernel import Checkpoint, RepositoryError, RevisionConflict

from ._codec import EncodedCheckpoint, decode_checkpoint, encode_checkpoint

_T = TypeVar("_T")
_CHECKPOINT_FIELDS = ("checkpoint_id", "run_id", "revision", "checkpoint_digest")
_HEAD_FIELDS = (
    "run_id",
    "revision",
    "checkpoint_id",
    "checkpoint_digest",
    "checkpoint_payload",
    "checkpoint_field_prefix",
    "checkpoint_payload_sha1",
)
_SETTLEMENT_INITIAL_DELAY = 0.05
_SETTLEMENT_MAX_DELAY = 1.0

_COMMIT_SCRIPT = """
local state_key = KEYS[1]
local checkpoint_prefix = ARGV[1]
local head_prefix = ARGV[2]

local function canonical_revision(value)
    if not value or not string.match(value, '^%d+$') then
        return false
    end
    return value == '0' or string.sub(value, 1, 1) ~= '0'
end

local function revision_gte(left, right)
    if string.len(left) ~= string.len(right) then
        return string.len(left) > string.len(right)
    end
    return left >= right
end

local function read_head()
    local head = redis.call(
        'HMGET', state_key,
        head_prefix .. 'run_id',
        head_prefix .. 'revision',
        head_prefix .. 'checkpoint_id',
        head_prefix .. 'checkpoint_digest',
        head_prefix .. 'checkpoint_payload',
        head_prefix .. 'checkpoint_field_prefix',
        head_prefix .. 'checkpoint_payload_sha1'
    )
    local any = head[1] or head[2] or head[3] or head[4]
        or head[5] or head[6] or head[7]
    if not any then
        return nil, nil
    end
    if not head[1] or not head[2] or not head[3] or not head[4]
        or not head[5] or not head[6] or not head[7]
        or head[1] == '' or head[3] == '' or head[5] == ''
        or head[6] == '' or head[7] == ''
        or not canonical_revision(head[2]) or string.len(head[4]) ~= 32
        or not string.match(head[7], '^[0-9a-f]+$') or string.len(head[7]) ~= 40
        or redis.sha1hex(head[5]) ~= head[7] then
        return nil, 'head_corrupt'
    end

    local linked = redis.call(
        'HMGET', state_key,
        head[6] .. 'checkpoint_id',
        head[6] .. 'run_id',
        head[6] .. 'revision',
        head[6] .. 'checkpoint_digest'
    )
    if not linked[1] or not linked[2] or not linked[3] or not linked[4]
        or linked[1] == '' or linked[2] == ''
        or not canonical_revision(linked[3]) or string.len(linked[4]) ~= 32
        or linked[1] ~= head[3] or linked[2] ~= head[1]
        or linked[3] ~= head[2] or linked[4] ~= head[4] then
        return nil, 'head_corrupt'
    end
    return head, nil
end

local stored = redis.call(
    'HMGET', state_key,
    checkpoint_prefix .. 'checkpoint_id',
    checkpoint_prefix .. 'run_id',
    checkpoint_prefix .. 'revision',
    checkpoint_prefix .. 'checkpoint_digest'
)
local checkpoint_exists = stored[1] or stored[2] or stored[3] or stored[4]
if checkpoint_exists then
    if not stored[1] or not stored[2] or not stored[3] or not stored[4]
        or stored[1] == '' or stored[2] == ''
        or not canonical_revision(stored[3]) or string.len(stored[4]) ~= 32 then
        return {'ledger_corrupt'}
    end
    if stored[1] ~= ARGV[3] then
        return {'id_hash_collision'}
    end
    if stored[2] == ARGV[4] and stored[3] == ARGV[5] and stored[4] == ARGV[7] then
        local head, head_error = read_head()
        if head_error or not head then
            return {'ledger_corrupt'}
        end
        if head[1] ~= ARGV[4] then
            return {'run_hash_collision'}
        end
        if not revision_gte(head[2], ARGV[5]) then
            return {'ledger_corrupt'}
        end
        if head[2] == ARGV[5]
            and (head[3] ~= ARGV[3] or head[4] ~= ARGV[7]
                or head[5] ~= ARGV[8] or head[6] ~= checkpoint_prefix) then
            return {'ledger_corrupt'}
        end
        return {'idempotent'}
    end
    return {'id_reused'}
end

local head, head_error = read_head()
if head_error then
    return {head_error}
end

local actual_revision = '-1'
if head then
    if head[1] ~= ARGV[4] then
        return {'run_hash_collision'}
    end
    actual_revision = head[2]
end

if actual_revision ~= ARGV[6] then
    return {'revision_conflict', actual_revision}
end

redis.call(
    'HSET', state_key,
    checkpoint_prefix .. 'checkpoint_id', ARGV[3],
    checkpoint_prefix .. 'run_id', ARGV[4],
    checkpoint_prefix .. 'revision', ARGV[5],
    checkpoint_prefix .. 'checkpoint_digest', ARGV[7],
    head_prefix .. 'run_id', ARGV[4],
    head_prefix .. 'revision', ARGV[5],
    head_prefix .. 'checkpoint_id', ARGV[3],
    head_prefix .. 'checkpoint_digest', ARGV[7],
    head_prefix .. 'checkpoint_payload', ARGV[8],
    head_prefix .. 'checkpoint_field_prefix', checkpoint_prefix,
    head_prefix .. 'checkpoint_payload_sha1', redis.sha1hex(ARGV[8])
)
return {'committed'}
"""


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


class _RedisModule(Protocol):
    def from_url(
        self,
        url: str,
        *,
        decode_responses: bool,
        socket_connect_timeout: float,
        socket_timeout: float,
        health_check_interval: int,
    ) -> _RedisClient: ...


class _RedisExceptions(Protocol):
    ConnectionError: type[Exception]
    ResponseError: type[Exception]
    TimeoutError: type[Exception]


@dataclass(frozen=True, slots=True)
class _RedisRuntime:
    client: _RedisModule
    exceptions: _RedisExceptions


class RedisRunRepository:
    """Multi-run CAS repository backed by one atomic Redis Lua script."""

    def __init__(
        self,
        url: str = "redis://127.0.0.1:6379/0",
        *,
        key_prefix: str = "jharness",
        socket_connect_timeout: float = 10.0,
        socket_timeout: float = 30.0,
        health_check_interval: int = 30,
    ) -> None:
        self._url = _non_empty_string(url, "url")
        prefix = _non_empty_string(key_prefix, "key_prefix")
        self._namespace = sha256(prefix.encode("utf-8")).hexdigest()
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

    async def commit(self, checkpoint: Checkpoint) -> None:
        """Atomically advance one run head, or accept an exact prior retry."""
        encoded = encode_checkpoint(checkpoint)
        await self._run(self._commit_encoded(encoded), "commit")

    async def get_head(self, run_id: str) -> Checkpoint | None:
        """Return the authoritative checkpoint for a run, if one exists."""
        run_id = _non_empty_string(run_id, "run_id")
        return await self._run(self._get_head(run_id), "read")

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
        await self.initialize()
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
        tracked_task = cast(asyncio.Task[object], task)
        self._operations.add(tracked_task)
        try:
            return await _settle_task(task)
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"Redis repository {label} failed") from exc
        finally:
            self._operations.discard(tracked_task)

    async def _initialize_client(self) -> _RedisClient:
        client = self._client
        if client is not None:
            return client
        async with self._initialize_lock:
            client = self._client
            if client is not None:
                return client
            client = _load_redis().client.from_url(
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

    async def _commit_encoded(self, checkpoint: EncodedCheckpoint) -> None:
        client = await self._initialize_client()
        exceptions = _load_redis().exceptions
        delay = _SETTLEMENT_INITIAL_DELAY
        outcome_unknown = False
        while True:
            try:
                result = await self._eval_commit(client, checkpoint)
            except Exception as exc:
                if isinstance(exc, (exceptions.ConnectionError, exceptions.TimeoutError)):
                    # EVAL may have committed before its response was lost. Keep
                    # replaying the idempotent transition until Redis supplies a
                    # definitive result; returning here would expose an unknown
                    # write outcome to the caller.
                    outcome_unknown = True
                elif not (
                    outcome_unknown
                    and isinstance(exc, exceptions.ResponseError)
                    and _is_redis_settlement_response(exc)
                ):
                    raise
            else:
                _handle_commit_result(result, checkpoint)
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, _SETTLEMENT_MAX_DELAY)

    async def _eval_commit(
        self,
        client: _RedisClient,
        checkpoint: EncodedCheckpoint,
    ) -> object:
        return await client.eval(
            _COMMIT_SCRIPT,
            1,
            self._state_key(),
            self._checkpoint_field_prefix(checkpoint.checkpoint_id),
            self._head_field_prefix(checkpoint.run_id),
            checkpoint.checkpoint_id,
            checkpoint.run_id,
            str(checkpoint.revision),
            _revision_text(checkpoint.expected_revision),
            checkpoint.digest,
            checkpoint.payload,
        )

    async def _get_head(self, run_id: str) -> Checkpoint | None:
        client = await self._initialize_client()
        head_prefix = self._head_field_prefix(run_id)
        values = _hash_values(
            await client.hmget(self._state_key(), _prefixed_fields(head_prefix, _HEAD_FIELDS)),
            len(_HEAD_FIELDS),
            "run head",
        )
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RepositoryError("stored Redis run head is incomplete")
        stored_run_id = _redis_utf8(values[0], "run id")
        if stored_run_id != run_id:
            raise RepositoryError("stored Redis run id hash collision")
        revision = _redis_revision(_redis_bytes(values[1], "revision"))
        checkpoint_id = _redis_utf8(values[2], "checkpoint id")
        digest = _redis_bytes(values[3], "checkpoint digest")
        payload = _redis_bytes(values[4], "checkpoint payload")
        checkpoint_prefix = _redis_utf8(values[5], "checkpoint field prefix")
        payload_sha1 = _redis_utf8(values[6], "checkpoint payload SHA-1")
        if checkpoint_prefix != self._checkpoint_field_prefix(checkpoint_id):
            raise RepositoryError("stored Redis run head has an invalid checkpoint link")
        if (
            len(digest) != 32
            or sha256(payload).digest() != digest
            or sha1(payload, usedforsecurity=False).hexdigest() != payload_sha1
        ):
            raise RepositoryError("stored Redis run head has an invalid checkpoint digest")
        checkpoint = decode_checkpoint(payload)
        if (
            checkpoint.id != checkpoint_id
            or checkpoint.snapshot.context.run_id != stored_run_id
            or checkpoint.snapshot.revision != revision
        ):
            raise RepositoryError("stored Redis run head is inconsistent")
        ledger = _hash_values(
            await client.hmget(
                self._state_key(),
                _prefixed_fields(checkpoint_prefix, _CHECKPOINT_FIELDS),
            ),
            len(_CHECKPOINT_FIELDS),
            "checkpoint ledger",
        )
        if any(value is None for value in ledger):
            raise RepositoryError("stored Redis run head has no complete checkpoint ledger entry")
        if (
            _redis_utf8(ledger[0], "checkpoint id") != checkpoint_id
            or _redis_utf8(ledger[1], "run id") != stored_run_id
            or _redis_revision(_redis_bytes(ledger[2], "revision")) != revision
            or _redis_bytes(ledger[3], "checkpoint digest") != digest
        ):
            raise RepositoryError("stored Redis run head and checkpoint ledger differ")
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

    def _state_key(self) -> str:
        return f"jharness:{{{self._namespace}}}:v1:state"

    def _checkpoint_field_prefix(self, checkpoint_id: str) -> str:
        return f"c:{_identifier_hex(checkpoint_id)}:"

    def _head_field_prefix(self, run_id: str) -> str:
        return f"r:{_identifier_hex(run_id)}:"


async def _settle_task(task: asyncio.Task[_T]) -> _T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            continue


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
    return _RedisRuntime(
        cast(_RedisModule, client),
        cast(_RedisExceptions, exceptions),
    )


def _handle_commit_result(result: object, checkpoint: EncodedCheckpoint) -> None:
    if not isinstance(result, (list, tuple)) or not result:
        raise RepositoryError("Redis commit script returned an invalid result")
    items = cast(list[object] | tuple[object, ...], result)
    status = _redis_utf8(items[0], "commit status")
    if status in {"committed", "idempotent"}:
        return
    if status == "revision_conflict":
        if len(items) != 2:
            raise RepositoryError("Redis commit script returned an invalid conflict")
        actual_text = _redis_utf8(items[1], "actual revision")
        actual = None if actual_text == "-1" else _revision_from_text(actual_text)
        raise RevisionConflict(checkpoint.run_id, checkpoint.expected_revision, actual)
    if status == "id_reused":
        raise RepositoryError(
            f"checkpoint id {checkpoint.checkpoint_id!r} was reused with new content"
        )
    if status == "id_hash_collision":
        raise RepositoryError("stored Redis checkpoint id hash collision")
    if status == "run_hash_collision":
        raise RepositoryError("stored Redis run id hash collision")
    if status in {"ledger_corrupt", "head_corrupt"}:
        raise RepositoryError("stored Redis repository data is invalid")
    raise RepositoryError("Redis commit script returned an unknown status")


def _redis_bytes(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise RepositoryError(f"stored Redis {label} is invalid")
    return value


def _hash_values(result: object, expected: int, label: str) -> tuple[object, ...]:
    if not isinstance(result, (list, tuple)):
        raise RepositoryError(f"stored Redis {label} returned an invalid field set")
    values = cast(list[object] | tuple[object, ...], result)
    if len(values) != expected:
        raise RepositoryError(f"stored Redis {label} returned an invalid field set")
    return tuple(values)


def _prefixed_fields(prefix: str, names: tuple[str, ...]) -> list[str]:
    return [f"{prefix}{name}" for name in names]


def _redis_utf8(value: object, label: str) -> str:
    if type(value) is not bytes:
        raise RepositoryError(f"stored Redis {label} is invalid")
    try:
        text = value.decode("utf-8")
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


def _identifier_hex(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _is_redis_settlement_response(error: Exception) -> bool:
    status = str(error).lstrip().upper()
    return status.startswith(
        ("BUSY", "CLUSTERDOWN", "LOADING", "MASTERDOWN", "READONLY", "TRYAGAIN")
    )


def _non_empty_string(value: str, label: str) -> str:
    if not isinstance(cast(object, value), str):
        raise TypeError(f"{label} must be a string")
    value = str.__str__(value)
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


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
