# Repository Implementations

`jharness-repository` supplies four implementations of the kernel
`RunRepository` protocol. The kernel still uses its lightweight invocation-local
repository when `Runtime(repository=...)` is omitted; persistence is always an
explicit host choice.

| Class | Storage | Service required | Lifecycle |
| --- | --- | --- | --- |
| `MemoryRunRepository` | Process memory | No | None |
| `SQLiteRunRepository` | SQLite database | No; Python embeds SQLite | `initialize()` / `close()` |
| `MySQLRunRepository` | MySQL InnoDB tables | MySQL endpoint | `initialize()` / `close()` |
| `RedisRunRepository` | Redis hashes | Redis endpoint | `initialize()` / `close()` |

Install `jharness-repository` alone for Memory and SQLite. MySQL and Redis are strict,
explicit extras rather than base dependencies:

```bash
uv add jharness-repository
uv add "jharness-repository[mysql]"
uv add "jharness-repository[redis]"
```

All four classes remain importable from `jharness.repository` in a base installation.
Initializing a remote backend without its selected extra raises `RepositoryError` with
the exact installation command; no alternate driver or compatibility path is tried.

All four implementations enforce the same rules from the normative
[`repository` contract](../contracts/v0/repository.md): one atomic checkpoint, strict
revision compare-and-set, a globally scoped checkpoint-ID ledger, exact retries before
revision checks, and no detached write after cancellation. Each concrete class also
offers `await get_head(run_id)` for host-owned recovery.

## Memory

Use memory storage for tests, local composition, or a process whose state is not
expected to survive restart:

```python
from jharness.kernel import Runtime
from jharness.repository import MemoryRunRepository

repository = MemoryRunRepository()
runtime = Runtime(model=model, repository=repository)
```

The class is thread-safe and supports multiple run IDs, but it is neither shared
between processes nor persistent.

## SQLite

SQLite uses the standard-library `sqlite3` module and performs blocking work on a
repository-owned worker, so no SQLite package or daemon is installed on the host:

```python
from jharness.repository import SQLiteRunRepository

async with SQLiteRunRepository("var/jharness/runs.sqlite3") as repository:
    runtime = Runtime(model=model, repository=repository)
    checkpoint = await runtime.start(messages).result()
    assert await repository.get_head(checkpoint.snapshot.context.run_id) == checkpoint
```

The adapter creates versioned `jharness_v1_*` tables lazily, uses WAL with full
synchronous writes, and coordinates writers from separate repository instances and
processes. Keep an in-memory database alive by retaining one repository instance with
`":memory:"`.

## MySQL

The selected database must already exist and the configured user must be able to
create and update InnoDB tables:

```bash
uv add "jharness-repository[mysql]"
```

```python
from jharness.repository import MySQLRunRepository

async with MySQLRunRepository(
    host="mysql.internal",
    port=3306,
    user="jharness",
    password=mysql_password,
    database="jharness",
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

`table_prefix` defaults to `jharness`. Identifiers are indexed by their SHA-256 keys
while the original values are retained and verified, avoiding MySQL index-length
limits without treating a hash collision as equality. Durable table names include the
adapter schema version (`v1`); PyMySQL calls run in a bounded repository-owned executor.

## Redis

Redis uses one versioned namespace hash. A Lua script validates the ID ledger, the
complete current head, and the run revision before publishing the new ledger and head
with one final `HSET` command:

```bash
uv add "jharness-repository[redis]"
```

```python
from jharness.repository import RedisRunRepository

async with RedisRunRepository(
    "redis://redis.internal:6379/0",
    key_prefix="production-agent",
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

The namespace hash key includes the adapter schema version (`v1`); checkpoint and run
IDs become bounded SHA-256 field prefixes while their original values remain stored
and collision-checked. Configure Redis persistence, replication, authentication, TLS,
and backup policy for the deployment's durability requirements; the adapter does not
silently change server policy or apply a TTL.

## Lifecycle and Cancellation

Database repositories initialize lazily, so calling `commit()` or `get_head()` without
an earlier `initialize()` is valid. Prefer an async context manager, or call `close()`
when the repository will no longer be used. Closing rejects new work and waits for
already accepted operations.

If cancellation arrives after a database operation has been submitted, the adapter
waits until that operation has a definitive success or failure instead of returning an
unknown outcome. Set backend connection/socket timeouts and `Runtime`'s
`repository_timeout` together; settling an in-flight backend transaction can delay
delivery of cancellation. If MySQL or Redis may have committed but its response is
lost, the adapter replays the same globally idempotent checkpoint until the backend
returns a definitive result. A prolonged backend outage can therefore keep that
operation pending beyond one configured socket timeout instead of reporting an
ambiguous failure.

## Integration Tests

The default test suite covers memory and SQLite without external services. MySQL and
Redis integration tests activate only when `JHARNESS_TEST_MYSQL_URL` and
`JHARNESS_TEST_REDIS_URL` are set. Project development uses disposable official Docker
images for those endpoints and does not install either server into the operating
system.
