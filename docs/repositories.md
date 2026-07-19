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
[`repository` contract](../contracts/v0/repository.md): one atomic `DurableCommit`,
strict revision/parent/history-base compare-and-set, a run-scoped checkpoint-ID ledger,
exact retries before revision checks, and no detached write after cancellation. Each
concrete class also offers `await get_head(run_id)` for host-owned complete recovery.

Checkpoint remains the complete portable recovery value. The commit proof separately
names an initial, append, replace, or unchanged history mutation, allowing adapters to
persist only new messages. Storage layouts use only the v2 namespace. Obsolete v1
tables and keys are ignored; this package provides no reader, migration, or alias for
them.

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
between processes nor persistent. It retains immutable checkpoint values and shared
`RunHistory` structure directly; commits and reads do not perform a JSON round trip.

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

The adapter creates `jharness_v2_run_heads`, `jharness_v2_checkpoint_ledger`, and
`jharness_v2_history_chunks` lazily. Head rows contain a compact checkpoint manifest;
append commits insert only bounded new history chunks. SQLite uses WAL with full
synchronous writes and coordinates writers from separate repository instances and
processes. Keep an in-memory database alive by retaining one repository instance with
`":memory:"`.

## MySQL

The selected database must already exist and the configured user must be able to
create and update InnoDB tables:

```bash
uv add "jharness-repository[mysql]"
```

```python
from jharness.repository import MySQLRunRepository, MySQLTLS

async with MySQLRunRepository(
    host="mysql.internal",
    port=3306,
    user="jharness",
    password=mysql_password,
    database="jharness",
    tls=MySQLTLS(ca="/etc/jharness/mysql-ca.pem"),
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

Passing `MySQLTLS` explicitly forwards the CA path and enables certificate
verification. Server-identity verification is also enabled by default; set
`verify_identity=False` only when the deployment deliberately cannot match the
certificate identity. For mutual TLS, provide the client certificate and key together:

```python
tls = MySQLTLS(
    ca="/etc/jharness/mysql-ca.pem",
    cert="/etc/jharness/mysql-client.pem",
    key="/etc/jharness/mysql-client-key.pem",
    key_password=mysql_key_password,
)
```

When `tls` is omitted, the adapter does not force TLS or add implicit certificate
settings. The configuration object does not import PyMySQL, so base-package imports
remain driver-free.

`table_prefix` defaults to `jharness`. The adapter creates v2 head, checkpoint-ledger,
and history-chunk InnoDB tables. Identifiers are indexed by their SHA-256 keys while the
original values are retained and verified, avoiding MySQL index-length limits without
treating a hash collision as equality. PyMySQL calls run in a bounded
repository-owned executor.

## Redis

Redis uses three keys per run for head, checkpoint ledger, and history chunks. Their
shared hash tag contains the run hash, so one run's atomic Lua operation stays in one
cluster slot. Different runs have different tags and can therefore occupy different
slots. A read-only probe settles exact retries without encoding or sending history; a
commit script rechecks all preconditions and publishes only the new chunks, ledger
entry, and head:

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

For Redis Cluster, select redis-py's cluster-aware client explicitly; it uses the same
`jharness-repository[redis]` extra:

```python
async with RedisRunRepository(
    "redis://redis-cluster.internal:6379",
    cluster=True,
    key_prefix="production-agent",
) as repository:
    runtime = Runtime(model=model, repository=repository)
```

`cluster=True` delegates topology discovery and `MOVED`/`ASK` routing to
`redis.asyncio.RedisCluster`. The repository itself does not implement redirect
routing. `cluster=False` is the default and continues to construct the standalone
`redis.asyncio` client. Both clients are imported only when the repository initializes.

Keys include the adapter schema version (`v2`); checkpoint and run IDs become bounded
SHA-256 components while their original values remain stored and collision-checked.
Configure Redis persistence, replication, authentication, TLS, and backup policy for
the deployment's durability requirements; the adapter does not silently change server
policy or apply a TTL.

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
lost, the adapter settles the same run-scoped idempotency key until the backend returns
a definitive result. A prolonged backend outage can therefore keep that
operation pending beyond one configured socket timeout instead of reporting an
ambiguous failure.

## Integration Tests

The default test suite covers memory and SQLite without external services. MySQL and
Redis integration tests activate only when `JHARNESS_TEST_MYSQL_URL` and
`JHARNESS_TEST_REDIS_URL` are set. Project development uses disposable official Docker
images for those endpoints and does not install either server into the operating
system. Every integration case owns a random namespace, removes its exact tables or
keys in `finally`, and verifies that the generated data is absent before completion.
