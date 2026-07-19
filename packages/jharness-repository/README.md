# jharness-repository

Official in-memory, SQLite, MySQL, and Redis implementations of the JHarness
`RunRepository` protocol.

```bash
# Memory and SQLite only
uv add jharness-repository

# Add exactly the remote drivers the application uses
uv add "jharness-repository[mysql]"
uv add "jharness-repository[redis]"
uv add "jharness-repository[mysql,redis]"
```

```python
from jharness.repository import MemoryRunRepository
```

MySQL TLS is explicit and remains lazy with the optional driver:

```python
from jharness.repository import MySQLRunRepository, MySQLTLS

repository = MySQLRunRepository(
    host="mysql.internal",
    user="jharness",
    password=mysql_password,
    database="jharness",
    tls=MySQLTLS(ca="/etc/jharness/mysql-ca.pem"),
)
```

`MySQLTLS` always enables certificate verification and verifies the server identity by
default. Mutual TLS additionally accepts paired `cert` and `key` paths plus an optional
`key_password`. Omitting `tls` leaves TLS policy to the endpoint and PyMySQL defaults.

Standalone Redis and Redis Cluster use the same optional `redis` extra. Select the
cluster-aware client explicitly:

```python
from jharness.repository import RedisRunRepository

repository = RedisRunRepository(
    "redis://redis-cluster.internal:6379",
    cluster=True,
    key_prefix="production-agent",
)
```

In cluster mode, redis-py's `RedisCluster` client owns topology discovery and redirect
routing. Each run's three keys share one hash tag and therefore one slot; different
runs use different tags and can be distributed across slots. The default
`cluster=False` continues to use the standalone client.

The base installation has no database-driver dependency. `MemoryRunRepository` needs
no service, and SQLite uses Python's embedded `sqlite3` module. MySQL and Redis load
their selected client only when that backend is initialized; there is no fallback or
compatibility driver. Their services remain supplied by the application, while the
JHarness development suite exercises them in disposable Docker containers.

All adapters consume kernel `DurableCommit` deltas. Memory shares immutable values;
SQLite, MySQL, and Redis store new history chunks under a v2 physical namespace and do
not read or migrate obsolete v1 data. Complete recovery remains available through
`await repository.get_head(run_id)`.

The source, contracts, and release process are maintained in the
[JHarness repository](https://github.com/Ezio2000/jharness).
