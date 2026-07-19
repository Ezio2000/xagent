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

The base installation has no database-driver dependency. `MemoryRunRepository` needs
no service, and SQLite uses Python's embedded `sqlite3` module. MySQL and Redis load
their selected client only when that backend is initialized; there is no fallback or
compatibility driver. Their services remain supplied by the application, while the
JHarness development suite exercises them in disposable Docker containers.

The source, contracts, and release process are maintained in the
[JHarness repository](https://github.com/Ezio2000/jharness).
