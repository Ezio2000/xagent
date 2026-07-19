# jharness-kernel

The dependency-free JHarness runtime kernel, including lifecycle values, execution,
checkpoints, diagnostics, extension ports, and portable wire codecs.

```bash
uv add jharness-kernel
```

```python
from jharness.kernel import Message, Runtime
```

Kernel owns persistent `RunHistory`, incremental `DurableCommit`, cursor-based pending
tool calls, and complete-history model requests. It has no runtime dependency.

The source, contracts, and release process are maintained in the
[JHarness repository](https://github.com/Ezio2000/jharness).
