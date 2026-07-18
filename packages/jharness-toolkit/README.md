# jharness-toolkit

JHarness tool registration, JSON Schema validation, Python function adaptation,
retry, and circuit-breaking utilities.

```bash
pip install jharness-toolkit
```

```python
from jharness.toolkit import ToolRegistry
```

Installing this distribution installs the matching `jharness-kernel` version.

`RetryingTool` accepts an explicit tuple of retryable exception classes. Retries are
available only for tools whose execution facts declare `idempotent=True`; settled
tool failures are returned immediately. Exhaustion raises `RetryExhaustedError` with
the ordered attempt errors, and retry delays use bounded exponential backoff with
jitter while observing cooperative cancellation.

`CircuitBreakingTool` is a closed/open/half-open circuit. An open circuit rejects
calls until `recovery_timeout_seconds` elapses, then permits one probe. Success resets
the circuit and failure reopens it. These decorators are in-memory policies for one
process and are not a substitute for a distributed rate limiter or durable breaker.
