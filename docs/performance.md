# Performance Model

Performance requirements are architectural bounds first and wall-clock measurements
second. Timing varies by machine; complexity, allocation, queue, and concurrency
behavior must remain deterministically testable.

## Required Bounds

- Snapshot evolution reuses unchanged immutable values. Public construction, wire
  decoding, and explicit history replacement validate complete history in `O(h)`;
  proven append-only evolution does not revisit existing message content.
- Trusted values are not serialized merely to copy them, and commits do not rebuild
  already validated aggregates.
- The ephemeral repository retains one compact fingerprint per checkpoint id rather
  than historical snapshots.
- Result-only execution allocates no event queue. Event mode bounds lossy deltas and
  progress while keeping lifecycle and checkpoint events lossless.
- Tool schemas compile once per registration. A selected batch creates at most one
  task per member and caps active calls with `max_tool_concurrency`.
- Provider adapters own one stream accumulator; diagnostics store one compact entry per
  event and grow linearly.
- Kernel creates no hidden worker pool; async ports own their blocking and cancellation
  behavior.

## Runtime Smoke Benchmark

Run from the repository root:

```bash
uv run python benchmarks/runtime_smoke.py
```

The benchmark compares identical asynchronous tools under serial and bounded-parallel
scheduling while checking the concurrency ceiling, durable model order, and one atomic
checkpoint per parallel batch.

The cleanup grace is deliberately bounded. A deliberately non-compliant port that
swallows cancellation may outlive the invocation as an abandoned task; the smoke
benchmark does not claim otherwise.

The reported timing belongs to the current machine. The speedup ratio is a regression
signal, not a production latency promise.

## Non-Goals

Kernel does not hide network latency with unbounded queues, background threads,
speculative side effects, or implicit retries. Provider transports and durable
repositories own connection pooling, batching, and backend-specific tuning.
