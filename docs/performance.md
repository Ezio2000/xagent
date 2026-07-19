# Performance Model

Performance requirements are architectural bounds first and wall-clock measurements
second. Timing varies by machine; complexity, allocation, queue, and concurrency
behavior must remain deterministically testable.

## Required Bounds

- `RunHistory` structurally shares a persistent skew-binary forest. Public
  construction, wire decoding, and explicit history replacement validate their
  complete input; proven append-only evolution adds each new message in `O(1)` without
  copying or revisiting the old prefix. Random access is `O(log h)`, complete traversal
  is `O(h)`, and a newest suffix is `O(k)` for the requested suffix size.
- History linkage uses a persistent radix proof and cursor-based unresolved tool calls.
  Fixed-size history and tool-result appends therefore do not depend on the number of
  earlier messages or pending calls.
- A `DurableCommit` carries `Initial`, `Append`, `Replace`, or `Unchanged` history data.
  Repositories encode only checkpoint core data and the accepted delta; they never
  serialize or decode old history on the ordinary commit path.
- Checkpoint idempotency retains one compact semantic fingerprint per run-scoped id.
  Exact non-initial retries are constant-time relative to history size.
- `N` fixed-size append commits perform `O(N)` cumulative history and persistence work.
  Complete recovery/export remains `O(h)`, the size of the value it returns, and an
  explicit replacement is linear in its replacement input.
- Trusted immutable values are not serialized merely to copy them, and Memory retains
  shared domain values instead of round-tripping JSON.
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

User-supplied model, history, approval, batch, repository, and tool implementations
remain responsible for the work they perform. Kernel sends complete current history in
each `ModelRequest`; materializing and encoding one request is `O(h)`, and cumulative
LLM input over `N` fixed-size turns may therefore be `O(N^2)`. That deliberate model
cost is outside the linear state-evolution and persistence guarantees.
