# Performance Model

Performance requirements are architectural bounds first and wall-clock measurements
second. Timing varies by machine; complexity, allocation, queue, and concurrency
behavior must remain deterministically testable.

## Required Bounds

- Snapshot evolution shallow-copies only changed aggregate tuples. Existing immutable
  messages, parts, context, catalog, and metric values retain object identity.
- A durable change with no history append or replacement reuses the exact history
  tuple and its private history proof. An append validates only appended messages while
  retaining tool-call-id evidence and a rolling 32-byte content digest. Tuple growth
  copies `O(h + a)` references but does not revisit existing message content.
- Public `RunSnapshot` construction, wire decoding, and explicit history replacement
  are trust boundaries and validate the complete resulting history in `O(h)`. Internal
  proven evolution has no public bypass flag or caller-supplied proof.
- Trusted immutable values are passed directly. Runtime does not serialize and decode
  them as a copying mechanism.
- Wire validation happens once at each trust boundary. Commit does not re-freeze or
  reconstruct domain values.
- The default ephemeral repository reuses the snapshot history digest and hashes only
  the remaining checkpoint fields. It retains one compact fingerprint per checkpoint
  id, not historical snapshots.
- Result-only invocation uses a null observation sink and constructs no event values.
- The observation queue exists only when `events()` selects event mode and is released
  when the invocation's sole event iterator finishes.
- A terminated invocation closes and drains its control channel, releases its driver,
  and retains no uncommitted controls.
- Lossy model deltas and tool progress use a bounded queue, defaulting to 1024 entries.
  Lifecycle and checkpoint observations remain lossless.
- Tool registration compiles input and output schemas once. Registration is average
  `O(1)` after compilation; catalog opening is `O(n)` and returns one immutable
  invocation snapshot.
- Binding reuses the immutable call, specification, and implementation reference.
- A selected batch creates at most one task per member and gates active invocations
  with `max_tool_concurrency`.
- Model streaming has one accumulator in the provider adapter. Kernel forwards deltas
  and consumes the complete response; it does not rebuild another response.
- A diagnostic trace stores one compact entry per event and grows `O(e)`, not with
  repeated history snapshots or derived summaries.
- Kernel creates no worker thread pool. Potentially blocking ports are async and
  cancellation-safe.
- Source functions stay within the configured McCabe complexity limit of 10.

These bounds require identity, allocation, queue-capacity, cancellation, and
complexity tests in addition to behavior tests.

The ephemeral repository's new-id fingerprint cost is `O(c)`, where `c` is
non-history checkpoint content; incorporating history is `O(1)` through its proven
digest. The digest is domain-separated and length-prefixed, and its JSON mapping keys
are sorted with null, boolean, integer, float, string, array, and object encoded as
distinct types. Durable repository adapters may reuse an equivalent backend checksum.

## Runtime Smoke Benchmark

Run from the repository root:

```bash
uv run python benchmarks/runtime_smoke.py
```

The benchmark compares identical asynchronous tools under serial and bounded-parallel
scheduling. It verifies:

- the configured active-concurrency ceiling;
- durable output ordered by model call position;
- one atomic checkpoint for a parallel batch;
- at least a 2x parallel speedup for the controlled workload;
- no detached tool tasks after completion or cancellation.

The reported timing belongs to the current machine. The speedup ratio is a regression
signal, not a production latency promise.

## Deterministic Regression Cases

Tests cover:

- result-only execution allocating no event queue;
- terminated invocations retaining no controls;
- a stalled event consumer remaining within the lossy-buffer bound;
- large history updates preserving identity for unchanged messages;
- large tool schemas compiling once per registration;
- repeated checkpoint retries not growing ephemeral repository history;
- trace size increasing linearly across long runs;
- deadline cleanup settling every owned model and tool task within fixed grace.

## Non-Goals

Kernel does not hide network latency with unbounded queues, background threads,
speculative side effects, or implicit retries. Provider transports and durable
repositories own connection pooling, batching, and backend-specific tuning.
