# Kernel v0 Repository

`Checkpoint` remains the one complete portable recovery value:

```text
Checkpoint
├── id                 run-scoped idempotency key
├── snapshot           complete recovery value
└── fact               compact semantic fact
```

The runtime sends that value to persistence inside one validated commit proof:

```text
DurableCommit
├── checkpoint
├── parent_checkpoint_id
├── history_change
└── digest             semantic checkpoint digest
```

`DurableCommit` is not another recovery value, journal entry, receipt, or portable JSON
shape. It supplies the atomic preconditions and trusted history delta that produced the
complete checkpoint.

## Port

```text
commit(durable_commit) -> None
```

Successful `commit` has no portable receipt or durability enum. Returning `None` means
the checkpoint is definitively authoritative. Adapters may expose backend diagnostics
and complete head reads outside this narrow write port.

## History Change

Exactly one closed history change accompanies every commit:

- `initial`: the complete non-empty history for revision `0`;
- `append`: a non-empty ordered suffix plus predecessor count and digest;
- `replace`: a complete non-empty replacement plus predecessor count and digest;
- `unchanged`: predecessor count and digest with no message payload.

The kernel validates that applying the change produces the checkpoint history count and
incremental digest before calling the repository. A repository validates a non-initial
change's predecessor manifest against its authoritative head without decoding or
scanning old messages.

## Atomicity and Revision

One successful commit atomically validates and publishes the checkpoint core, history
change, idempotency ledger entry, and new run head. Readers observe the complete prior
checkpoint or the complete next checkpoint, never a mixed boundary.

For a start checkpoint, snapshot revision is `0`, `parent_checkpoint_id` is null, the
history change is `initial`, and the run must not exist. For revision `n > 0`, all of the
following must match the current head:

- revision `n - 1`;
- `parent_checkpoint_id`;
- predecessor history count;
- predecessor history digest.

A mismatch is rejected without merging or overwriting another writer's checkpoint.
Revision comparison remains a durability backstop, not a distributed execution lease.
Hosts serialize active invocation ownership and keep leases and fencing in deployment
infrastructure.

## Idempotency

The idempotency key is `(run_id, checkpoint_id)`. Repeating the same semantic checkpoint
content with that key is a successful no-op, including after the run has advanced past
that revision. Reusing the key for different content is an error. The same checkpoint id
in a different run is independent and valid.

An exact retry is checked before revision and parent preconditions. A ledger entry is
invalid if its run head is missing, is older than the ledger revision, or names a
different checkpoint at the same revision. Runtime reuses one id while settling an
attempted boundary and generates a new id for a new semantic change.

## Cancellation and Timeout

`commit` is cancellation-safe. If cancellation escapes, the checkpoint must not become
visible later. If persistence may already have crossed its atomic boundary, the adapter
settles the operation before returning success or a definitive error. It must not detach
a write whose outcome is unknown.

Kernel applies a fixed infrastructure timeout to commits. Ordinary commits are also
bounded by the remaining work deadline. After the work deadline, only a terminal
`Limited(deadline)` checkpoint may use the fixed cleanup timeout. An adapter that can
lose a commit response replays or probes the same run-scoped idempotency key until it
obtains a definitive outcome.

## Complexity

For history length `H`, fixed-size accepted delta `D`, and `N` fixed-size append
commits:

| Operation | Required work relative to history |
| --- | --- |
| append commit | `O(D)`; it does not encode, copy, or read the old `H` messages |
| unchanged commit | `O(1)` |
| exact non-initial retry | `O(1)` |
| `N` fixed-size appends | `O(N)` cumulative |
| explicit replacement | `O(size of replacement input)` |
| complete head recovery/export | `O(H)` |

Backend latency and fixed checkpoint-core fields are outside the history variable, but
an adapter must not hide a complete-history read or serialization in an ordinary commit.
Physical garbage collection of superseded replacement generations is separate from the
atomic commit path.

These repository bounds do not cover model input. Kernel materializes the complete
current history for each `ModelRequest`, which is `O(H)` per call and may produce
`O(N^2)` cumulative LLM input over `N` fixed-size turns.

## Failure

Repository failure leaves the prior checkpoint authoritative. Runtime does not emit
`checkpoint_committed` and does not create another unpersisted terminal transition in
response to that failure.

## Facts and Observation

The closed fact kinds remain `started`, `resumed`, `model_turn`, `tool_batch`,
`conversation_insert`, `history_rewrite`, and `control`. Facts contain only compact
semantic deltas and stable operation ids. Snapshot owns full history, metrics, context,
lifecycle state, run id, and revision.

After repository success, runtime emits one live `checkpoint_committed` event containing
checkpoint id, fact, and the resulting compact `RunView`. Event delivery is not part of
repository atomicity and is not a persistence substitute.
