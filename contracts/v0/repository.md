# Kernel v0 Repository

`Repository` persists one complete durable unit:

```text
Checkpoint
├── id                 idempotency key
├── snapshot           complete recovery value
└── fact               compact semantic fact
```

The checkpoint wire version appears only on the checkpoint envelope. Snapshot
owns run identity and revision; fact never mirrors either value.

## Port

```text
commit(checkpoint) -> None
```

Successful `commit` has no portable receipt or durability enum. Returning
`None` means the checkpoint is definitively authoritative. Adapters may expose
backend diagnostics and host recovery reads outside this narrow write port.

## Atomicity and Revision

One successful commit atomically validates and stores the checkpoint. For a
start checkpoint, snapshot revision is `0` and the run must not exist. For
revision `n > 0`, the current stored revision must be exactly `n - 1`.

The prior revision is derived from the new snapshot. It is not repeated as an
`expected_revision` field. Snapshot and fact are never written separately.

## Conflict

A repository rejects a start for an existing run and rejects any nonconsecutive
revision. It must not silently overwrite or merge another writer's checkpoint.

Revision comparison is a durability backstop, not a distributed execution
lease. A host serializes active invocations for each run id. Queue ownership,
leases, and fencing tokens belong to deployment infrastructure.

## Idempotency

Repeating the same checkpoint content with the same id is a successful no-op.
Reusing an id for different checkpoint content is a conflict. Runtime reuses the id
when settling one attempted boundary and generates a new id for a new semantic change.
This closes the ordinary lost-response ambiguity without a receipt or a second
semantic revision.

## Cancellation and Timeout

`commit` is cancellation-safe. If cancellation escapes, the checkpoint must not
become visible later. If persistence may already have crossed its atomic
boundary, the adapter settles the operation before returning success or a
definitive error. It must not detach a write whose outcome is unknown to the
runtime.

Kernel applies a fixed infrastructure timeout to commits. Ordinary commits are
also bounded by the remaining work deadline. After the work deadline, only a
terminal `Limited(deadline)` checkpoint may use the fixed cleanup timeout.
Adapters that cannot settle cancellation must add an id ledger or equivalent
mechanism before implementing this port.

## Failure

Repository failure leaves the prior checkpoint authoritative. The runtime does
not emit `checkpoint_committed` and does not create another unpersisted terminal
transition in response to that failure.

## Facts

The closed fact kinds are:

- `started`
- `resumed`
- `model_turn`
- `tool_batch`
- `conversation_insert`
- `history_rewrite`
- `control`

Facts contain only compact semantic deltas and stable operation ids. Snapshot
owns full history, metrics, context, lifecycle state, run id, and revision.

`RunView` is the deterministic compact projection of snapshot revision, history
count, metrics, and state. A `checkpoint_committed.after` value must equal the
projection of the committed checkpoint's snapshot.

## Observation

After repository success, the runtime emits one live `checkpoint_committed`
event containing checkpoint id, fact, and the resulting compact `RunView`.
Event delivery is not part of repository atomicity and is not a substitute for
persistence.
