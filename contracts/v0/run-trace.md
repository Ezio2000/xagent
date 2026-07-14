# Kernel v0 Run Trace

A run trace is a compact diagnostic artifact for one invocation. It is not a
repository, command queue, or recovery input.

## Shape

```text
Trace
├── header(run_id, invocation_id, request_kind, metadata_keys)
└── entries[]
```

`build_trace` consumes invocation events, validates their common envelope, and
stores that envelope once in the header. Each entry is the event sequence,
kind, timestamp, and data. Entries have no wrapper id, per-entry prior view, or
terminal-view mirror.

## Ordering

Entry sequences are strictly increasing and identify their position. A trace
starts with `invocation_started`, may contain live operation entries and
`checkpoint_committed` entries, and ends with `invocation_stopped`.

Checkpoint entries contain checkpoint id, fact, and the resulting `RunView`.
Their revisions are strictly increasing. The starting view in
`invocation_started` initializes verification for continue and resume; start
uses a null starting view.

## Verification

`verify_trace` never invokes live models or tools and does not reconstruct full
message bodies. It walks entries once, retaining the current compact view, and
verifies:

- every committed revision follows the current durable revision;
- each fact kind is legal for the current lifecycle state;
- the supplied after view equals the deterministic fact transition;
- model completion increments planning steps exactly once;
- tool-batch checkpoints increment tool calls by committed outcome count;
- live deltas and progress do not advance durable state;
- parallel physical completion may differ, but every live completion must
  match exactly one committed call and every non-failure outcome requires a
  live completion; a failure without tool lifecycle entries is a precomputed
  binding or approval failure, and committed call sequence preserves model
  order;
- resume first restores the exact `Suspended.resume_to` state, except that an
  already-expired inherited deadline may directly commit `Limited(deadline)`;
- terminal state has no later checkpoint;
- `invocation_stopped.last_checkpoint_id` identifies the final committed entry.

The operation intentionally omits enough payload that it cannot recreate a
recovery checkpoint.

## Payload Policy

Traces contain ids, counts, kinds, metadata-key summaries, compact facts, and
compact state views. They do not copy complete message history, content bodies,
provider metadata, arbitrary suspension metadata values, or large tool output.
