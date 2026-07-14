# Event Stream

Invocation events are a read-only observation channel. They cannot mutate a
request, model response, tool call, result, change, checkpoint, or another
event.

## One Execution

`Runtime.start`, `continue_from`, and `resume` each return one single-use
`Invocation`. Event and result APIs refer to that invocation's same execution.
Creation-time request validation happens before an invocation exists, so a
rejected request emits no lifecycle events.

- Selecting `events()` before execution starts enables one ordered event
  consumer; `result()` may then await the same execution.
- Selecting `result()` first starts result-only mode with a null event sink;
  subscribing later is rejected.
- Awaiting `result()` repeatedly returns the same value or raises the same
  terminal error.
- Closing the event iterator requests cancellation and settles owned work.
- When execution settles, its live control channel closes, discards uncommitted
  and later control submissions, and never reopens.

The mode rule avoids event allocation for callers that only need the result and
prevents a second execution from being created for observation.

## Envelope

Every domain event contains:

```text
run_id          stable logical run id
invocation_id   id of this start/continue/resume execution
sequence        strictly increasing invocation-local integer
kind            closed event kind
created_at      wall-clock observation timestamp
data            immutable JSON payload
```

`phase` is derived from `kind` and is not stored. Schema version belongs to the
portable wire document envelope, not every domain object. Durable run ordering
uses snapshot revision; event sequence is invocation-local.

## Live Events

Live event kinds are:

- `invocation_started`;
- `model_started`, `model_delta`, `model_finished`;
- `approval_requested`, `approval_decided`;
- `tool_started`, `tool_progress`, `tool_finished`;
- `tool_cancel_requested`;
- `invocation_stopped`.

`model_finished` and `tool_finished` mean the external operation settled. They
do not imply durability. A deadline, cancellation, repository failure, or
atomic-batch failure can leave those observations outside durable history.

For tools, the active-concurrency permit is held from `tool_started` through
`tool_finished`. Physical finish order remains observable while durable messages
are ordered by model call position.

`model_delta` and `tool_progress` are explicitly lossy. Runtime retains at most
the configured bounded number of queued lossy events, defaulting to 1024.
Lifecycle, approval, tool start/finish, checkpoint, and stop events are
lossless. A slow consumer therefore cannot create unbounded streaming memory.

Consumers must never use live events as resume state.

## Checkpoint Event

After `RunRepository.commit(checkpoint)` succeeds, runtime emits:

```text
checkpoint_committed
```

Its payload contains:

- checkpoint id;
- the compact semantic fact;
- a compact after `RunView` with revision, state kind, pending call ids,
  counters, usage, and terminal or suspension presence.

It does not repeat full message history or arbitrary suspension metadata. The
full checkpoint remains available from the invocation result and repository.
Observation is not a persistence API.

No checkpoint event is emitted before repository success. Retrying an identical
checkpoint after an ambiguous adapter failure may resolve idempotently, but the
invocation emits the event at most once.

## Delivery and Cancellation

- Events from one invocation are delivered in sequence order.
- An event consumer cannot prevent or roll back an accepted checkpoint.
- Event payloads are immutable and directly reuse trusted domain values where
  their wire shape permits.
- Cancellation returns the last committed snapshot during cleanup; it does not
  invent a durable cancelled state.
- Invocation termination drains the live control channel. A later pause,
  insertion, or tool-cancellation request is ignored instead of being retained.
- Application-specific events use an application channel, not an extensible
  core-event namespace.

Applications may translate events to SSE, WebSocket, logs, metrics, or their
own event bus. Transport failure does not mutate runtime semantics.

## Diagnostics

The optional diagnostics component compacts invocation events into one trace. The verifier
treats committed facts and after views as durable; live progress cannot advance
state. Derived step ids, repeated before views, and repeated final summaries are
not trace fields.
