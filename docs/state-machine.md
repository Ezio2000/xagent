# State Machine

Run lifecycle is a flat discriminated union. Each variant carries only the data
that is valid for that state.

## State Variants

```text
RunState
├── Planning
├── ToolsPending
│   └── calls: non-empty ordered tuple
├── Suspended
│   ├── resume_to: Planning | ToolsPending
│   └── suspension: Suspension
├── Completed
│   └── content: non-empty final content
├── Failed
│   └── error: stable structured error
└── Limited
    └── reason: stable limit reason
```

`Planning` is stateless. `ToolsPending` always has at least one call.
`Suspended.resume_to` stores the exact semantic continuation. Terminal variants
have no outgoing transition. Display status is derived from the variant.

## Legal Transitions

Planning either continues planning, requests tools, suspends, or terminates. Tool work
either leaves a pending suffix, returns to planning, suspends with its exact
continuation, or terminates. Resume restores the continuation saved by `Suspended`.
The complete transition table is normative in
[`contracts/v0/state-machine.md`](../contracts/v0/state-machine.md).

## Durable Boundaries

Every durable change produces one atomic `Checkpoint(snapshot, fact)`. A
checkpoint also occurs when history changes without changing the state variant.
Portable fact kinds are:

- `started` for the initial checkpoint;
- `resumed` for resume acknowledgement;
- `model_turn` for one complete model response;
- `tool_batch` for one serial call or one parallel batch;
- `conversation_insert` for one inserted external message;
- `history_rewrite` for one accepted reduction;
- `control` for suspension, failure, or limit decisions.

There is no durable partial model response, partial tool call, or partial
parallel batch. A fact records only data introduced at that boundary; run id,
revision, metrics, and current state remain authoritative in its snapshot.

## Start, Continue, and Resume

Start creates revision `0` in `Planning`; continue re-enters a nonterminal active
checkpoint without a synthetic transition; resume first persists restoration of a
saved suspension. Terminal checkpoints reject both recovery operations. Recovery
restarts from durable semantics, never an in-flight language-runtime task.

Request validation, selectors, appended messages, and inherited-deadline behavior are
defined in [`contracts/v0/run-control.md`](../contracts/v0/run-control.md).

## Control Boundaries

An `Invocation` accepts three control operations:

- pause the invocation;
- insert one external conversation message;
- request cooperative cancellation of one active tool call.

Pause during model work cancels the operation and commits
`Suspended(Planning)`. Partial model deltas disappear.

Pause during tool execution waits for the selected serial call or parallel
batch to settle and reach an atomic boundary. The resulting suspension preserves
the next legal state. This prevents ambiguous side-effect state.

Conversation insertion interrupts model work, appends one external message,
and commits `Planning`. An insertion received during tool work is held only in
the current invocation until a planning boundary. If the invocation stops first,
the uncommitted input is not durable and the host must resubmit it.

Cooperative tool cancellation affects only the named active call. Unknown,
stale, and completed call ids are no-ops. It does not create a lifecycle state.

Controls are invocation-scoped and cannot accumulate for a later execution.

## Boundary Precedence

When several decisions become observable at one boundary, apply them in this
order:

1. expired hard deadline;
2. terminal outcome produced by the completed step;
3. first committed waiting result in model call order;
4. pause;
5. legal conversation insertion;
6. normal continuation.

This precedence is portable behavior and requires conformance coverage.

## Metrics and Limits

Planning and tool counters advance only in a successfully committed checkpoint:

- a complete committed model response increments `planning_steps`;
- each committed tool result increments `tool_calls`;
- committed usage accumulates field by field.

Capacity limits stop before a next step would exceed its count. Token limits are
evaluated after committing reported usage. Deadline expiration never commits a
partial model response or partial tool batch.

One monotonic deadline covers the entire invocation's model, catalog, approval,
tool, history, and ordinary repository awaits. Cleanup grace is separate and may
only settle owned work and commit `Limited(deadline)`.

## History Reduction

A `HistoryReducer` may propose a non-growing history only at `Planning`. Each accepted
rewrite is its own checkpoint before the next model attempt. The host owns the summary
algorithm; kernel validates and commits the proposal without letting it change state,
context, metrics, or revision.
