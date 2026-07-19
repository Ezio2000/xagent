# Kernel v0 Tool Scheduling

## Binding

One immutable tool catalog is opened per invocation. Each pending call is bound
before approval. Binding validates catalog membership and input schema and
captures the exact implementation and spec.

Invalid calls become precomputed tool failures and do not reach approval.

## Selection

A batch policy receives only the pending prefix bounded by the remaining
tool-call budget and maximum batch size, then selects one non-empty prefix of
that candidate window. Kernel validates:

- the selected ids and values exactly match that prefix;
- a serial batch contains exactly one call;
- a parallel batch contains only tools whose specs declare parallel,
  read-only, and idempotent execution;
- size and active concurrency limits are respected.

The policy cannot execute calls, emit tool completion, or replace results.

## Approval

Valid bound calls are submitted as one ordered request tuple. The policy returns
one allow, deny, or suspend decision per request.

If any decision suspends, no call in the selected batch is invoked or committed.
The first suspension in model order controls the state. Denials become
precomputed tool failures. Allowed bindings retain the selected serial or
parallel schedule.

## Execution

Each allowed binding is invoked exactly once as one logical call. Tool
decorators may perform internal retry but expose one result to the kernel.

Implementation exceptions and output-schema failures normalize to failure
outcomes. Runtime cancellation and hard deadline expiration remain control
flow.

Tool progress and physical completion are live observations. They are not
durable until the batch transition commits.

An active-concurrency permit spans the portable observation interval from
`tool_started` through `tool_finished`. A permit is not reused for another call
until the prior call's queued progress has been observed and its
`tool_finished` event has been emitted.

## Commit

Results are ordered by pending-call position, converted to tool messages, and
committed in one checkpoint. A serial call is a batch of one. A parallel batch
is one atomic verification unit.

`ToolResult.outcome` is the one model-visible representation and is written
unchanged to `ToolMessage.outcome`. A waiting result additionally carries a
host-only suspension. The runtime stores that value in `Suspended` state and
does not copy it into the message.

When a committed result waits, the first waiting result in model order controls
the suspension. All batch messages remain committed. `resume_to` is
`ToolsPending` when calls remain and `Planning` otherwise.

Without waiting, the next state is `ToolsPending` when calls remain and
`Planning` otherwise.

## Interruption

An interrupted parallel batch commits none of its calls. Resume or continue may
invoke the batch again because parallel eligibility requires read-only and
idempotent behavior.
