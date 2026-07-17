# Kernel v0 Run Control

## Invocation Creation

The portable operations are:

```text
Runtime.start(messages, context?) -> Invocation
Runtime.continue_from(checkpoint) -> Invocation
Runtime.resume(checkpoint, selector?, append_messages, metadata) -> Invocation

Invocation.result() -> Checkpoint
Invocation.events() -> async Event stream
```

`run-request.schema.json` is the equivalent cross-process request wire. Start
accepts new history. Before creating an invocation, start synchronously rejects
an empty history or a history that is not valid in `Planning`; rejection emits
no invocation event and commits nothing. Continue and resume accept the complete
recovered `Checkpoint`, never a detached snapshot or revision tuple.

Start commits revision `0` before external work. Continue accepts a checkpoint
whose state is `Planning` or `ToolsPending`. Resume accepts `Suspended`, restores
its exact `resume_to` state, and commits a `resumed` checkpoint before external
work. If the inherited hard deadline has already expired, resume instead commits
`Limited(deadline)` directly under cleanup grace; it must not spend that grace on
an acknowledgement or start external work. `Completed`, `Failed`, and `Limited`
reject continue and resume.

Appended resume messages are valid only when `resume_to` is `Planning`.

Only one invocation may own a logical run id at a time. Continue is crash
recovery, not concurrent execution. Deployment workers fence ownership before
calling the runtime; repository revision comparison prevents a lost update but
cannot undo an external effect attempted by a stale worker.

## Suspension Selector

A resume selector may match reason, source, wait id, and a subset of suspension
metadata. At least one selector field is required. A mismatch rejects the
request without committing a checkpoint.

## Invocation Control

An `Invocation` accepts pause, conversation insert, and active-tool
cancellation while it is executing. These commands are live input, not durable
repository state.

Controls submitted before execution starts are delivered when that invocation
attaches its live control channel. The channel closes when execution settles:
uncommitted controls and later control submissions are discarded.
An invocation never becomes a reusable or durable command queue.

### Pause

Pause during planning cancels `Model.invoke` and commits the previous durable
history as `Suspended(resume_to=Planning)`. Observed deltas are discarded.

Pause during a tool batch waits for that atomic batch boundary and then
suspends the remaining active state. It does not interrupt an ambiguous
side-effect boundary.

### Conversation Insert

An insert during planning cancels the in-flight model operation, appends one
external message, commits a `conversation_insert` checkpoint, and continues
planning.

An insert received during tool execution remains queued until `Planning`. It
never appears between an assistant tool request and its tool messages. If the
invocation suspends or ends first, the host resubmits the uncommitted insert;
the runtime does not hide a durable command queue.

### Tool Cancellation

Cancellation is cooperative and active-call scoped. Unknown, stale, and
completed ids are no-ops. Cancellation does not create a lifecycle state and
does not forcefully terminate host-owned work.

## Deadline

One monotonic work deadline bounds model, tool, approval, history, and ordinary
repository work. It never commits partial model output or a partial parallel
batch.

After expiry, runtime may use a small fixed cleanup deadline only to cancel owned tasks
and attempt a terminal `Limited(deadline)` checkpoint. That
grace cannot start external work or extend the run deadline. If the terminal
commit fails, the last committed checkpoint remains authoritative.

## Boundary Precedence

When decisions meet at one legal boundary:

1. expired hard deadline;
2. completed terminal result produced by the current step;
3. first waiting tool outcome in model order;
4. invocation pause;
5. legal queued conversation insert;
6. normal continuation.

This precedence is portable and does not depend on callback timing after a
boundary is already committed.
