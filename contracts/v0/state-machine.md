# Kernel v0 State Machine

## Lifecycle

Portable lifecycle is one flat discriminated union:

```text
Planning
ToolsPending(non-empty ordered calls)
Suspended(resume_to: Planning | ToolsPending, suspension)
Completed(final content)
Failed(error)
Limited(limit reason)
```

Wire kinds are `planning`, `tools_pending`, `suspended`, `completed`, `failed`,
and `limited`. There is no wrapper status, continuation, or outcome object, and
no independent mutable status field.

## Legal Transitions

```text
Planning -> ToolsPending
Planning -> Suspended(resume_to=Planning)
Planning -> Completed | Failed | Limited

ToolsPending -> ToolsPending
ToolsPending -> Planning
ToolsPending -> Suspended(resume_to=ToolsPending | Planning)
ToolsPending -> Failed | Limited

Suspended(resume_to=S) -> S
```

`Completed`, `Failed`, and `Limited` have no outgoing transitions.

## Checkpoints

Every durable change creates exactly one `Checkpoint(id, snapshot, fact)` and
increments snapshot revision exactly once. Fact kinds are `started`, `resumed`,
`model_turn`, `tool_batch`, `conversation_insert`, `history_rewrite`, and
`control`.

A model turn is durable only after `Model.invoke` returns a complete response.
A serial tool call is a tool batch of one. A parallel batch is durable only
after every selected call has a normalized result. A checkpoint stores tool
messages in model call order.

## Metrics

- `planning_steps` increases by one for each committed complete model response.
- `tool_calls` increases by the number of committed tool messages.
- usage accumulates only reported fields from committed model responses.
- counters never decrease.

Interrupted model calls and uncommitted parallel results do not advance
metrics.

## Completion

A complete model response always commits its assistant message, usage, and one
planning-step increment together. Its model-turn fact has a `result` that
determines the after state: `completed` uses its part count, `tools_pending`
uses its ordered call ids, and `limited` records `max_total_tokens`. No separate
terminal checkpoint is required for the same response.

## Suspension

`Suspended.resume_to` stores the exact active state. A waiting tool result is
committed before the suspended checkpoint becomes visible. Its model-visible
outcome is written once to `ToolMessage`; its host-only suspension is written
only to `Suspended`. If calls remain, `resume_to` is `ToolsPending`; otherwise
it is `Planning`.

Approval suspension occurs before any selected call is invoked or committed and
therefore preserves the complete selected prefix in `resume_to=ToolsPending`.

## Failure and Limits

Tool validation, denial, and implementation failures become model-visible tool
outcomes. Model, protocol, and infrastructure failures move to `Failed` when a
terminal checkpoint is possible.

Repository failure is outside the state-machine transition. The attempted
checkpoint is not authoritative and the last successfully committed checkpoint
remains current.
