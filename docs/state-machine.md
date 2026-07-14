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

| From | To | Cause |
| --- | --- | --- |
| creation | `Planning` | Start checkpoint. |
| `Planning` | `Planning` | Valid history rewrite or conversation insertion. |
| `Planning` | `ToolsPending` | Complete model response requests tools. |
| `Planning` | `Suspended(Planning)` | Pause interrupts model work or stops at a planning boundary. |
| `Planning` | `Completed` | Complete model response is final. |
| `Planning` | `Failed` | Model, protocol, or infrastructure failure. |
| `Planning` | `Limited` | Deadline, planning, or usage limit. |
| `ToolsPending` | `ToolsPending` | A committed prefix leaves pending calls. |
| `ToolsPending` | `Planning` | The final pending batch commits. |
| `ToolsPending` | `Suspended(ToolsPending)` | Approval or boundary pause preserves remaining calls. |
| `ToolsPending` | `Suspended(Planning)` | A final committed waiting result resumes at planning. |
| `ToolsPending` | `Failed` | Protocol or infrastructure failure. |
| `ToolsPending` | `Limited` | Deadline, tool-call, or usage limit. |
| `Suspended(C)` | `C` | Valid resume checkpoint. |

A waiting result that leaves calls behind suspends with
`resume_to=ToolsPending(remaining)`. If it completes the pending list, it
suspends with `resume_to=Planning`.

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

- `Runtime.start(...)` synchronously validates the supplied history against
  `Planning` before creating an invocation. Invalid history emits no event and
  commits nothing. A valid start creates `Planning` and commits revision `0`
  before model work begins.
- `Runtime.continue_from(checkpoint)` accepts a checkpoint whose snapshot is
  `Planning` or `ToolsPending`. It performs no synthetic state change and
  assumes the previous owner has relinquished execution.
- `Runtime.resume(checkpoint, ...)` accepts a checkpoint whose snapshot is
  `Suspended`, validates an optional selector, restores the exact `resume_to`
  state, applies permitted appended messages, and commits that acknowledgement
  before work resumes. If the inherited hard deadline is already expired, it
  commits `Limited(deadline)` directly instead; cleanup grace cannot be used for
  the acknowledgement or new external work.
- Appended resume messages are valid only when `resume_to` is `Planning`.
- `Completed`, `Failed`, and `Limited` reject continuation and resume.

Resume never restores a language-runtime task or in-flight provider/tool attempt. It
restarts from the durable semantic state.

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

Controls may be submitted before invocation execution starts. When that
execution settles, its control channel closes and discards any uncommitted
controls. Later pause, insertion, and tool-cancellation submissions are
ignored; they cannot accumulate for a nonexistent future execution.

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

A `HistoryReducer` may propose a valid history only at `Planning`. A proposal
must contain no more messages; equal count is allowed when message content is
summarized. A successful rewrite is its own checkpoint before model work.

The reducer cannot change context, metrics, state, or revision. The host owns
the summary algorithm; kernel owns its legal boundary, validation, and atomic
commit.
