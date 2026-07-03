# State Machine

The v0.1 state machine has six states:

- `planning`: call the model and ask it for either a final answer or tool calls.
- `executing_tools`: execute or accept requested tool calls and append outputs.
- `paused`: invocation-terminal state with a resumable `pause` payload.
- `completed`: terminal state with a final answer.
- `failed`: terminal state for unrecoverable runtime failures.
- `limit_exceeded`: terminal state for iteration, tool-call, token, or timeout
  limits.

State values are closed runtime-owned contract values. Adding a state requires
updating the SDK enum, `spec/v0/state.schema.json`, event and trace schemas,
transition/replay validation, loop dispatch, docs, and conformance cases in the
same change.

Transitions:

```text
planning -> completed
planning -> executing_tools
planning -> paused
planning -> failed
planning -> limit_exceeded

executing_tools -> planning
executing_tools -> paused
executing_tools -> failed
executing_tools -> limit_exceeded
```

`paused` stops the current invocation but is not a failed or completed run. The
snapshot includes `pause.reason`, `pause.source`, optional `pause.wait_id`,
`pause.resume_status`, and `pause.metadata`. `run_snapshot(ResumeInput(...))`
resumes a paused snapshot by restoring `pause.resume_status`, clearing the pause
metadata, and appending validated resume messages from the strict resume input
contract.
Snapshots with
`completed`, `failed`, or `limit_exceeded` are invocation-terminal and are not
valid resume inputs.

Pause is only durable at safe boundaries. A boundary pause is applied before a
new model call, after a committed model response, after a serial tool commit, or
after a full parallel tool batch commit. Interrupting model generation may stop a
stream or model call early, but any partial `model_delta` output remains live UI
progress only and is not appended as an assistant message.
There is no v0 `cancelled` terminal status. Hosts that need a durable user-abort
boundary should pause or interrupt, then discard or retain the paused snapshot
according to application policy. Directly cancelling the SDK task is host-local
control and does not produce a portable resumable state.

Tool calls carry an open non-empty `mode` string. Core runtimes recognize
`execute` and `accept`: execute-mode calls append a `ToolObservation`, and
accept-mode calls append either a `ToolAcceptance` acknowledgement or a
`ToolRejection` error result. Accept mode completes immediately from the
runtime's perspective. Any later external result enters through conversation
insertion, not by reopening the original tool call.

Conversation insertion is a separate planning-time input path. If external
input is inserted while a model call is in flight, the runtime cancels that
model call, appends an `external` message, emits `conversation_inserted`,
checkpoints, and asks the model to plan again. It is not a pause and does not
depend on a tool call.

Every status change emits `state_changed`, including transitions to `paused` and
terminal transitions, and then emits `checkpoint` with the current
`RunSnapshot`. The loop also emits `checkpoint` after committing a model response
to history, after committing conversation inserts, and after committing tool
outputs to history. Serial tools
commit one result at a time. Parallel tool batches commit and checkpoint
atomically after every call in the batch has completed.
If a boundary pause is applied at a model-response boundary, the checkpoint for
that boundary is the resulting `paused` snapshot; the runtime must not expose an
intermediate resumable snapshot that has the model response but not the pause
decision.
If a committed tool result triggers external wait or fail-fast behavior, the
checkpoint for that commit is the resulting `paused` or `failed` snapshot; the
runtime must not expose an intermediate resumable snapshot that has the tool
observation but not the control decision.

Configured timeout limits are enforced as hard async deadlines around model,
tool, and hook awaits. Synchronous hooks run off the event loop so a blocking
hook cannot block the agent loop past the runtime deadline. Timeout is
`limit_exceeded`, not `paused`. Standard model usage token fields are
accumulated in `AgentState.total_usage`; if `LoopLimits.max_total_tokens` is
exceeded after a model response is committed, the run transitions to
`limit_exceeded`. If a later response omits usage or omits an individual usage
field, previously accumulated fields remain unchanged.
Iteration and tool-call limits are capacity limits: when the committed counter
equals the configured maximum, the runtime stops before starting the next
planning or tool step. Token limits are budget limits: `max_total_tokens` trips
only after cumulative `total_tokens` becomes greater than the configured
maximum.

During `executing_tools`, the runtime may run explicitly safe tool calls in
parallel up to `LoopLimits.max_parallel_tool_calls`. Unsafe or undeclared tools
remain serial barriers. Tool completion events may arrive in completion order,
but tool observations are committed to message history in the original
tool-call order. If a timeout or other runtime limit interrupts a parallel
batch, the next checkpoint remains at the last fully committed batch boundary.

SDKs should expose `RunSnapshot` as the durable checkpoint boundary. It contains
`AgentState` plus `RuntimeContext`, including wall-clock `started_at` and
`deadline` fields. Host applications own persistence and decide why to pause,
where to store snapshots, and what messages or callback results to add before
resuming.
Snapshots and hook/event payloads are defensive copies of runtime state. The
reference SDK favors immutable boundaries over structural sharing; long runs
therefore pay copy cost proportional to accumulated message history at each
checkpoint and hook boundary.

SDKs should also expose a compact `RunTrace` for one runtime invocation. Trace
steps record semantic boundaries such as model calls, tool calls, pause, resume,
checkpoints, terminal output, limits, and errors. Replay validators consume
trace records and referenced checkpoint summaries; they do not call live models
or tools and do not define new checkpoint boundaries. Trace records use
metadata key summaries instead of raw host metadata values.
