# State Machine

The v0.1 state machine has six states:

- `planning`: call the model and ask it for either a final answer or tool calls.
- `executing_tools`: execute requested tool calls and append observations.
- `paused`: invocation-terminal state with a resumable `pause` payload.
- `completed`: terminal state with a final answer.
- `failed`: terminal state for unrecoverable runtime failures.
- `limit_exceeded`: terminal state for iteration, tool-call, or timeout limits.

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

Every status change emits `state_changed`, including transitions to `paused` and
terminal transitions, and then emits `checkpoint` with the current
`RunSnapshot`. The loop also emits `checkpoint` after committing a model response
to history and after committing tool observations to history. Serial tools
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
`limit_exceeded`, not `paused`.

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

SDKs should also expose a compact `RunTrace` for one runtime invocation. Trace
steps record semantic boundaries such as model calls, tool calls, pause, resume,
checkpoints, terminal output, limits, and errors. Replay validators consume
trace records and referenced checkpoint summaries; they do not call live models
or tools and do not define new checkpoint boundaries. Trace records use
metadata key summaries instead of raw host metadata values.
