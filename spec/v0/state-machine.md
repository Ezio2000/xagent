# Agent Runtime v0 State Machine

Status values:

- `planning`
- `executing_tools`
- `paused`
- `completed`
- `failed`
- `limit_exceeded`

The loop starts in `planning`.

When the model returns tool calls, the runtime appends the assistant tool-call
message and transitions to `executing_tools`. Each tool call carries
an open non-empty `mode` string. Core runtimes recognize `execute` and `accept`.

When all tool outputs are appended, the runtime returns to `planning`.
Execute-mode calls append observations. Accept-mode calls append acceptance
acknowledgements or rejection errors and complete immediately from the runtime's
perspective.

When the model returns no tool calls, the runtime completes with the model
content parts as the final answer.

When host code requests a boundary pause, or when a tool result requests an
external wait, the runtime transitions to `paused` at the next durable boundary.
Paused state must include a `pause` object with `reason`, `source`,
`resume_status`, optional `wait_id`, and `metadata`. `resume_status` must be
`planning` or `executing_tools`. Resuming a paused `RunSnapshot` through the
strict resume input contract restores `resume_status`, clears `pause`, appends
validated resume messages, and continues from that boundary. Invocation-terminal
snapshots with `completed`, `failed`, or `limit_exceeded` status must not be
resumed.
If the next durable boundary is a terminal state such as `completed`, `failed`,
or `limit_exceeded`, the terminal state wins and the runtime must not convert
the run to `paused`.

When model streaming is enabled, SDKs may emit `model_delta` events between
`model_started` and `model_completed`. These events are live progress only. The
state machine must not append assistant messages or checkpoint streamed partial
content until the complete `ModelResponse` is available.
If host code interrupts model generation before a complete response exists, the
runtime may pause from the previous durable state; partial `model_delta` content
must remain uncommitted.

If external input is inserted while planning, SDKs must append it as an
`external` message, emit `conversation_inserted`, checkpoint, and continue
planning. If a model call is in flight, SDKs may cancel it before committing the
insert. Conversation insertion is independent of pause and tool execution.

The runtime must stop with `limit_exceeded` when any configured limit is
exceeded. SDKs that track model usage must accumulate standard response usage
token fields in `AgentState.total_usage`; `max_total_tokens` is evaluated
against cumulative `usage.total_tokens` after each model response. Later
responses that omit usage or individual usage fields must not clear previously
accumulated fields.

SDKs must emit `state_changed` for every status transition, including
invocation-terminal transitions to `paused`, `completed`, `failed`, and
`limit_exceeded`.

SDKs must expose a durable `RunSnapshot` value containing `AgentState` and
`RuntimeContext`. A `checkpoint` event must be emitted after each model response
is committed to message history, after conversation insertions are committed to
message history, after tool outputs are committed to message history and removed
from `pending_tool_calls`, and after each state transition. Serial tool calls
commit one output at a time; parallel tool batches
commit atomically as specified in `tool-scheduling.md`.
If a boundary pause is requested at a non-terminal model-response boundary, SDKs
must apply the pause before emitting a resumable checkpoint for that boundary;
the paused checkpoint is the durable resume point. A model response that
completes the run is a terminal boundary; terminal state wins as described
above.
If a committed tool result requests external wait or fail-fast behavior, SDKs
must not emit an intermediate checkpoint that can be resumed without that
control decision. The paused or failed checkpoint is the durable resume point
for that committed tool result. `RuntimeContext.started_at` and
`RuntimeContext.deadline` are wall-clock epoch seconds; SDKs may use monotonic
clocks internally for live timeout enforcement.

For paused runs, SDKs should emit `pause_requested`, then `state_changed`, then
`checkpoint`, then `run_paused`, then `run_completed`. For terminal failures and
limits, SDKs should emit `state_changed`, then `checkpoint`, then `error`, then
`run_completed`.

Tool scheduling semantics, including simple parallel execution, are specified in
`tool-scheduling.md`.

Model streaming semantics are specified in `model-stream.md`.

Pause and interrupt semantics are specified in `run-control.md`.

Run traces are specified in `run-trace.md` and `run-trace.schema.json`. A trace
is a compact record of semantic runtime steps and is used for deterministic
replay validation. It must not create additional durable resume boundaries.
