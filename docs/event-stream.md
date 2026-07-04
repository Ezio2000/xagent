# Event Stream

The SDK emits neutral runtime events. Applications can adapt these events to
SSE, WebSocket, logs, CLI output, or tests.

Known v0.1 event types:

- `run_started`
- `state_changed`
- `model_started`
- `model_delta`
- `model_error`
- `model_completed`
- `tool_started`
- `tool_progress`
- `tool_cancel_requested`
- `tool_completed`
- `approval_requested`
- `approval_completed`
- `background_task_started`
- `background_task_updated`
- `background_task_completed`
- `child_run_started`
- `child_run_completed`
- `conversation_inserted`
- `pause_requested`
- `checkpoint`
- `final`
- `error`
- `run_paused`
- `run_completed`

Event `type` is an open string. SDKs should expose these known constants.
Known core event types are runtime-owned: hooks must not replace core events or
emit custom events using those type strings. A hook object that implements
`on_event(event, context, emitter)` receives an `EventEmitter` and may call
`emitter.emit(type, data)` to append non-core events to the same ordered stream.
Other hook methods do not receive an emitter; they should encode data in
returned protocol objects or emit from a later `on_event`.
If an SDK allows `on_event` to return a replacement for a non-core event, the
runtime-owned envelope remains authoritative: `run_id`, `sequence`,
`created_at`, and `schema_version` stay unchanged, and only the replacement
`type` and `data` are used. Prefer `EventEmitter` for custom progress events.
Custom events emitted by hooks are passed through `on_event` like any other
event, so hooks can observe or rewrite custom event chains. SDKs should bound
that cascade so a hook that emits a new event for every observed event cannot
loop forever.

`model_delta` is emitted only when model streaming is enabled and the model
adapter supports it. It is live rendering progress, not durable state. Known
payload kinds are standardized in `contracts/v0/events.schema.json` and
`contracts/v0/model-stream.md`: `text_delta`, `tool_call_delta`, `reasoning_delta`,
and `usage_delta`.

`model_error` is emitted when a model attempt raises a structured provider
error. It closes that attempt for event and trace accounting. If
`data.retry` is true, another `model_started` may follow in the same planning
iteration with the same `data.iteration` value; otherwise the run transitions to
`failed`. The terminal `error` event is still emitted only after a terminal
checkpoint.

`tool_started` and `tool_completed` include the normalized tool invocation
`mode`. A tool may emit `tool_progress` through its tool execution context while
it runs. Progress events are live UI/audit events only; they are not durable
message history and are not required for resume. Hosts may request cooperative
tool cancellation through `RunController.cancel_tool(...)`, which emits
`tool_cancel_requested` and makes `context.cancel_requested` visible to the
tool when the requested tool call is active. Requests for unknown, stale, or
already completed tool-call ids are no-ops. Runtime cancellation is
cooperative: core does not forcefully terminate subprocesses, threads, network
requests, or host-owned workers. For core-known
modes, `tool_completed.data.result.result_kind` is `observation` for
execute-mode output and either `acceptance` or `rejection` for accept-mode
output; extension modes use non-empty custom result kinds. When a tool result is
produced by runtime policy or runtime validation rather than the tool
implementation, such as approval denial or input-schema rejection,
`implementation_invoked` is `false`.

`background_task_started`, `background_task_updated`, and
`background_task_completed` are emitted when a tool result carries a
`BackgroundTask` reference. These events standardize task identity, lifecycle,
and tool-call correlation. `BackgroundTask.status` is host-owned and open; the
small `lifecycle` field chooses which core event is emitted. The host still owns
the queue, worker, retry policy, callback transport, and task storage. Later
worker updates are host events or normal resume/conversation-insert inputs, not
runtime-managed background execution. A waiting tool result should still use the
normal pause/resume protocol for durable continuation.

`child_run_started` and `child_run_completed` are emitted only for runs whose
`RuntimeContext` declares a `parent_run_id`. `parent_tool_call_id` and
`run_kind` are child-run fields and require `parent_run_id`. These events
standardize parent-child run correlation for subagents or delegated work
without adding a scheduler, workspace manager, or permission-inheritance policy
to core.

`approval_requested` and `approval_completed` are emitted when an
`ApprovalPolicy` is configured. They record the host decision point before the
tool implementation is called. `allow` continues to normal tool execution,
`deny` commits a tool error or rejection without calling the tool, and `pause`
uses the normal pause sequence before the pending tool call is resolved.

`conversation_inserted` is emitted when host or external input preempts
planning and enters message history as an `external` message. The event carries
the normalized insertion payload and the message that was appended. It is
followed by a checkpoint before the runtime asks the model to plan again.

A `checkpoint` event carries a full `RunSnapshot` payload after durable state
commits and state transitions so host applications can persist resumable
progress. Host applications should persist `checkpoint` for resume and treat
`model_delta` as optional UI progress.

`pause_requested` records the core pause request that is being applied. Its
`origin` is `control` for controller/host pauses and approval pauses, and
`tool_result` for pauses carried by committed tool results. Approval pauses use
`request.source: approval` and `request.wait_id` equal to the tool-call id;
`request.source` otherwise remains a public label. It is followed by
`state_changed` to `paused` and a `checkpoint` carrying the paused snapshot. A
clean pause emits `run_paused` instead of `final` or `error`; it means the
current invocation stopped cleanly and can be continued with
`run_snapshot(ResumeInput(...))`. If a hook fails after the paused checkpoint,
the invocation may end with `error` and `run_completed`, with or without a
visible `run_paused`, while the last checkpoint remains paused. The host still
owns user-message policy, approval UI, queues, callbacks, storage, and any
external task execution.

Events should be JSON-serializable. They should not include the full message
history except for `checkpoint`, whose purpose is durable persistence. Final
output events may include final content parts because they are the run output.
`run_started` and `run_completed` carry compact state summaries: counts, message
roles, pending tool-call ids, status, final/error presence, and the current
pause state when paused, rather than full message bodies. Pause state in events
includes host-supplied pause metadata; trace payloads compact that metadata to
key summaries.

Run trace is separate from the event stream and from an optional durable event
journal. Core SDKs may derive trace steps from known core events and resume
inputs, but trace records are not emitted as additional runtime events. This
keeps event consumers stable while giving tests, debuggers, and conformance
runners a deterministic replay surface.

Every event envelope includes:

- `type`: open event type string.
- `data`: event payload.
- `run_id`: stable id shared by all events from a run.
- `sequence`: monotonically increasing run-local sequence.
- `created_at`: wall-clock timestamp.
- `schema_version`: event schema version.
