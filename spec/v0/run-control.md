# Agent Runtime v0 Run Control

Run control defines portable pause, interrupt, and conversation insertion
semantics. Concrete SDK handles may be language-specific; the behavior is
shared.

## Pause Request

A run has at most one pending pause request. Submitting a new request replaces
the previous pending request.

Request fields:

- `reason`: non-empty machine-readable reason.
- `source`: non-empty source label such as `host` or `tool`.
- `wait_id`: optional external wait identifier.
- `metadata`: small JSON object owned by the host or tool.
- `interrupt`: whether the request may interrupt an in-flight model call.

`source` is a public label and may be chosen by hosts or tools. SDKs must not
infer runtime origin from `source`; the emitted `pause_requested` event records
`origin: "control"` for controller/host requests and `origin: "tool_result"` for
pauses carried by committed tool results.

## Boundary Pause

A non-interrupt request stops at the next durable boundary. Durable boundaries
include before a model call, after a committed model response, after a committed
serial tool observation, and after a committed parallel tool batch.
If the next durable boundary is a terminal state such as `completed`, `failed`,
or `limit_exceeded`, that terminal state wins and the run is not converted to
`paused`. Pending requests are cleared when the current invocation ends, whether
the request was applied or superseded by a terminal state.

When the request is applied, SDKs emit `pause_requested` with `origin:
"control"`, transition to `paused`, checkpoint the paused snapshot, emit
`run_paused`, and end the current invocation with `run_completed`. The applied
request is cleared.

## Interrupt

An interrupt request may cancel an in-flight model call or model stream. Partial
stream deltas are live progress only and must not be appended to durable message
history. The paused snapshot resumes from the previous durable state, usually
`planning`.

Interrupt does not define user-message policy. Hosts decide whether to resume
the paused snapshot as-is, append new messages, or start a different run.

## Conversation Insert

A conversation insert is host-owned input that enters message history during a
live invocation. It is represented as an `external` message with an insertion
id, source label, optional correlation id, content parts, and metadata.

If an insert arrives while the runtime is planning, SDKs must append the
external message, emit `conversation_inserted`, checkpoint, and plan again. If a
model call or stream is in flight, SDKs may cancel it before committing the
insert; partial model deltas remain non-durable UI progress.

Conversation insertion is independent of pause:

- it does not transition to `paused`;
- it does not require a tool call;
- it may reference an earlier `ToolAcceptance.correlation_id`, but the runtime
  does not require that relationship;
- it is checkpointed as normal message history before the next model call.

## Resume Input

Resuming a durable snapshot is a strict runtime boundary. SDKs must validate the
portable resume input shape in `resume-input.schema.json` before mutating live
state.

Valid resume input contains:

- `snapshot`: the durable `RunSnapshot`.
- `append_messages`: normal message-protocol messages to append while resuming a
  paused snapshot.
- `expected_pause`: required wire key whose value is either `null` or a selector
  for `pause.reason`, `pause.source`, `pause.wait_id`, or pause metadata.
- `metadata`: host-owned resume bookkeeping that is not model-visible.

The schema encodes portable cross-field constraints and SDKs must also enforce
them in typed constructors. `completed`, `failed`, and `limit_exceeded`
snapshots are invocation-terminal and must not be resumed. `planning` and
`executing_tools` snapshots may resume only with empty `append_messages` and
`expected_pause: null`. `paused` snapshots restore `pause.resume_status`, clear
`pause`, clear pause-local error state, append validated `append_messages`, and
then continue from the restored status. If `pause.resume_status` is
`executing_tools`, `append_messages` must be empty so pending tool observations
remain adjacent to the assistant tool-call message. If `expected_pause` is
non-null, it must match the paused snapshot before the run continues.

Resume inputs must preserve tool-call history integrity. Every `tool` message in
the restored history must be part of the contiguous observation block
immediately following the assistant message that declared the matching
`tool_calls`. Completed tool messages must match assistant tool-call order. If
unresolved tool calls remain, `pending_tool_calls` must exactly equal the
unresolved suffix of that assistant message. A resumed `planning` history must
not contain orphan assistant `tool_calls` or orphan `tool` messages.

## Timeout Priority

Runtime timeout and limit checks take precedence over pause. If the run has
already exceeded a configured deadline or limit, SDKs must transition to
`limit_exceeded` instead of `paused`.

## Tool External Wait

A tool result may carry a pause request for external work. The tool observation
is committed first. SDKs must not emit an intermediate checkpoint that can be
resumed without applying the external-wait decision; the paused checkpoint is the
durable resume point for that committed tool result. Tool-result pause requests
are boundary waits, must set `interrupt: false`, and emit `pause_requested` with
`origin: "tool_result"`. If multiple committed tool results in the same batch
carry pause requests, the first result in
model-provided tool-call order is applied and later pause requests in that batch
are ignored. The paused snapshot resumes to `executing_tools` when unexecuted
pending tool calls remain; otherwise it resumes to `planning`.

Callback payloads are not a separate runtime channel. To make external callback
data visible after a paused external wait, hosts must encode that data in the
resume input using the normal message protocol before calling
`run_snapshot(ResumeInput(...))`. For non-paused live invocations, hosts may use
conversation insertion to append the callback data as an `external` message.
`pause.metadata` and resume `metadata` are host-visible bookkeeping and are not
model-visible by themselves.
