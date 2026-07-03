# Agent Runtime v0 Run Trace

Run trace is a compact semantic record for one runtime invocation. It is used by
tests, conformance runners, debuggers, and replay validators. It is not a log
store, persistence channel, queue, callback transport, or UI event stream.

The wire shape is specified in `run-trace.schema.json`.

Trace `metadata` is intentionally compact. It stores only `metadata_keys`, never
host-owned metadata values. Step payloads follow the same rule: pause, model,
tool, and usage metadata may be represented by sorted key lists, ids, lengths,
counts, or other stable summaries, but not by raw host or provider objects.

## Step Order

Trace steps must have strictly increasing, unique `step_id` values. A trace
starts with `run_started`, or with `resume` followed by `run_started`.

When `resume` is present:

- `before_status` must be `paused`, `planning`, or `executing_tools`.
- `after_status` must be `planning` or `executing_tools`.
- Non-paused resumes must preserve status.
- `resume.after_status` must match the following `run_started.after_status`.

## Replay Rules

Replay validators must validate a trace without calling live models or tools.
Recorded model and tool results are inputs to replay; they are not regenerated.

Replay must validate the invariants below. A JSON-valid trace that violates any
of these rules is semantically invalid.

### Status And Step Invariants

- status transitions match the state machine;
- each `state_changed.before_status` matches the current replay status;
- each `state_changed` payload `from` and `to` value matches the step
  `before_status` and `after_status`;
- after `run_started` establishes the replay status, each non-transition step
  with `before_status` or `after_status` preserves the current replay status;
- `planning` may transition only to `executing_tools`, `paused`,
  `completed`, `failed`, or `limit_exceeded`;
- `executing_tools` may transition only to `planning`, `paused`, `failed`, or
  `limit_exceeded`;
- `run_started` establishes the initial replay status and must report the same
  status in its payload;
- `resume`, when present, restores only `planning` or `executing_tools`, and
  its restored status must match the following `run_started`.

### Model Invariants

- `model_call`, `model_delta`, `model_error`, and `model_result` occur only
  while `planning`, and model deltas/errors/results must belong to an open
  model call;
- `conversation_insert` occurs only while `planning`;
- a `conversation_insert` may close an in-flight model call without a
  `model_result`, because the model output was cancelled and not made durable;
- a retryable `model_error` closes the failed attempt; a later retry must open a
  new `model_call`;
- `planning -> completed` and `planning -> executing_tools` transitions require
  a preceding, closed `model_result` after the last planning checkpoint;
- `planning -> completed` requires a `model_result` with zero tool calls;
- `planning -> executing_tools` requires a `model_result` with at least one tool
  call;
- after `planning -> executing_tools`, the next checkpoint of any status must
  report `pending_tool_call_ids` whose count matches the preceding
  `model_result.tool_call_count`;
- no `tool_call` step may appear before that checkpoint obligation has been
  satisfied. This includes a host pause that converts the first durable boundary
  into a `paused` checkpoint instead of an `executing_tools` checkpoint;
- compact `model_result.has_tool_calls` must equal
  `model_result.tool_call_count > 0`.

### Tool Invariants

- `tool_call` and `tool_result` occur only while `executing_tools`, and each
  tool result must match an open tool call, including the invocation `mode`;
- a tool call id must not be opened twice in the same execution segment;
- if `pending_tool_call_ids` are known from the last checkpoint, every
  `tool_call` must belong to that pending set;
- parallel tool results may be recorded in completion order, but the set of
  completed result ids must match the pending calls before the runtime returns
  from `executing_tools` to `planning`;
- `executing_tools -> planning` transitions require a preceding, closed
  `tool_result` after the last executing-tools checkpoint;
- `executing_tools -> planning` must leave no open tool calls and must have
  observed all pending tool results for the current execution segment;
- if a tool result carries a pause request, that request must be applied before
  returning to `planning`.
- accept-mode tool results are trace summaries with `result_kind: "acceptance"`
  and a `correlation_id`, or `result_kind: "rejection"` with `is_error: true`;
  neither form may carry a pause.

### Checkpoint And Accounting Invariants

- a trace whose final status is `completed` must not leave a model call or tool
  call open;
- each `checkpoint` status matches the current replay status;
- paused checkpoints include pause payloads;
- non-paused checkpoints do not include pause payloads;
- the last checkpoint status matches the invocation-terminal status;
- `checkpoint.message_count` must not advance durable history after a
  `model_delta` unless a complete `model_result` has been recorded;
- `pending_tool_call_ids` in compact checkpoint state must be unique;
- `final.part_count` must match the final-part count in the completed
  checkpoint;
- `total_tool_calls` must not decrease across reported states;
- `total_tool_calls` must not exceed the baseline from `run_started` plus the
  number of observed `tool_result` steps;
- at each successful commit boundary `executing_tools -> planning`,
  `total_tool_calls` must exactly equal that baseline plus every `tool_result`
  observed so far in the invocation;
- paused, failed, or limit-exceeded traces may report fewer committed tool
  calls than observed `tool_result` steps when those results were not followed
  by a successful durable commit boundary.

### Terminal Invariants

- traces end with `run_completed`;
- `run_completed` must be the final trace step and must report the current
  invocation-terminal status when it carries a state summary;
- `final` is valid only after `completed`;
- `error` is valid only after an invocation-terminal state and its status must
  match that state;
- `pause_requested.origin` determines whether replay should match a controller
  pause (`control`) or a pause-bearing tool result (`tool_result`);
- completed runs end with `state_changed`, `checkpoint`, `final`,
  `run_completed`, except that post-terminal hook failure may replace or follow
  the final event with `error`;
- paused runs end with `pause_requested`, `state_changed`, `checkpoint`,
  `run_paused`, `run_completed`, except that post-terminal hook failure may
  replace or follow `run_paused` with `error`;
- failed and limit-exceeded runs end with `state_changed`, `checkpoint`,
  `error`, `run_completed`, except that post-terminal hook failure may append a
  second `error` before `run_completed`.

Paused traces may come from interrupted in-flight model activity, and failed or
limit-exceeded traces may come from interrupted in-flight model or tool
activity. Replay validators still require a terminal checkpoint, but they must
not require paused, failed, or limit-exceeded traces to satisfy the same
no-open-call rule as `completed` traces.

## Streaming

`model_delta` trace steps are live progress only. They must not become durable
assistant messages. If a checkpoint occurs after stream deltas but before a
complete `model_result`, the checkpoint message count must not exceed the last
durable checkpoint message count observed before those deltas.

## Tool External Wait

A `pause_requested` step with `origin: "tool_result"` must match a preceding,
not-yet-consumed `tool_result` trace step whose result carries the same compact
pause request. In parallel batches, other tool results from the same committed
batch may appear before the `pause_requested` step. Tool observations are
committed before the paused checkpoint, and the paused checkpoint is the durable
resume point. A `pause_requested` step with `origin: "control"` is a
controller/host pause even if its public `source` label is `tool`.

## Payload Size

Trace payloads should contain semantic summaries and stable references, not full
message history or large tool/model blobs. Large values should be represented by
ids, indexes, hashes, or compact summaries.
