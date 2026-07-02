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

Replay must validate:

- status transitions match the state machine;
- each `state_changed.before_status` matches the current replay status;
- `planning` may transition only to `executing_tools`, `paused`,
  `completed`, `failed`, or `limit_exceeded`;
- `executing_tools` may transition only to `planning`, `paused`, `failed`, or
  `limit_exceeded`;
- `model_call`, `model_delta`, and `model_result` occur only while `planning`,
  and model deltas/results must belong to an open model call;
- `planning -> completed` and `planning -> executing_tools` transitions require
  a preceding, closed `model_result` after the last planning checkpoint;
- `tool_call` and `tool_result` occur only while `executing_tools`, and each
  tool result must match an open tool call;
- `executing_tools -> planning` transitions require a preceding, closed
  `tool_result` after the last executing-tools checkpoint;
- a completed trace must not leave a model call or tool call open;
- each `checkpoint` status matches the current replay status;
- paused checkpoints include pause payloads;
- non-paused checkpoints do not include pause payloads;
- the last checkpoint status matches the invocation-terminal status;
- traces end with `run_completed`;
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
