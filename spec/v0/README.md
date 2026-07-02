# Agent Runtime v0 Contract Map

`spec/v0` is the cross-language contract surface. Schemas define portable JSON
wire shapes. Markdown files define semantic rules that JSON Schema cannot
express clearly, such as event order, resumability, replay invariants, and
runtime boundary priority.

## Normative Documents

- `state-machine.md`: canonical statuses, allowed transitions, durable
  checkpoint boundaries, and terminal invocation states.
- `run-control.md`: pause, interrupt, resume input, timeout priority, and
  external-wait behavior.
- `tool-scheduling.md`: serial versus parallel tool scheduling, batch commit
  order, checkpoint atomicity, and tool error behavior.
- `model-stream.md`: live model delta semantics and the rule that partial
  streamed content is never durable state.
- `run-trace.md`: compact trace records and deterministic replay validation.

## Wire Schemas

- `messages.schema.json`: model-neutral message and content-part wire shape.
- `model-request.schema.json` and `model-response.schema.json`: provider-neutral
  model adapter boundary.
- `model-error.schema.json`: structured provider/runtime model errors.
- `tools.schema.json` and `tool-result.schema.json`: tool specs, tool calls, and
  compact tool results.
- `state.schema.json`, `runtime-context.schema.json`, and
  `run-snapshot.schema.json`: durable checkpoint state.
- `resume-input.schema.json`: strict resume boundary input.
- `events.schema.json`: runtime event stream envelope and payload summaries.
- `run-trace.schema.json`: compact run-trace envelope and step payloads.
- `limits.schema.json`: loop limits and scheduling controls.

## Boundary Rules

The v0 documents share these boundary decisions:

- `checkpoint` is the only durable resume boundary. `model_delta` and live
  tool progress are observable events, not durable state by themselves.
- A paused invocation is terminal for that invocation but resumable through
  `resume-input.schema.json` when its snapshot is valid.
- `completed`, `failed`, and `limit_exceeded` snapshots are
  invocation-terminal and are not valid resume inputs.
- Terminal status wins over pending pause. If a boundary both completes or
  limits the run and has a pending pause request, the run must not be converted
  to `paused`.
- Parallel tool batches commit atomically in model-provided order. Observed
  `tool_completed` events for an interrupted uncommitted parallel batch do not
  advance the durable checkpoint.
- Trace replay validates recorded semantic boundaries without calling live
  models or tools. It must reject traces that describe impossible state,
  checkpoint, or commit histories even when the JSON shape is valid.

## Schema Versus SDK Validation

JSON Schema is the shared structural gate. SDK constructors and replay
validators must enforce semantic constraints that are awkward or impossible to
express portably in JSON Schema. Examples include duplicate tool-call id
rejection, restored tool-message adjacency, and replay accounting between
`model_result`, checkpoints, tool results, and `total_tool_calls`.

When a behavior is portable across SDKs, update this directory and the shared
conformance cases together. Python-only conveniences belong in the Python SDK
documentation, not in `spec/v0`.
