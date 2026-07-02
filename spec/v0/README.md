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

## Contract Index

Each schema owns one portable wire shape. SDKs may expose native classes around
these shapes, but the JSON form is the cross-language boundary.

| File | Owns | Portable Contract |
| --- | --- | --- |
| `messages.schema.json` | Messages, content parts, assistant tool-call shape, and tool-message linkage. | Roles including `external`, part types, tool-call ids/names/modes/arguments, tool-call uniqueness semantics, tool output message linkage, and extension points for future content. |
| `model-request.schema.json` | Runtime-to-model adapter request. | Message history, tool specs, model options, tool choice, response format, and request metadata boundary. |
| `model-response.schema.json` | Model-to-runtime response. | Final content parts, requested tool calls using the shared tool-call shape, finish reason, usage, model id, response id, and response metadata boundary. |
| `model-error.schema.json` | Structured model/provider failures. | Stable message, provider, error code, status code, retryability, request id, and error metadata boundary. |
| `tools.schema.json` | Tool specifications exposed to models. | Tool name, description, supported invocation modes, input/output schema, scheduling annotations, and tool metadata boundary. |
| `tool-result.schema.json` | Tool output. | Execute-mode observations, accept-mode acknowledgements or rejections, extension output kinds, content parts, error/pause boundaries, correlation ids, and output metadata boundary. |
| `limits.schema.json` | Runtime limits and scheduling knobs. | Iteration limits, tool-call limits, timeout, stop-on-tool-error, and max parallel tool calls. |
| `state.schema.json` | Durable agent state. | Status, messages, pending tool calls, counters, final parts, error summary, and pause state. |
| `runtime-context.schema.json` | Runtime invocation context. | Run id, start time, optional deadline, host metadata boundary, and event sequence. |
| `run-snapshot.schema.json` | Durable resume checkpoint. | State plus context at a checkpoint boundary. |
| `resume-input.schema.json` | Host-to-runtime resume boundary. | Snapshot, append-only messages, optional expected-pause selector, and resume metadata. |
| `events.schema.json` | Ordered runtime event stream. | Event envelope, event names, sequence ordering, and compact payload summaries. |
| `run-trace.schema.json` | Compact semantic trace. | Trace envelope, ordered trace steps, status summaries, stable references, and replayable payloads. |

The Markdown files own semantic rules that schemas only partially express:

| File | Owns |
| --- | --- |
| `state-machine.md` | Status meanings, allowed transitions, terminal states, and checkpoint placement. |
| `run-control.md` | Pause, interrupt, conversation insertion, resume, timeout priority, and external waits. |
| `tool-scheduling.md` | Serial and parallel tool execution, batch atomicity, and tool error behavior. |
| `model-stream.md` | Streaming deltas, accumulator behavior, and non-durable partial output. |
| `run-trace.md` | Trace step order, deterministic replay rules, and compact payload policy. |

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

## Portable Versus SDK-Local

Portable contract:

- JSON shapes accepted by the schemas above.
- Status names, event names, trace step names, and transition semantics.
- Durable checkpoint contents and resume-input validation.
- Message, model, tool, limit, pause, snapshot, event, and trace behavior needed
  by conformance cases.
- Replay acceptance or rejection for a compact `run-trace.schema.json` value.

SDK-local detail:

- Native class names, method names, package layout, and helper APIs.
- Exception classes, stack traces, and diagnostic wording outside conformance
  expectations.
- Provider SDK request/response objects and transport-specific errors before
  they are translated into v0 model shapes.
- Persistence stores, approval flows, plugin systems, UI rendering, queues, and
  deployment runtime.
- Internal semantics of `metadata` keys and values. Metadata fields included in
  v0 schemas are still part of the wire shape; SDKs should preserve them where
  the schema includes them. Run traces are the exception: they intentionally
  record stable `metadata_keys` summaries instead of raw metadata values.

## Schema Versus SDK Validation

JSON Schema is the shared structural gate. SDK constructors and replay
validators must enforce semantic constraints that are awkward or impossible to
express portably in JSON Schema. Examples include duplicate tool-call id
rejection, restored tool-message adjacency, and replay accounting between
`model_result`, checkpoints, tool results, and `total_tool_calls`.

When a behavior is portable across SDKs, update this directory and the shared
conformance cases together. Python-only conveniences belong in the Python SDK
documentation, not in `spec/v0`.
