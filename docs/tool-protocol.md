# Tool Protocol

A tool is a model-neutral capability with an explicit invocation mode.

Tools expose:

- `spec.name`
- `spec.description`
- `spec.input_schema`: JSON Schema for invocation arguments.
- `spec.modes`: supported non-empty invocation modes. Core runtimes recognize
  `execute` and `accept`; other modes are extension points handled by tool
  `invoke`.
- `spec.output_schema`
- `spec.annotations`
- `spec.metadata`
- `execute(invocation, context)` when the tool supports `execute`.
- `accept(invocation, context)` when the tool supports `accept`.
- `invoke(invocation, context)` when the tool supports extension modes.

`ToolInvocation` contains `id`, `name`, `mode`, `arguments`, and call metadata.
Model adapters map provider syntax into that shape. For a model-facing operator
such as `accept(web_search({"query": "..."}))`, the normalized runtime shape is
the original tool name plus `mode: "accept"`; core does not create wrapper tool
names such as `accept_web_search`.
Before a tool implementation is called, the runtime validates
`ToolInvocation.arguments` against `spec.input_schema`. Validation failure is an
invalid tool call: execute-mode calls commit an error `ToolObservation`,
accept-mode calls commit a `ToolRejection`, and extension modes commit an error
`ToolOutput` with a custom `tool_error` result kind. The model can observe that
tool error on the next planning turn and recover.

Execute-mode tools return `ToolObservation`:

- `parts`: multimodal content parts appended as the tool observation.
- `metadata`: optional small JSON-serializable details.
- `is_error`: whether the result represents a tool failure.
- `pause`: optional core pause request for external waits.

Accept-mode tools return `ToolAcceptance` when work was accepted, or
`ToolRejection` when the invocation could not be accepted:

- `parts`: short model-visible acknowledgement that the work was accepted.
- `correlation_id`: stable id that a later external insertion can reference.
- `metadata`: host-owned details for hooks/events only.
- `ToolRejection` carries `is_error: true`, optional `correlation_id`, and no
  pause request.

Extension-mode tools return `ToolOutput` with a non-empty custom `kind`. Core
runtime validation preserves the generic output shape and only applies
mode/result-kind coupling to known `execute` and `accept` modes.

Only `parts` and runtime-owned markers such as `result_kind`, `is_error`, and
`correlation_id` are copied into the durable tool message. Tool output metadata
is available to hooks/events during the current invocation, but it is not copied
into model-visible history, checkpoints, or trace payload values.

The portable output shape is specified in `spec/v0/tool-result.schema.json`.

When an execute-mode tool starts external work and needs a callback before the
run should continue, it can return a waiting observation. The runtime commits
the tool observation, applies the external-wait decision, and checkpoints the
resulting `paused` snapshot. The emitted `pause_requested` event uses `origin:
"tool_result"`; `pause.source` remains the public source label carried by the
tool request. It must not emit an intermediate checkpoint that can be resumed
without the external-wait decision. The host owns the external job, callback
transport, persistence, and any context it adds before resuming the snapshot. If
a committed batch contains multiple waiting observations, the first one in
model-provided tool-call order supplies the pause metadata. Tool-result pauses
are boundary waits and cannot interrupt model execution.

Accept mode is different from a waiting observation. It completes immediately
from the runtime's point of view and commits an acceptance or rejection tool
message. Any later external result must enter through the conversation insertion
protocol, not through the original tool call.

The registry exposes neutral `ToolSpec` values. Provider adapters are
responsible for converting those specs to provider-specific tool formats.

## Scheduling

Tool calls are serial by default. The runtime may execute a consecutive batch of
tool calls concurrently only when all of these are true:

- `LoopLimits.max_parallel_tool_calls` is greater than `1`;
- the tool spec declares `annotations.parallel_safe == true`;
- the tool spec declares `annotations.read_only == true`;
- the tool spec declares `annotations.idempotent == true`.

Unknown tools and tools without those annotations are scheduling barriers and
run serially. Parallel execution affects wall-clock scheduling only: tool
results are still committed to message history in the original model-provided
tool-call order. A serial tool call is checkpointed after its result is
committed and any fail-fast or external-wait decision is applied. A parallel
batch is checkpointed only after the full batch is committed and any fail-fast
or external-wait decision is applied, so durable snapshots never expose a
partial parallel batch or a committed wait/error without its control outcome.
When `LoopLimits.stop_on_tool_error` is enabled, tool execution is serial even if
`max_parallel_tool_calls` is greater than `1`; this preserves fail-fast behavior
and unambiguous checkpoints.

Advanced hosts may replace the default scheduler by passing a
`tool_scheduler_factory`. The factory result must implement the scheduler
protocol: `next_batch(calls)` chooses the next ordered batch and
`run_batch(batch, execute, stop_on_error=...)` yields `tool_started` and
`tool_completed` progress. Implementations do not need to inherit from the
default `ToolScheduler` class.

Common scheduling annotations:

```python
ToolSpec(
    name="search",
    description="Search indexed documents.",
    input_schema={"type": "object"},
    annotations={
        "parallel_safe": True,
        "read_only": True,
        "idempotent": True,
    },
)
```
