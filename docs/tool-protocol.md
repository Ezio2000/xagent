# Tool Protocol

A tool is an executable capability with a model-neutral `ToolSpec`.

Tools expose:

- `spec.name`
- `spec.description`
- `spec.input_schema`
- `spec.output_schema`
- `spec.annotations`
- `spec.metadata`
- `execute(arguments, context)`

Tool results separate model-visible content from host metadata:

- `parts`: multimodal content parts appended as the tool observation.
- `metadata`: optional small JSON-serializable details.
- `is_error`: whether the result represents a tool failure.
- `pause`: optional core pause request for external waits.

Only `parts` and runtime-owned markers such as `is_error` are copied into the
durable tool message. Tool result metadata is available to hooks/events during
the current invocation, but it is not copied into model-visible history,
checkpoints, or trace payload values.

The portable result shape is specified in `spec/v0/tool-result.schema.json`.

When a tool starts external work and needs a callback before the run should
continue, it can return a waiting result. The runtime commits the tool
observation, applies the external-wait decision, and checkpoints the resulting
`paused` snapshot. The emitted `pause_requested` event uses `origin:
"tool_result"`; `pause.source` remains the public source label carried by the
tool request. It must not emit an intermediate checkpoint that can be resumed
without the external-wait decision. The host owns the external job, callback
transport, persistence, and any context it adds before resuming the snapshot. If
a committed batch contains multiple waiting results, the first one in
model-provided tool-call order supplies the pause metadata. Tool-result pauses
are boundary waits and cannot interrupt model execution.

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
