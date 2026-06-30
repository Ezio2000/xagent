# Agent Runtime v0 Tool Scheduling

Tool calls are serial unless the runtime is configured with
`max_parallel_tool_calls > 1` and every tool in the current scheduling group is
explicitly safe to parallelize.

## Parallel Eligibility

A tool call is parallel-eligible only when its `ToolSpec.annotations` contains:

- `parallel_safe: true`
- `read_only: true`
- `idempotent: true`

Unknown tools and tools missing any of those annotations are serial barriers.

## Batching

SDKs process `pending_tool_calls` in model-provided order. A consecutive run of
parallel-eligible calls may execute concurrently up to
`max_parallel_tool_calls`. A non-eligible call runs alone and separates the
parallel batches before and after it.

Example:

```text
[safe A, safe B, unsafe C, safe D, safe E]
```

is scheduled as:

```text
parallel(A, B)
serial(C)
parallel(D, E)
```

## Events And Commits

`tool_started` and `tool_completed` events may be emitted in runtime execution
order. Event payloads should include stable tool-call identity and may include
batch metadata such as `batch_id`, `parallel`, and `index`.

Tool observation messages must be committed to message history in the original
model-provided tool-call order. Serial calls may commit and checkpoint one at a
time. Parallel batches must commit and checkpoint atomically after every call in
the batch has completed.

`checkpoint` events represent durable resume points. For a parallel batch, tool
results are not durable until every call in the batch completes and the batch is
committed in model-provided order. If a timeout or other runtime limit interrupts
a parallel batch, the checkpoint state remains at the last fully committed batch
boundary. Hosts may have observed `tool_completed` events for uncommitted calls,
but resume is allowed to rerun those calls because parallel eligibility requires
tools to be read-only and idempotent.

## Errors

When `stop_on_tool_error` is false, tool failures are committed as tool
observations with error metadata and the loop returns to `planning`.

When `stop_on_tool_error` is true, SDKs must use serial tool execution. This
keeps fail-fast behavior, message order, and checkpoint/resume semantics
unambiguous.
