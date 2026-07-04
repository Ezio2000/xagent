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
Before a tool implementation is called, `AgentLoop` asks the configured
`ToolRegistryProtocol` to validate the call. The default `toolkit.ToolRegistry`
validates `ToolInvocation.arguments` against `spec.input_schema` with JSON
Schema. Validation failure is an invalid tool call: execute-mode calls commit
an error `ToolObservation`, accept-mode calls commit a `ToolRejection`, and
extension modes commit an error `ToolOutput` with a custom `tool_error` result
kind. The model can observe that tool error on the next planning turn and
recover. The corresponding tool lifecycle event and trace payloads use
`implementation_invoked: false`.

If an approval policy is configured, valid tool calls are sent to it for an
`allow`, `deny`, or `pause` decision before calling the tool implementation.
Invalid tool calls are committed as runtime validation errors and are not sent
to approval policy. `deny` commits a mode-appropriate tool error or rejection
without calling the tool. `pause` stops before tool execution with the call
still pending and resumes through the normal run-control protocol. Approval is
a runtime decision point, not an OS-level sandbox or UI implementation.
Approval-facing risk is derived from `spec.annotations`. If a tool declares a
nested `annotations.risk` object, core validates the standardized fields
`filesystem`, `network`, `subprocess`, `destructive`, and
`requires_approval`, while preserving additional risk fields for host policy.
`filesystem` and `network` are open non-empty strings with recommended values,
not closed vocabularies. If no nested `risk` object is present, the risk summary
passed to the approval policy is `{}`. Scheduling hints such as `parallel_safe`,
`read_only`, and `idempotent` are not approval risk.

Execute-mode tools return `ToolObservation`:

- `parts`: multimodal content parts appended as the tool observation.
- `metadata`: optional small JSON-serializable details.
- `is_error`: whether the result represents a tool failure.
- `pause`: optional core pause request for external waits.
- `background_task`: optional host-owned task reference when the tool accepted
  or started background work.

Accept-mode tools return `ToolAcceptance` when work was accepted, or
`ToolRejection` when the invocation could not be accepted:

- `parts`: short model-visible acknowledgement that the work was accepted.
- `correlation_id`: stable id that a later external insertion can reference.
- `metadata`: host-owned details for hooks/events only.
- `background_task`: optional host-owned task reference.
- `ToolRejection` carries `is_error: true`, optional `correlation_id`, and no
  pause request.

Extension-mode tools return `ToolOutput` with a non-empty custom `kind`. Core
runtime validation preserves the generic output shape and only applies
mode/result-kind coupling to known `execute` and `accept` modes.

Tool implementations receive a `ToolExecutionContext`. They can call
`context.emit_progress({...})` to emit live `tool_progress` events and poll
`context.cancel_requested` to observe host cancellation requests made through
`RunController.cancel_tool(...)`. Cancellation is active-call scoped:
requests for unknown, stale, or already completed tool-call ids are no-ops.
Cancellation is cooperative and does not forcefully stop tool code or
host-owned subprocesses.

Only `parts` and runtime-owned markers such as `result_kind`, `is_error`,
`correlation_id`, and optional `background_task` references are copied into the
durable tool message. Tool output metadata is available to hooks/events during
the current invocation, but it is not copied into model-visible history,
checkpoints, or trace payload values.

The portable output shape is specified in `contracts/v0/tool-result.schema.json`.

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

Background task references are not queues. They identify host-owned work and
surface task lifecycle events from tool results. `status` is host-owned and
open; `lifecycle` is the runtime-owned `started`, `updated`, or `completed`
classification used to choose the emitted event. The host owns workers, durable
job state, retries, callbacks, later worker updates, and any eventual resume or
conversation insertion.

Message content can reference host-owned artifacts through
`ContentPart(type="artifact", data={"artifact": ...})`. Core preserves the
reference, media type, name, size, hash, and metadata but does not dereference,
retain, or garbage-collect artifact payloads.

The registry exposes neutral `ToolSpec` values. Provider adapters are
responsible for converting those specs to provider-specific tool formats.
Hosts may supply any object that implements `ToolRegistryProtocol`; production
registries should enforce the same validation semantics as `toolkit.ToolRegistry`.

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
protocol: `next_batch(calls)` chooses the next non-empty prefix batch and
`run_batch(batch, execute, stop_on_error=...)` yields `tool_started` and
`tool_completed` progress. Custom schedulers must yield `tool_started` before
calling the supplied `execute` function for that call, must call `execute`
exactly once for each completed batch call, and must not replace the returned
result. Implementations do not need to inherit from the default `ToolScheduler`
class. When an approval policy is configured, the Python reference runtime
conservatively resolves and executes at most one scheduled tool call at a time
so approval pauses cannot leave a partially approved batch visible.

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

Common approval risk annotations:

```python
ToolSpec(
    name="bash",
    description="Run a shell command.",
    input_schema={"type": "object"},
    annotations={
        "parallel_safe": False,
        "read_only": False,
        "idempotent": False,
        "risk": {
            "filesystem": "write",
            "network": "none",
            "subprocess": True,
            "destructive": True,
            "requires_approval": True,
        },
    },
)
```
