# Tool Protocol

The tool protocol has one invocation operation, one immutable prepared binding,
and a closed result union. The kernel owns provider-neutral values and orchestration;
the implementation toolkit owns language-native adaptation and JSON Schema validation.

## Tool Call and Specification

```text
ToolCall
├── id         stable id supplied by the model adapter
├── name       registered tool name
└── arguments  immutable JSON object
```

A `ToolSpec` contains name, description, input schema, optional output schema,
execution facts, and risk facts. Execution facts are:

```text
concurrency  serial | parallel
read_only    boolean
idempotent   boolean
```

`parallel` requires both `read_only` and `idempotent`. The catalog rejects an
invalid combination when it opens. Risk is an immutable JSON object with
standardized filesystem, network, subprocess, destructive, and
requires-approval fields. Scheduling facts do not imply approval risk.

## Invocation Catalog and Binding

`ToolCatalogProvider.open_catalog()` returns one immutable catalog for an
invocation. Registry changes are visible only to later invocations.

```text
binding = catalog.bind(call)
```

Binding performs one trust-boundary operation:

1. resolve the tool name in the frozen catalog;
2. validate arguments against the compiled input schema;
3. capture the exact immutable spec and implementation;
4. return a `ToolBinding` used for approval and invocation.

The binding prevents time-of-check/time-of-use changes. It also validates
structured output before returning it to kernel. Unknown names, invalid input,
and invalid output become model-visible failure results. Invalid values never
reach approval or durable success.

## Tool Implementation

Every tool has one async operation:

```text
invoke(call: ToolCall, context: ToolContext) -> async ToolResult
```

The default toolkit does not accept synchronous implementations. Blocking work
must be managed explicitly by the implementation or host. A tool must honor
task cancellation and settle every owned side effect before cancellation
escapes.

`ToolContext` exposes immutable run context, bounded progress emission, and a
cooperative active-call cancellation query. It exposes no mutable runtime state.

## Outcome and Result

`ToolOutcome` is the single model-visible union:

### `ToolSuccess`

A completed model-visible observation with content and optional structured
content.

### `ToolFailure`

A model-visible error. Input/output validation errors, approval denial,
implementation exceptions, and decorator exhaustion use this variant unless
they are runtime control flow. A tool failure does not terminate the run; the
next planning step may recover.

### `ToolAccepted`

An acknowledgement of host-owned background work with a stable correlation id.
The call is complete. Later updates arrive through conversation insertion and
never reopen the call.

### `ToolWaiting`

A model-visible waiting observation and optional task reference.

`ToolResult` then has exactly two branches:

```text
SettledResult(outcome: ToolSuccess | ToolFailure | ToolAccepted)
WaitingResult(outcome: ToolWaiting, suspension: Suspension)
```

The waiting observation is committed first, then state becomes `Suspended`
with the remaining semantic continuation. External execution and callbacks
remain host-owned.

A tool message stores the result's exact `outcome`. There is no second summary
DTO. Host-only suspension metadata is not sent to the model. Artifact references
also have one authoritative representation.

## Approval

Only valid `ToolBinding` values reach `ApprovalPolicy`. Approval receives one
ordered selected batch and returns an ordered allow, deny, or suspend decision
for each member.

- allow invokes the binding;
- deny produces `ToolFailure` without invocation;
- if any decision suspends, no member of that batch is invoked or committed;
- the first suspension in model order controls the durable state.

Approval cannot rewrite calls, specs, selections, or results. Approval awaits
share the invocation's monotonic deadline.

## Batch Selection

`BatchPolicy` is a pure synchronous strategy that selects a non-empty prefix of
pending calls. Kernel validates the selection and owns execution.

The default policy:

- selects one serial or unresolved call;
- otherwise selects consecutive parallel-safe calls;
- respects maximum batch size and active concurrency;
- holds each permit from `tool_started` through `tool_finished`;
- treats every serial tool as a checkpoint barrier.

There is no execution scheduler port. A strategy chooses legal work but cannot
run calls, choose physical completion order, or redefine checkpoint timing.

## Execution and Checkpoint

For one selected prefix:

```text
select
  -> bind and validate
  -> approve
  -> invoke or precompute failures
  -> validate results
  -> order by model call position
  -> create one typed Change
  -> reduce to one Checkpoint
  -> commit atomically
```

Live completion may follow physical order. Durable tool messages always follow
model order. An interrupted parallel batch commits none of its calls. Repeating
that batch is safe because every member was declared read-only and idempotent.

A waiting result is included in the same atomic batch checkpoint. The first
waiting result in model order determines suspension after all preceding batch
results have been ordered.

## Retry, Timeout, Progress, and Cancellation

Kernel owns the absolute invocation deadline, not per-tool retry or
circuit-breaker state. Toolkit decorators provide those policies around one logical
tool invocation. Decorator attempts do not increment committed tool counters.

`RetryingTool` retries only the configured exception classes and only for an
idempotent tool; model-visible settled failures are not retried. Concrete retry,
backoff, exhaustion, and circuit-recovery behavior belongs to the
[`jharness-toolkit` README](../packages/jharness-toolkit/README.md).

Progress is live-only and bounded by `RunLimits.max_buffered_progress`. A tool
that exceeds its bound receives a tool-level error from its context.

Cancellation is cooperative and scoped to an active call. Unknown, stale, and
completed ids are no-ops. Kernel never forcefully terminates host threads,
processes, workers, or external jobs.
