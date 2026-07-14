# Standard Conformance Tools

The normative catalog is `tools.contract.json`. Every runner implements those
tools with one async invocation operation.

## Shared Rules

- Arguments are validated by each canonical `ToolSpec` before approval.
- Optional values use manifest defaults without coercion.
- Implementation exceptions normalize to `ToolFailure`.
- `delayed_echo` and `parallel_wait` are explicitly parallel, read-only, and
  idempotent.
- Each case receives a fresh catalog and fresh stateful tool instances.
- Structured output is validated before kernel observation.

## Behaviors

| Tool | Behavior |
| --- | --- |
| `echo` | Returns `ToolSuccess` with the supplied text. |
| `delayed_echo` | Sleeps for the requested delay and returns success; used for parallel completion-order tests. |
| `fail` | Raises the stable implementation error `tool failed`, normalized to failure. |
| `accept` | Returns `ToolAccepted`; correlation id defaults to the call id. |
| `wait` | Returns serial `ToolWaiting` with the requested suspension. |
| `parallel_wait` | Delays, then returns a read-only waiting result for atomic batch tests. |
| `progress` | Emits ordered `{ "step": value }` records and cooperatively observes cancellation. |
| `strict_count` | Requires an integer count and returns its decimal text; used for input validation. |
| `invalid_output` | Returns structured data that deliberately violates its declared output schema. |

`covered_by` entries name executable conformance cases. The runner verifies
that every listed case exists and that every standard tool is exercised.
