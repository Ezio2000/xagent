# Model Protocol

Kernel communicates with providers through one model-neutral async operation.
Concrete adapters live in an implementation's provider layer or host packages.

## Model

```text
Model.invoke(
    request: ModelRequest,
    context: RunContext,
    stream: boolean,
    emit_delta: optional async DeltaSink,
) -> async ModelResponse
```

`Model.invoke` always returns one complete `ModelResponse`. `stream=True`
requests incremental observation through the async `emit_delta` sink;
`stream=False` uses no sink. The operation's semantic result is identical in
both modes.

The provider adapter owns wire-chunk accumulation and response-body lifetime.
Kernel does not reconstruct a second final response from deltas. Potentially
blocking work is async; kernel creates no thread to run synchronous adapters.

The operation must honor task cancellation promptly and close or settle every
provider-owned response before cancellation escapes.

## Request

`ModelRequest` contains:

- immutable conversation history;
- immutable tool specifications from the invocation catalog;
- model options;
- tool choice;
- optional response format.

Tool calls contain id, name, and arguments only. Provider-specific behavior
belongs in the adapter profile or provider-local metadata; it does not add
portable call modes.

Kernel exposes no request mutation hook. Request shaping, routing, fallback,
caching, and telemetry are ordinary `Model` decorators.

## Response

`ModelResponse` contains:

- final content parts;
- zero or more ordered tool calls;
- finish reason;
- usage;
- provider, model, and response identifiers when available;
- adapter-local metadata that is not copied into history unless a portable
  field owns it.

A response with calls commits one assistant message and `ToolsPending(calls)`.
A response without calls commits one assistant message and `Completed(content)`.
Planning metrics and usage advance only with that checkpoint.

The adapter validates provider completion and returns a structurally complete
response. Kernel validates the untrusted port return once before reduction.

## Streaming

The delta union contains only incremental values:

- content delta;
- tool-call delta;
- reasoning delta;
- usage delta.

Start and completion are represented by lifecycle events around `Model.invoke`,
not model delta variants. A delta is live-only and may be dropped by a bounded
observation queue. The returned `ModelResponse` is the sole complete result.

The sink is async and ordered. An adapter awaits each emission, does not launch
detached emitter tasks, and stops emission before returning or raising. If
pause, insertion, provider failure, cancellation, or deadline interrupts the
call, no partial assistant message is durable.

## Errors and Retry

Adapters raise one structured provider-neutral `ModelError` containing stable
code, message, retryability, provider status, and request id when known. An
unhandled model failure becomes `Failed` only if the terminal checkpoint can be
persisted; otherwise the previous checkpoint remains authoritative.

Kernel performs one logical `Model.invoke` per planning step. Retry, provider
fallback, routing, and circuit breaking are decorators. A streaming decorator
must not publish deltas from a failed attempt and then silently retry as if they
belonged to one stream.

## Capabilities

A model exposes immutable capabilities such as streaming, tools,
multimodality, structured output, and usage reporting. Runtime reads them once
per invocation and rejects an unsupported request before the provider call.

Provider packages translate external APIs into this protocol. They never own
runtime state, tool execution, approval, history reduction, control, or commit
semantics.
