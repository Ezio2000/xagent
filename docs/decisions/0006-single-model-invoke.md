# ADR 0006: Single Model Invocation

Status: Accepted
Date: 2026-07-13

## Context

Separate complete and streaming protocols duplicate client APIs and encourage
both provider and kernel to accumulate the same stream.

## Decision

Define one async `Model.invoke(request, context, *, stream, emit_delta)` method
that always returns a complete `ModelResponse`. In streaming mode the provider
adapter owns chunk accumulation and awaits an optional ordered delta sink.

Delta variants contain only incremental content, reasoning, tool-call, and
usage values. Runtime events represent model start and finish.

## Consequences

- All adapters implement one semantic operation.
- Kernel never reconstructs a second final response.
- Complete and streaming calls share error, cancellation, deadline, and metric
  semantics.
- Retry and fallback remain model decorators outside kernel orchestration.
