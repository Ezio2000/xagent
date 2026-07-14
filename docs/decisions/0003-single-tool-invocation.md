# ADR 0003: Single Tool Invocation

Status: Accepted
Date: 2026-07-13

## Context

Multiple call modes and implementation methods couple scheduling to tool
business behavior. Correctness would require mode/result validation throughout
the engine.

## Decision

Every tool implements one async `invoke` method. Calls contain id, name, and
arguments. Model-visible outcomes use the closed union `ToolSuccess |
ToolFailure | ToolAccepted | ToolWaiting`. `ToolResult` is either a settled
outcome or a waiting outcome paired with host-only suspension data.

An immutable invocation catalog binds and validates a call before approval.
The bound invocation captures the implementation and validates structured
output before returning it to the kernel.

## Consequences

- Acceptance and waiting are explicit result semantics.
- Retry and circuit breaker behavior move to toolkit decorators.
- The prepared-binding concept remains, narrowed to TOCTOU safety.
- Tool messages reuse the result's model-visible outcome rather than a second
  summary DTO.
