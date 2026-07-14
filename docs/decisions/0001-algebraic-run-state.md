# ADR 0001: Flat Algebraic Run State

Status: Accepted
Date: 2026-07-13

## Context

Lifecycle correctness should be visible in the type shape. Wrapper state,
continuation wrappers, status fields, and optional terminal fields create more
representations than there are semantic states.

## Decision

Use the flat union `Planning | ToolsPending | Suspended | Completed | Failed |
Limited`.

`ToolsPending` contains a non-empty ordered call tuple. `Suspended` contains the
exact `Planning | ToolsPending` state to resume plus suspension data. Terminal
variants contain their required outcome data. Status is derived.

## Consequences

- Invalid state/continuation combinations are unconstructible.
- Resume restores one saved semantic state.
- Transition dispatch has one branch per actual lifecycle state.
- JSON Schema uses one discriminator and one branch per variant.
- State change requires matching schema and conformance coverage.
