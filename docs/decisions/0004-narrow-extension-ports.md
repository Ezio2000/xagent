# ADR 0004: Narrow Extension Ports

Status: Accepted
Date: 2026-07-13

## Context

General runtime hooks can mutate requests, calls, outputs, events, and
transitions. A programmable scheduler can control execution timing and result
delivery. Kernel code must defensively validate every extension action and run
synchronous host code behind worker isolation.

## Decision

Extend behavior through narrow async ports, pure selection policies, and
protocol decorators. Runtime event consumers are read-only. `BatchPolicy`
selects a pending prefix but kernel code alone executes the batch and commits
its results.

Potentially blocking ports are async. Pure value access, call binding, and
batch selection are synchronous and may not perform I/O.

## Consequences

- No extension can rewrite a state transition or core event.
- Model and tool request/response customization uses decorators.
- Open/closed extension is preserved at stable policy boundaries without
  exposing state-machine invariants.
