# ADR 0008: Trace Verification

Status: Accepted
Date: 2026-07-13

## Context

A diagnostic artifact containing only observations and compact durable data can
verify ordering and transition consistency but cannot re-execute model and tool
effects. Re-implementing lifecycle rules in diagnostics also permits runtime
and verification to drift.

## Decision

Expose opt-in `build_trace` and `verify_trace` in
`jharness.kernel.diagnostics`. A trace stores one header and ordered event entries. A
`checkpoint_committed`
entry carries its fact and compact after view. Derived step ids, before views,
and final summaries are not stored.

Runtime and diagnostics share a pure
`verify_change(before_view, fact, after_view)` rule. Verification performs no
external I/O and makes no claim to re-execute effects.

## Consequences

- API naming states the actual guarantee.
- Durable transition rules have one implementation.
- Trace size grows linearly with entries.
- Diagnostics remains optional and read-only.
