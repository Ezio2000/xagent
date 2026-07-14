# ADR 0005: Runtime and Single-Use Invocation

Status: Accepted
Date: 2026-07-13

## Context

Configuration, execution, control, event collection, result wrapping, and
workflow convenience can represent one run several times and accidentally
start independent executions for result and event APIs.

## Decision

Use one immutable `Runtime` as port/configuration assembly. Its distinct
`start`, `continue_from`, and `resume` methods return one single-use
`Invocation`.

The invocation owns exactly one execution. Its `result`, `events`, and control
operations refer to that execution. Event-first mode enables one event consumer;
result-first mode uses a null sink and rejects later event subscription. The
result is the last committed `Checkpoint`, without a wrapper; its snapshot owns
state.

## Consequences

- Illegal start/continue/resume input combinations are separated by method
  signatures.
- Result and event observation cannot double-execute external effects.
- Result-only use allocates no queue or event values.
- Workflow assembly requires no additional façade package.
- An invocation cannot be restarted or observed by multiple competing event
  consumers.
