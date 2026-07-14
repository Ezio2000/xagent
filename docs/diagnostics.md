# Diagnostics

Diagnostics provides opt-in trace construction and verification. It is an optional
kernel component and does not run unless a caller requests it.

## Trace Shape

A trace contains one invocation header and one ordered event-entry sequence:

```text
RunTrace
├── header
│   ├── run_id
│   ├── invocation_id
│   ├── request_kind
│   └── metadata_keys
└── entries
    └── event fields without repeated header identity or wire version
```

`checkpoint_committed` entries contain checkpoint id, semantic fact, and the
compact after `RunView`. Each fact and event is stored once.

Sequence is the entry identity. The before view for a committed fact is the
previous committed after view, beginning with the starting view carried by
`invocation_started`. The final view is the last committed after view. Therefore
the trace stores no repeated step id, before view, history-role list, or final
summary.

## Construction

`build_trace` compacts the ordered events captured for one invocation. It
validates matching run and invocation identity, moves common identity to the
header, and constructs an immutable trace. Result-only execution does not
implicitly construct a trace.

Applications that need diagnostics explicitly enable event observation. This
is a read-only sink; it cannot change engine input, output, event sequence, or
persistence.

## Verification

```text
verification = verify_trace(trace)
```

Verification checks:

- event sequence and lifecycle ordering;
- run/invocation identity consistency;
- live versus durable event rules;
- checkpoint ids are unique within the trace;
- monotonically consecutive checkpoint revisions;
- each `(before_view, fact, after_view)` through kernel's pure `verify_change`
  rule;
- terminal and suspension consistency;
- metric, usage, pending-call, and history deltas;
- model-order tool batch facts and atomicity;
- deadline and cancellation cleanup constraints.

A resume trace normally begins its durable entries with `resumed`. The sole
exception is an inherited deadline that was already expired at invocation
start; that trace begins directly with a `control/limited/deadline` checkpoint.

Runtime reduction and diagnostics call the same pure fact-verification rules.
Diagnostics does not maintain another lifecycle table.

`verify_trace` returns a structured verification result or raises a stable trace
error. It never calls a model, tool, approval policy, history reducer, or
repository and performs no I/O.

## Scope

A trace proves internal ordering and durable transition consistency. It cannot
re-execute external effects or prove that a provider returned the same bytes.
The API promises only the evidence it can verify.

Portable trace schema and invalid-trace conformance cases cover every durable
fact kind and event family. Diagnostic tests also require trace growth to remain
linear in the number of entries.
