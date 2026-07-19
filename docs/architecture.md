# Architecture

This document defines the runtime architecture. Portable wire shapes are
normative in `contracts/v0`; portable behavior is normative in
`conformance/cases`; accepted design constraints are recorded under
[`decisions/`](decisions/README.md).

## System Shape

The runtime is an immutable state-transition kernel around injected model,
tool, approval, history, persistence, and batch-selection ports:

```text
Runtime.start / continue_from / resume
                  |
                  v
             Invocation
          / result | events \
         /         |          \
   control     one engine    read-only observation
                    |
          +---------+---------+
          |                   |
          v                   v
     Model.invoke       bound Tool.invoke
          |                   |
          +---------+---------+
                    v
              typed Change
                    |
                    v
       reduce(snapshot, change)
                    |
                    v
       Checkpoint(snapshot, fact)
                    |
                    v
        Repository.commit atomically
```

`RunSnapshot` is the immutable state aggregate. `Checkpoint` is the complete
recovery value that pairs a snapshot with its semantic fact and id. Effects
produce values; only the pure reducer constructs the next checkpoint. No
extension port can mutate runtime state or redefine a durable boundary.

## Python Package Boundaries

Five coordinated distributions expose five public namespace packages:

| Package | Owns |
| --- | --- |
| `jharness.kernel` | State, runtime/invocation, model and tool ports, control, limits, events, portable codecs, policies, atomic repository, and diagnostics. |
| `jharness.toolkit` | Concrete tool registration, JSON Schema validation, Python function adapters, retry, and circuit-breaking decorators. |
| `jharness.models` | Model clients, profiles, transport lifecycle, error normalization, and endpoint-local codecs. |
| `jharness.repository` | Memory, SQLite, MySQL, and Redis implementations of the kernel repository port. |
| `jharness.tools` | Ready-to-use filesystem, shell, structured interaction, and child-agent tools. |

Kernel is the dependency foundation. Toolkit, models, repository, and tools may import
its public API but may not import one another. Applications compose these packages from the
outside. Detailed ownership and build gates are documented in
[`python-package-boundaries.md`](python-package-boundaries.md).

The repository-local conformance runner owns fixture interpretation, schema
validation, deterministic doubles, and portable behavior execution. It is development
tooling and is excluded from all published distributions.

## Runtime and Invocation

`Runtime` is immutable configuration and port assembly. `start`, `continue_from`, and
`resume` have distinct legal inputs and each creates one single-use `Invocation`.
The invocation owns one execution; its authoritative result is the last committed
`Checkpoint`.

Events, result waiting, and live controls observe that same execution rather than
creating hidden runs or durable command queues. See [`event-stream.md`](event-stream.md)
for observation semantics and [`contracts/v0/run-control.md`](../contracts/v0/run-control.md)
for normative request and control behavior.

## Flat Lifecycle

Durable state is one of:

```text
Planning
ToolsPending(calls)
Suspended(resume_to: Planning | ToolsPending, suspension)
Completed(content)
Failed(error)
Limited(reason)
```

State contains exactly the data valid for its variant. Status strings are
derived presentation values, not mutable fields. `Completed`, `Failed`, and
`Limited` are terminal. Detailed transition rules live in
[`state-machine.md`](state-machine.md).

## Planning

At `Planning`, the engine optionally commits a valid history reduction, builds
one provider-neutral `ModelRequest`, and calls `Model.invoke`. Complete and
streaming execution use the same model operation. An optional async delta sink
receives bounded live content, reasoning, tool-call, and usage deltas.

The returned `ModelResponse` is complete. It produces one assistant message and
either `ToolsPending` or `Completed`. Partial deltas never enter durable history.
Model usage and planning counters advance only with the successful checkpoint.

## Tools

At `ToolsPending`, a pure policy selects a non-empty pending prefix. The engine:

1. binds and validates each call against one immutable invocation catalog;
2. resolves approval for the ordered bound batch;
3. invokes allowed bindings under the concurrency and deadline limits;
4. validates results and orders them by model call position;
5. returns one typed change for the entire selected batch.

Only explicitly read-only and idempotent calls may run in parallel. A parallel
batch is all-or-nothing and commits in model order. A serial tool is a commit
barrier. Prepared binding captures both the validated spec and implementation,
preventing catalog changes between approval and execution.

## Change, Checkpoint, and Repository

Effects never edit a snapshot. They return a closed typed `Change`. The pure
reducer applies one change to one snapshot and produces:

```text
Checkpoint
├── id        stable idempotency key
├── snapshot  complete next recovery state
└── fact      compact semantic description of this boundary
```

The fact stores only boundary-specific data. Run id and revision come from the
snapshot; expected revision is derived as `revision - 1`. Revision `0` requires
that the run does not exist. Later revisions require the repository's current
revision to be exactly the predecessor.

`RunRepository.commit(checkpoint)` atomically persists the snapshot and fact.
Retrying an identical checkpoint id is an idempotent success; reusing an id for
different content is an error. Success returns no mirror receipt. Repository
failure leaves the previous committed checkpoint authoritative, and committed
observation is emitted only after success.

Compare-and-swap is not a distributed execution lease. Hosts serialize
invocation ownership per run id and use queue leases or fencing in deployment
infrastructure.

## Wire Boundary

Domain values do not serialize themselves. `jharness.kernel.wire` owns explicit codecs
for portable top-level documents and private codecs for their nested values.
Decoding validates unknown keys, discriminators, finite JSON numbers, and
cross-field invariants once at the trust boundary. Internal immutable values
are then reused directly.

Model request/response codecs remain in the models layer; tool JSON Schema
validation remains in the toolkit layer. There is no reflection-driven serializer
or schema framework inside the kernel.

## Observation and Diagnostics

Live events describe in-flight work and may be lossy. A committed event names
the successfully stored checkpoint fact and a compact run view; it is never a
persistence substitute. Event consumers are read-only.

Diagnostics compacts those events and verifies durable transitions without
re-executing models or tools. See [`event-stream.md`](event-stream.md) and
[`diagnostics.md`](diagnostics.md).

## Concurrency and Deadlines

One monotonic deadline bounds ordinary work, active tool concurrency is capped, and
lossy observation queues are bounded. Async ports must honor cancellation and settle
owned work; a non-compliant task may be reported as abandoned after the fixed cleanup
grace. Cancelling an individual result waiter does not itself cancel the run.

Normative deadline behavior lives in
[`contracts/v0/run-control.md`](../contracts/v0/run-control.md); operational bounds and
cleanup limits live in [`performance.md`](performance.md).

## Extension and Host Ownership

The runtime uses narrow ports and normal composition:

- strategy objects select batches and propose history rewrites;
- decorators provide retry, fallback, telemetry, caching, and request/result
  shaping around model and tool protocols;
- repository adapters own backend transactions;
- read-only consumers translate events to application transports.

Hosts own authentication, authorization, queues, leases, approval UI, artifact
payload storage, provider credentials, external task state, callbacks, and
application events. Kernel owns portable run semantics only.

General mutation hooks, service locators, dynamic implementation discovery,
programmable execution schedulers, and split snapshot/journal ports are outside
this architecture.
