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

## Implementation Boundaries

Every language implementation exposes three conceptual layers:

| Layer | Owns |
| --- | --- |
| Kernel | State, runtime/invocation, model and tool ports, control, limits, events, codecs, policies, atomic repository, and optional diagnostics. |
| Toolkit | Concrete tool registration, JSON Schema validation, language-native adapters, retry, and circuit-breaking decorators. |
| Providers | Provider clients, profiles, transport lifecycle, error normalization, and provider-local codecs. |

Packaging is language-specific and is documented by each implementation. The
kernel layer may not depend on an implementation's toolkit or providers.
Applications assemble implementations from the outside.

Conformance runners own fixture interpretation, schema validation, deterministic
doubles, and portable behavior execution. A runner is development tooling, not a
runtime product.

## Runtime and Invocation

`Runtime` is an immutable configuration and port assembly. It exposes distinct
methods because their legal inputs differ:

- `start(...)` creates and persists a new planning snapshot;
- `continue_from(checkpoint)` recovers a nonterminal durable checkpoint whose
  prior owner has relinquished execution;
- `resume(checkpoint, ...)` acknowledges a suspension and persists its exact
  saved continuation before work resumes.

Each method returns one single-use `Invocation`. The invocation owns one engine
execution, one invocation id, control delivery, and optional observation. Its
result is the last successfully committed `Checkpoint`; state is available as
`checkpoint.snapshot`. There is no result wrapper with a second status view.

`Invocation.events()` and `Invocation.result()` never create separate runs. If
event streaming is selected before execution starts, both APIs observe the same
execution. If result-only mode starts first, the engine uses a null observation
sink and a later event subscription is rejected. Awaiting the result more than
once returns the same terminal value or error. An invocation cannot be restarted.

Closing the event iterator requests cancellation of that invocation. It does
not create a durable cancelled state. Pause, conversation insertion, and active
tool cancellation are invocation-scoped control operations, not persisted
command queues. They may be submitted before execution starts, but termination
closes and drains the live channel; later submissions are ignored.

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

Domain values do not serialize themselves. Each implementation owns explicit codecs
for portable top-level documents and private codecs for their nested values.
Decoding validates unknown keys, discriminators, finite JSON numbers, and
cross-field invariants once at the trust boundary. Internal immutable values
are then reused directly.

Provider request/response codecs remain in the provider layer; tool JSON Schema
validation remains in the toolkit layer. There is no reflection-driven serializer
or schema framework inside the kernel.

## Observation and Diagnostics

Live events describe in-flight work and may be lossy. A committed event names
the successfully stored checkpoint fact and a compact run view; it is never a
persistence substitute. Event consumers are read-only.

An implementation's optional diagnostics component constructs a trace from invocation
events. A trace stores each entry once; `checkpoint_committed` entries carry the fact and
compact after view. `verify_trace` checks ordering, event rules, and each durable
change through the same pure verification rules used by runtime code. It does
not claim to re-execute models or tools and does not implement a second
lifecycle machine.

## Concurrency and Deadlines

- One invocation opens one immutable tool catalog.
- At most one task exists per selected batch member.
- Active tools never exceed `RunLimits.max_tool_concurrency`.
- Lossy deltas and progress use bounded queues; lifecycle and checkpoint events
  remain lossless.
- One monotonic work deadline covers model, catalog, approval, tool, history,
  and ordinary repository awaits.
- A separate fixed cleanup grace may only settle owned tasks and persist a
  deadline terminal checkpoint.
- Every async port must honor task cancellation and settle owned work before
  cancellation escapes.
- Kernel creates no hidden thread pool and leaves no detached work.

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
