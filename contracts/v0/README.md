# JHarness v0 Contract Map

`contracts/v0` is the cross-language source of truth for portable wire shapes
and runtime semantics. Each portable value has one authoritative representation.

Schema identifiers use the non-resolving publication namespace:

```text
https://jharness.invalid/spec/v0/<file>.schema.json
```

Implementations resolve every id from their bundled contract directory. They
must not fetch schemas over the network.

## Normative Documents

- `state-machine.md`: flat lifecycle, transitions, metrics, and checkpoint
  boundaries;
- `run-control.md`: Runtime/Invocation start, continue, resume, control,
  deadline, and precedence;
- `repository.md`: atomic checkpoint persistence and cancellation safety;
- `tool-scheduling.md`: binding, approval, prefix batching, execution, and
  atomic tool commit;
- `model-stream.md`: one model operation and four live delta variants;
- `run-trace.md`: compact trace construction and deterministic verification.

## Schema Index

| File | Owns |
| --- | --- |
| `messages.schema.json` | Content parts, one canonical artifact payload, tool calls, one model-visible tool outcome, and messages. |
| `model-request.schema.json` | Provider-neutral request, model options, tool choice, and response format. |
| `model-response.schema.json` | Complete provider-neutral response and usage. |
| `model-error.schema.json` | Structured provider-neutral model failure. |
| `tools.schema.json` | Tool specs, execution facts, and approval risk. |
| `tool-result.schema.json` | Tool outcome plus host-only waiting suspension. |
| `approval.schema.json` | Ordered approval requests and decisions. |
| `limits.schema.json` | Portable run budgets and bounded concurrency. |
| `state.schema.json` | Flat lifecycle, active resume targets, suspension, and metrics. |
| `run-context.schema.json` | Stable run context and host correlation. |
| `run-snapshot.schema.json` | Revisioned durable run aggregate nested in a checkpoint. |
| `checkpoint.schema.json` | Atomic checkpoint, semantic fact, and compact RunView. |
| `run-request.schema.json` | Start request and checkpoint-based continue/resume requests. |
| `events.schema.json` | Invocation-local observation and checkpoint commit events. |
| `run-trace.schema.json` | One header plus compact event entries for verification. |

Versioned aggregate wires (`Checkpoint`, invocation event, and trace) carry
`schema_version` only on their top-level envelope. Nested domain values do not
repeat it. Snapshot has no version field because it is recovered only as part
of a versioned checkpoint.

## Offline Reference Graph

```text
messages
├── model-request ──> tools
├── model-response
└── tool-result ──> state

state ──> messages, model-response
run-snapshot ──> run-context, messages, state
checkpoint ──> run-snapshot, state, model-response
run-request ──> checkpoint, messages, run-context, state
approval ──> messages, tools, state
events ──> approval, checkpoint, messages, model-response
run-trace ──> events, checkpoint
```

Every relative `$ref` resolves within this directory. The root `$id` of each
schema is unique and matches its file name.

## Boundary Rules

- `Checkpoint` is the complete portable recovery value; a detached snapshot is
  not accepted by continue or resume.
- `Invocation.result()` returns the last authoritative `Checkpoint`.
- Lifecycle is exactly `Planning`, `ToolsPending`, `Suspended`, `Completed`,
  `Failed`, or `Limited`.
- `Suspended.resume_to` is exactly `Planning` or `ToolsPending`.
- One durable boundary increments snapshot revision once and writes one fact in
  the same checkpoint.
- `Repository.commit(checkpoint)` returns no receipt; success means the
  checkpoint is authoritative.
- Only `checkpoint_committed` advances durable trace state. Other events are
  live observation.
- A parallel tool batch commits all ordered outcomes or none.
- Resume restores the active state saved in `Suspended.resume_to`.
- Terminal checkpoints cannot continue or resume.
- Tool failure is model-visible; model/protocol/infrastructure failure is a
  terminal run state when it can be committed.
- Portable tools expose one invoke operation. Portable models expose one invoke
  operation and optionally emit four delta variants.
- Artifact data and tool outcomes each have one authoritative representation.

## Schema and Semantic Validation

JSON Schema owns structural validity. SDK boundary codecs and conformance
runners additionally enforce semantic rules such as:

- unique tool-call ids in one assistant message;
- tool messages linked to the immediately preceding unresolved calls;
- start history being non-empty and valid in `Planning` before invocation
  creation;
- `parallel` execution requiring read-only and idempotent facts;
- non-empty, uniquely identified calls in `ToolsPending`;
- checkpoint revision `0` for start and consecutive later revisions;
- equal ordered `call_ids` and `outcome_kinds` lengths in tool-batch facts;
- model-turn result, part count, calls, usage, and limit reason determining its
  complete after view;
- a tool-batch fact carrying a compact suspension exactly when its after state
  is `Suspended`;
- continue/resume checkpoint state compatibility;
- appended resume messages only for `resume_to=Planning`;
- monotonic metrics and revisions;
- trace entry sequence, fact transition, and final checkpoint-id consistency.

Portable behavior changes update schemas, normative documents, and conformance
cases together.
