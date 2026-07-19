# ADR 0011: Linear History and Durable Commits

Status: Accepted
Date: 2026-07-19
Supersedes: [ADR 0002](0002-atomic-checkpoint.md)

## Context

`Checkpoint` must remain the complete portable recovery value, but repeatedly copying
an ever-growing tuple, rebuilding a tool-call-id set, serializing the complete
checkpoint, and writing that checkpoint on every commit makes state evolution and
persistence cumulatively quadratic in history length. A persistence port that accepts
only the new full checkpoint also hides the trusted history delta that the reducer
already knows.

## Decision

Keep `Checkpoint(id, snapshot, fact)` as the sole complete recovery truth and retain
one atomic repository boundary. Represent snapshot history as an immutable persistent
`RunHistory` whose reverse-chronological skew-binary forest gives `O(1)` persistent
append, `O(log H)` random access, `O(H)` traversal, and a newest suffix independent of
the old prefix. Retain an incremental history digest, a persistent call-id proof, and
an unresolved-call cursor so trusted append evolution examines only new messages.

Pass repositories a `DurableCommit` containing the complete resulting checkpoint, its
parent checkpoint id, its semantic digest, and exactly one closed history change:

- initial history;
- non-empty append;
- complete replacement;
- unchanged history.

The history change carries the predecessor count and digest when a predecessor exists.
It is validated against the resulting checkpoint before reaching the repository. A
repository atomically checks run-scoped checkpoint idempotency, revision, parent, and
history base before publishing the new head. The envelope is a commit proof, not a
second recovery model, journal, receipt, or portable wire representation.

Repository implementations store checkpoint core data separately from immutable
history chunks. Ordinary appends write only new chunks. Memory retains immutable
values directly. SQLite, MySQL, and Redis use a new v2 physical namespace and do not
read, migrate, or alias the obsolete layout. Redis keys are scoped and hash-tagged per
run so unrelated runs can occupy different cluster slots.

Every model call continues to receive the complete current durable history. LLM input,
request materialization, and provider encoding costs are deliberately outside the
state-evolution and persistence bounds in this decision; kernel does not truncate the
conversation to satisfy those bounds.

The required complexity bounds are:

- fixed-size append evolution and commit are independent of prior history length;
- `N` fixed-size appends perform `O(N)` history and persistence work;
- an exact retry is `O(1)` relative to history size;
- one complete-history model request is `O(H)`, and cumulative LLM input may be
  `O(N^2)` by explicit product choice outside these guarantees;
- explicit replacement is linear in the replacement input;
- recovery/export is linear in the complete history it returns.

## Consequences

- Public history is a persistent sequence rather than a tuple; obsolete constructors
  and repository signatures are replaced without forwarding aliases.
- Checkpoint remains portable and complete while backend writes no longer encode old
  messages on every revision.
- Checkpoint ids may repeat in different runs; uniqueness and idempotency are scoped by
  `(run_id, checkpoint_id)`.
- Model behavior retains the complete transcript by default; provider-specific context
  limits remain the responsibility of hosts and model adapters.
- A full recovery read and a deliberate history replacement still pay the size of the
  data they return or accept; those lower bounds are explicit rather than hidden in an
  ordinary append path.
