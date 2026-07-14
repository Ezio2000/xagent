# ADR 0002: Atomic Checkpoint

Status: Accepted
Date: 2026-07-13

## Context

Recovery state and the semantic fact that produced it form one durable truth.
Separate snapshot, journal, transition, commit, and receipt representations
duplicate identity and allow partial visibility.

## Decision

Use one `Checkpoint(id, snapshot, fact)` and one
`RunRepository.commit(checkpoint)` port. The repository atomically compares the
derived predecessor revision and stores both snapshot and fact.

Revision `0` requires an absent run. Revision `n` requires current revision
`n-1`. Checkpoint id is the idempotency key. An identical retry succeeds;
reusing the id for different content fails. Successful commit returns no mirror
receipt. A committed event is emitted only after success.

## Consequences

- Snapshot and audit fact cannot diverge.
- Run id, revision, expected revision, and commit id are not mirrored across
  multiple command and receipt objects.
- Repository failure leaves the prior checkpoint authoritative.
- Backend adapters own any multi-table transaction.
- Compare-and-swap remains a persistence guard, not an execution lease.
