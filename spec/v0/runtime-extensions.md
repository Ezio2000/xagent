# Agent Runtime v0 Core Extension Protocols

This document defines model-neutral extension protocols that affect runtime
durability, tool authorization, or auditability. SDKs may expose native protocol
classes, but concrete databases, approval UIs, sandboxes, dashboards, and
enterprise policy engines remain host-owned.

The portable JSON shapes for `ApprovalRequest`, `ApprovalDecision`,
`CheckpointSummary`, `StoredCheckpoint`, and `JournalRecord` are defined in
`runtime-extensions.schema.json`.

## Run Store

A run store persists durable checkpoints. `checkpoint` events still carry the
portable `RunSnapshot` payload; a store record wraps that snapshot with host
addressing metadata:

- `run_id`
- `checkpoint_id`
- `parent_checkpoint_id`
- `sequence`
- `status`
- `snapshot`
- `created_at`
- `metadata`

SDKs that support a run store must save the checkpoint before treating that
checkpoint as the latest durable boundary. Store save success must happen before
the corresponding `checkpoint` event is dispatched to hooks, emitted to callers,
or appended to a `RunJournal`. If saving fails, the checkpoint must not be
dispatched, emitted, journaled, or become the runtime's last durable checkpoint.
Core-generated checkpoint ids are deterministic: `checkpoint-{sequence}`, where
`sequence` is the corresponding checkpoint event sequence. A checkpoint saved
after resuming from a snapshot must use the resumed snapshot's
`checkpoint-{snapshot.context.sequence}` id as `parent_checkpoint_id`.
`load_checkpoint(run_id, null)` must return the snapshot from the checkpoint
with the greatest stored checkpoint `sequence` for that run. `list_checkpoints(run_id)` must
return summaries for checkpoints stored for the run; ordering is host-defined
unless a conformance case states otherwise.
Terminal snapshots with
`completed`, `failed`, or `limit_exceeded` remain invocation-terminal and must
not become valid resume inputs merely because they were stored.

Concrete storage engines, retention policy, leases, distributed locking,
tenant authorization, and checkpoint search are outside core.

## Approval Policy

An approval policy is a host-owned decision point before tool implementation
execution. Runtime-owned tool-name and input-schema validation happens before
approval; invalid calls are committed as non-invoked tool errors and are not
sent to the approval policy. For valid calls, the runtime passes a normalized
`ToolCall`, optional `ToolSpec`, runtime context, risk metadata derived from
tool annotations, and host metadata. The policy returns one of:

- `allow`: execute the tool normally.
- `deny`: do not call the tool implementation; commit a mode-appropriate tool
  error or rejection so the model can observe the denial and recover.
- `pause`: pause before tool execution with `resume_status: executing_tools` and
  leave the pending tool call unresolved. Approval pauses use the normal
  `pause_requested` event with `origin: control`, `request.source: approval`, and
  `request.wait_id` set to the tool-call id.

Runtimes that implement approval must emit `approval_requested` and
`approval_completed` events. Approval events are audit steps; they do not change
agent status by themselves. A later pause still uses the normal
`pause_requested`, `state_changed`, `checkpoint`, `run_paused`, `run_completed`
sequence.

Approval is not a sandbox. Core approval semantics do not enforce OS-level file
access, network access, subprocess isolation, tenant policy, or UI workflow.

## Run Journal

A run journal is an optional append-only record of emitted runtime events. It is
for audit, UI history, diagnostics, and indexing. It does not replace
`RunTrace`.

Journal records reference:

- the emitted `AgentEvent`;
- optional `checkpoint_id` for checkpoint events;
- optional host-filled `trace_step_id`;
- optional `payload_ref` and `payload_hash` for host-managed large payloads;
- host metadata.

`RunTrace` remains the compact deterministic replay surface. Journal playback
must not be required to reconstruct durable agent state; resuming and
time-travel-style forks should start from a `RunSnapshot`.

SDKs that support a run journal must append each event before delivering that
event to the caller. If journal append fails, the event that failed to append
must not be delivered to the caller or treated as journaled. For checkpoint
events, this append happens after the checkpoint has been saved to `RunStore`;
therefore a journal failure may leave a durable checkpoint that was not
delivered or journaled in the current event stream.

Concrete log stores, blob stores, dashboards, OpenTelemetry exporters,
redaction engines, and search indexes are outside core.
