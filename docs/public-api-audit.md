# Python Public API Naming Audit

Date: 2026-07-04

This audit covers the Python import surfaces exported from each package root.
The current decision is to keep `kernel.__all__` focused on core runtime
protocols and move optional helper APIs to sibling packages. Anything not
exported from a package root is internal unless a later audit promotes it.

## Boundary Decisions

- Keep runtime orchestration names: `AgentLoop`, `AgentResult`, `AgentState`,
  `AgentStatus`, `LoopLimits`, `RuntimeContext`, `RuntimeHook`,
  `LimitReasons`, `ModelErrorDecision`, `ToolSchedulerFactory`, and
  `RunSnapshot`.
- Keep resume/control names: `ResumeInput`, `PauseSelector`, `PauseRequest`,
  `RunController`, `ConversationInsert`, `ToolCancelRequest`, and
  `PauseState`.
- Keep core extension protocol names: `RunStore`, `StoredCheckpoint`,
  `CheckpointSummary`, `ApprovalPolicy`, `ApprovalRequest`,
  `ApprovalDecision`, `ApprovalAction`, `RunJournal`, and `JournalRecord`.
- Keep event names: `AgentEvent`, `EventType`, `EventTypes`, `EventEmitter`,
  and `QueuedEvent`.
- Keep message names: `Message`, `ContentPart`, `ArtifactRef`, and `ToolCall`.
- Keep model and tool protocol names: `ModelRequest`, `ModelResponse`,
  `ModelClient`, `StreamingModelClient`, `ModelOptions`, `ToolChoice`,
  `ResponseFormat`, `ModelCapabilities`, `ModelUsage`,
  `ToolSpec`, `ToolInvocation`, `ToolExecutionContext`, `ToolObservation`,
  `ToolAcceptance`, `ToolRejection`, `ToolOutput`, `BackgroundTask`,
  `ExecutableTool`, `AcceptableTool`, `InvocableTool`, `Tool`,
  `ToolRegistryProtocol`, and `normalized_tool_risk`.
- Keep model streaming names: `ModelStreamEvent`, `ModelContentDelta`,
  `ModelToolCallDelta`, `ModelReasoningDelta`, `ModelUsageDelta`,
  `ModelStreamStarted`, and `ModelStreamCompleted`.
- Keep scheduler detail names: `ToolCatalog`, `ToolScheduler`,
  `ToolSchedulerProtocol`, `ToolBatch`, `ToolStarted`, and `ToolCompleted`.
  They are useful for tests, advanced hosts, and future SDK alignment even
  though most users will interact through `AgentLoop`.
- Keep error names: `AgentError`, `ModelError`, `ModelProviderError`,
  `ModelErrorInfo`, `ToolError`, `InvalidToolCall`, `DuplicateToolError`, and
  `LimitExceeded`.

## Naming Rationale

`ResumeInput` is intentionally not `ResumeRequest`. It is a strict value object
for crossing a durable runtime boundary. It contains a snapshot, optional
append-only messages, an optional expected-pause selector, and host metadata.
The name keeps it aligned with `resume-input.schema.json`.

`RunController` remains separate from `LoopLimits`. The controller is the
host-owned imperative handle for pause, interrupt, and conversation insertion.
It also carries cooperative tool-cancel requests; cancellation remains a
host/tool contract, not a forced process kill. Limits are static run
configuration, including token budgets and bounded model retry counts.

`PauseSelector` names the `expected_pause` matcher used by `ResumeInput`. It is
not a general query object; it only matches the paused snapshot before resume.

`Message`, `ContentPart`, and `ToolCall` are the public message protocol names.
They intentionally avoid provider-specific terms such as chat, prompt, block,
or function call. `ArtifactRef` is the portable host-owned artifact reference
stored in content part data. `ToolCall` is model-requested work.
`ToolInvocation` is the tool-facing view of that work, including an open `mode`
string. `ToolObservation` is execute-mode output, `ToolAcceptance` is
accept-mode acknowledgement for external completion, `ToolRejection` is
accept-mode failure output, and `ToolOutput` is the generic extension output
shape. `BackgroundTask` is the optional host-owned background work reference
that tool outputs can surface in events and durable tool-message metadata.

Public trace names moved to `diagnostics`: `RunTrace`, `TraceStep`,
`TraceStepKinds`, `ReplayResult`, `ReplayError`, and `replay_trace`.
`AgentResult.trace` is a v0 trace payload mapping; callers that need object
helpers should pass it to `diagnostics.RunTrace.from_dict(...)` or
`diagnostics.replay_trace(...)`. The kernel may still record and emit trace
payloads during a run, but it must not define trace object classes,
trace-from-events helpers, replay, or trace validation APIs.

`RunStore`, `ApprovalPolicy`, and `RunJournal` are protocol names rather than
implementation names. The SDK owns the portable boundary semantics for durable
checkpoint persistence, tool approval decisions, and append-only event
journaling, while concrete stores, approval UIs, sandboxes, dashboards, and
policy engines remain host-owned. `StoredCheckpoint`, `CheckpointSummary`,
`ApprovalRequest`, `ApprovalDecision`, and `JournalRecord` are value objects at
those boundaries.

`EventEmitter` is the right name for hook-owned custom event emission. It emits
`QueuedEvent` payloads, while the runtime remains the owner of `AgentEvent`
envelopes, core event ordering, run ids, and sequence numbers. `QueuedEvent` is
public because `EventEmitter.drain()` returns queued custom events without a
runtime envelope.

`RuntimeHook` is the public structural hook marker for observing and rewriting
runtime boundaries. Hook objects do not need to inherit behavior from a base
class; the kernel discovers implemented hook methods structurally. The name is
intentionally broader than event hook because the same hook object can cover
model, tool, transition, and event boundaries.
`ModelErrorDecision` is the typed return value for `on_model_error`, keeping
retry and user-facing message policy host-owned instead of deriving it directly
from provider metadata.

`ModelClient` and `StreamingModelClient` name the adapter protocols. The
streaming protocol is separate because streaming is optional and remains live
progress until a complete `ModelResponse` is available.

`ModelStreamEvent` and the delta/completed/started names describe provider-
neutral streamed model progress. `ModelStreamAccumulator` moved to `modelkit`
because it is useful for adapter tests and for adapters that need to assemble
streamed deltas into a final `ModelResponse`, but it is not required to run the
kernel loop.

`ModelCapabilities`, `ModelOptions`, `ToolChoice`, `ResponseFormat`, and
`ModelUsage` keep provider-neutral model configuration, capability discovery,
and accounting separate from the model client implementation. The helper
`model_capabilities` moved to `modelkit`; it stays lower-case there because it
is a helper function, not a value type.

`ToolRegistry` moved to `toolkit`. `kernel` keeps `ToolRegistryProtocol` and
tool protocol/value types so hosts can inject a custom registry without making
the kernel depend on JSON Schema or a concrete registry implementation.

Prompt construction helpers such as `user_text`, `system_text`,
`assistant_text`, `tool_text`, and `external_text` live in `prompting`. Kernel
message types remain provider-neutral data structures.

Reusable scripted models and event collectors live in `harness`, not `kernel`
or `conformance`.

`ModelProviderError` is the adapter-facing wrapper for structured provider
failures. `ModelErrorInfo` is the serializable payload. This keeps provider
transport details out of `AgentState` while preserving useful diagnostics.
`ToolSchedulerFactory` is public so advanced hosts can replace scheduling
policy. It receives a read-only `ToolCatalog`, not executable tool
implementations. Custom schedulers implement `ToolSchedulerProtocol`;
`ToolScheduler` remains the default implementation.

## Public Versus Internal

The following implementation names are intentionally not part of the public API:

- `RunControlState`
- `RuntimeTimeoutError`
- `RuntimePauseInterrupt`
- `RuntimeConversationInsert`
- `TraceRecorder`
- `KernelModelStreamAccumulator`
- `runtime_model_capabilities`
- private validation and compaction helpers

These names may change without a compatibility path. If a future SDK needs one
of them as portable surface area, promote it deliberately with spec and
conformance coverage.

## Future Rename Rules

Before v1, breaking public renames are allowed when they improve the contract.
Do them directly and update docs, examples, conformance cases, and tests in the
same change. Do not add deprecated aliases or dual names unless the project
explicitly starts supporting historical compatibility.
