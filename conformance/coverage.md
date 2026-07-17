# Portable Behavior Coverage

This matrix maps every normative v0 behavior family to at least one portable fixture.
It is completion evidence that contracts, Python runtime behavior, and observable
results remain in lockstep. Focused unit tests may add implementation detail, but they
do not replace the cases listed here.

## State Machine and Requests

| Normative behavior | Cases |
| --- | --- |
| Start commits revision 0; a complete turn finishes atomically | `final_only` |
| Invalid start history is rejected before invocation creation | `start_history_requires_planning` |
| Planning produces ordered pending calls; tool results return to planning | `one_tool_then_final` |
| Tool prefixes may retain a tools-pending continuation | `tool_batch_size_limit`, `waiting_result_with_pending` |
| Planning and tool continuations suspend and resume exactly; an already-expired resume closes directly | `resume_suspended_planning`, `resume_suspended_tools`, `resume_expired_deadline` |
| Continue preserves a planning checkpoint | `continue_planning_checkpoint` |
| Terminal checkpoints reject both request variants | `continue_terminal_rejected`, `resume_terminal_rejected` |
| Resume messages require a plan continuation | `resume_tools_rejects_appended_messages`, `resume_tools_with_messages_invalid` |
| Selectors match reason/source/id/metadata subsets and reject mismatches | `resume_selector_metadata_subset`, `resume_selector_mismatch` |
| Completed, failed, and limit outcomes are closed terminal variants | `final_only`, `model_error_finishes_run`, `max_planning_steps`, `max_tool_calls`, `max_total_tokens` |
| Metrics advance only with committed complete work | `stream_pause_discards_partial`, `parallel_timeout_atomicity`, `stream_reasoning_usage` |
| All seven durable fact kinds occur as single-revision checkpoint units | `final_only`, `resume_suspended_planning`, `one_tool_then_final`, `conversation_insert_during_model`, `history_rewrite_before_model`, `pause_during_model` |

## Run Control and Precedence

| Normative behavior | Cases |
| --- | --- |
| Planning pause discards partial work | `pause_during_model`, `stream_pause_discards_partial` |
| Tool pause waits for the atomic batch boundary | `pause_precedes_tool_insert` |
| Planning insert interrupts and replans | `conversation_insert_during_model` |
| Tool-time insert waits for the next legal plan boundary | `conversation_insert_during_tool` |
| Active cancellation is cooperative; inactive ids are no-ops | `tool_cancel_active`, `tool_cancel_inactive_noop` |
| One deadline bounds model, tool, approval, history, and ordinary repository work, including an inherited expired resume | `deadline_before_model_completion`, `deadline_precedes_tool_pause`, `deadline_during_approval`, `deadline_during_history_rewrite`, `repository_work_deadline_atomicity`, `resume_expired_deadline` |
| Deadline beats pause | `deadline_precedes_tool_pause` |
| A completed terminal step beats a late pause | `terminal_completion_precedes_pause` |
| First waiting result in model order beats completion timing and host controls | `parallel_wait_first_wins`, `waiting_precedes_boundary_controls` |
| Pause beats a queued insert | `pause_precedes_tool_insert` |
| Normal continuation remains the fallback | `final_only`, `one_tool_then_final` |

## Tool Binding, Policy, Execution, and Commit

| Normative behavior | Cases |
| --- | --- |
| Catalog membership and input schema are checked before approval | `unknown_tool_failure`, `tool_input_validation` |
| Output failures and implementation exceptions normalize to tool failures | `tool_output_validation`, `tool_failure_recovery` |
| Invalid calls never reach approval or invocation | `unknown_tool_failure`, `tool_input_validation` |
| Allow, deny, suspend, and malformed approval behavior are closed and guarded | `approval_allow`, `approval_deny`, `approval_suspend`, `approval_policy_guards` |
| Any suspension makes the selected batch atomic and first-in-model-order | `approval_suspend_atomic_batch` |
| Empty/non-prefix/serial-multiple/unsafe/oversized strategy output is rejected | `batch_policy_guards` |
| Parallel declarations require read-only and idempotent facts | `parallel_spec_requires_safe_facts` |
| Batch size and active concurrency are bounded | `tool_batch_size_limit`, `max_tool_concurrency` |
| Calls are invoked once; progress is live and bounded | `one_tool_then_final`, `tool_progress`, `tool_progress_overflow` |
| Physical completion may vary while durable results remain model-ordered; live completions and non-precomputed outcomes match the durable fact | `parallel_tool_ordering`, `trace_tool_fact_missing_completion_invalid` |
| Parallel interruption commits no partial result | `parallel_timeout_atomicity` |
| Accepted and waiting results have distinct closed semantics | `accepted_result_then_final`, `waiting_result_suspends`, `waiting_result_with_pending` |

## Model Streaming

| Normative behavior | Cases |
| --- | --- |
| Content deltas are live while the returned response is the sole durable model result | `stream_final` |
| Tool-call id/name/arguments deltas are assembled by the provider into the returned response and execute normally | `stream_tool_call` |
| Reasoning stays live; cumulative usage merges without clearing omitted fields | `stream_reasoning_usage` |
| A returned response is terminal for one model invocation; no later fixture outcome is consumed | `stream_reasoning_usage` |
| Pause and deadline discard partial content | `stream_pause_discards_partial`, `stream_deadline_discards_partial` |
| A model error after live deltas discards partial content and finishes as protocol failure | `stream_protocol_error_discards_partial` |

## Repository, Wire Values, and Replay

| Normative behavior | Cases |
| --- | --- |
| Checkpoint, fact, event, and trace wires use their single canonical current representation | every scenario, all validation cases |
| Identical checkpoint replay is a no-op; id collisions and stale revisions conflict | `repository_commit_idempotency` |
| Repository failure leaves the prior revision authoritative | `repository_commit_failure` |
| A work-deadline cancellation cannot appear later as a partial ordinary commit | `repository_work_deadline_atomicity` |
| Duplicate call ids, artifact refs, pending calls, and execution facts obey their structural and semantic invariants | `duplicate_tool_call_id_invalid`, `artifact_ref_invalid`, `tools_pending_requires_calls`, `parallel_spec_requires_safe_facts` |
| Multimodal/artifact final content survives exact wire round-trip | `artifact_final` |
| Every scenario produces a verification-valid trace | every scenario (unconditional runner invariant) |
| Revision gaps in checkpoint transitions are rejected | `trace_revision_gap_invalid` |
| Tool-batch facts reject missing live completion evidence | `trace_tool_fact_missing_completion_invalid` |

The inventory count is asserted by `tests/conformance/test_contracts.py`; fixture names
are unique and every fixture is schema-validated before execution.
