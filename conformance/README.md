# Agent Runtime Conformance Cases

`conformance/cases` contains portable behavior fixtures for every SDK. Cases are
JSON objects. Unknown keys are invalid so misspelled expectations fail early.

`case_type` defaults to `run`. Current case types are `run`, `resume`, and
`model_response_negative`.

## Shared Conventions

All case files must define `name`. Runtime cases also define `model_steps`,
`expected_status`, and `expected_tool_calls`.

`model_steps` and `resume_model_steps` are arrays of
`model-response.schema.json` objects. Each step currently requires `parts` and
`tool_calls`; optional fields include `finish_reason`, `usage`, `model`,
`response_id`, and `metadata`.

`limits`, when present, uses `limits.schema.json` and may include
`max_iterations`, `max_total_tool_calls`, `timeout_seconds`,
`stop_on_tool_error`, and `max_parallel_tool_calls`.

`stream_model_steps`, when present, is an array of objects with `events`.
Supported stream event types are:

- `text_delta`: requires `index`, `text_delta`, and `part_type`.
- `tool_call_delta`: requires `index` and may include `id`, `name`, and
  `arguments_delta`.
- `sleep`: requires `seconds`.
- `pause_request`: requests the configured stream-time pause.

Expectation fields describe assertions made by the runner. Fields whose names
start with `forbidden_` assert that a bad intermediate state or event did not
appear.

## Common Field Shapes

`pause_request` uses the `PauseRequest` shape:

```json
{
  "reason": "host_requested",
  "source": "host",
  "wait_id": null,
  "metadata": {},
  "interrupt": false
}
```

`expected_pause` uses the compact paused-state shape:

```json
{
  "reason": "external_callback",
  "resume_status": "planning",
  "source": "tool",
  "wait_id": "job-1",
  "metadata": {}
}
```

`resume_expected_pause` uses the `PauseSelector` shape. All four keys are
present; `reason`, `source`, and `wait_id` may be `null`, but at least one
selector field or metadata entry must be set.

```json
{
  "reason": null,
  "source": null,
  "wait_id": "job-1",
  "metadata": {}
}
```

`forbidden_checkpoint_status_tool_counts` is an array of status/count pairs:

```json
[
  { "status": "planning", "total_tool_calls": 1 }
]
```

## `run`

`run` cases execute a fresh runtime invocation from an initial user message.
They cover normal completion, limits, streaming, pause boundaries, tool errors,
parallel tool scheduling, and external waits.

Required keys:

- `name`
- `model_steps`
- `expected_status`
- `expected_tool_calls`

Common optional keys:

- `limits`
- `pause_request`
- `pause_request_timing`
- `stream_model_steps`
- `expected_final_text`
- `expected_message_roles`
- `expected_tool_texts`
- `expected_pending_tool_call_ids`
- `expected_pause`
- `expected_model_deltas`
- `forbidden_event_types`
- `forbidden_checkpoint_statuses`
- `forbidden_checkpoint_tool_counts`
- `forbidden_checkpoint_status_tool_counts`
- `forbidden_unpaused_checkpoint_tool_counts`
- `forbidden_checkpoint_message_roles`

`pause_request_timing`, when present, is either `during_model_call` or
`stream_event`.

## `resume`

`resume` cases first run to a checkpoint selected by the case, then resume that
checkpoint through `ResumeInput`. They cover paused snapshots, planning and
executing-tools checkpoints, expected-pause matching, appended resume messages,
and terminal snapshot rejection.

`resume` cases use all `run` keys and additionally require:

- `resume_checkpoint_status`
- either `expected_resume_status` or `expected_resume_error`

Resume-only optional keys:

- `resume_model_steps`
- `resume_append_messages`
- `resume_expected_pause`
- `resume_checkpoint_total_tool_calls`
- `expected_resume_final_text`
- `expected_resume_tool_calls`
- `expected_resume_message_roles`
- `expected_resume_tool_texts`
- `expected_resume_trace_prefix`

Resume-only keys are invalid unless `case_type` is `resume`. This keeps normal
run cases from accidentally depending on resume-specific runner behavior.

## `model_response_negative`

`model_response_negative` cases validate portable model-response constructor
rules that cannot be fully expressed in JSON Schema. The case's
`model_response` must still match `model-response.schema.json`, then the SDK
must reject it with an error containing `expected_error`.

Required keys:

- `name`
- `case_type`
- `model_response`
- `expected_error`

No other keys are allowed. Negative-only keys `model_response` and
`expected_error` are invalid in `run` and `resume` cases.

## Adding Cases

Add or update a conformance case when behavior must be shared by every SDK.
Keep Python-only implementation details in Python tests instead.

When adding a case:

1. Choose the narrowest case type.
2. Use existing tool names and model-step shapes when possible.
3. Add only expectations needed to prove the portable behavior.
4. Run the Python conformance tests and JSON/schema validation.
5. Update this README if a new case type, field, tool, or stream event is added.
