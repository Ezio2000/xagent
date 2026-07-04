# Runtime Conformance Cases

`conformance/cases` contains portable behavior fixtures for every SDK. Cases are
JSON objects. Unknown keys are invalid so misspelled expectations fail early.

Run the Python reference runner from the repository root:

```bash
uv run conformance conformance/cases
```

Use `--spec-dir contracts/v0` when running against a non-standard checkout
layout.

## Implementing A Runner For Another SDK

The case files are the cross-SDK input. The Python runner is the reference
implementation and reference harness for current behavior, but the normative
inputs are the fixtures, `contracts/v0`, and this document. Other SDKs should
implement their own runner against the same JSON fixtures instead of importing
Python code or depending on Python object layouts.

A non-Python runner should provide the same deterministic harness:

- Load every `*.json` case in sorted order and reject unknown keys.
- Validate fixture fields that map to v0 schemas, plus runner-specific field
  checks. During execution, validate emitted events, snapshots, result messages,
  resume input, and traces against the v0 schemas.
- Convert `model_steps` and `resume_model_steps` into scripted model responses.
- Emit stream events from `stream_model_steps` when the case requests streaming.
- Provide the standard conformance tools: `echo`, `accept`, `handoff`, `fail`,
  `delayed_echo`, `wait`, `parallel_wait`, and `strict_count`.
- Assert every `expected_*` field and every `forbidden_*` field present in the
  case. Absence of an expectation means the runner should not assert it.

The runner may use SDK-native classes, async primitives, error types, and test
frameworks. The portable contract is the JSON fixture, emitted event/snapshot
wire shape, final status, trace replay result, and documented behavior.
