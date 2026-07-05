# Runtime Conformance Cases

`conformance/cases` contains portable behavior fixtures for every SDK. Cases are
JSON objects described by `conformance/case.schema.json`. Unknown keys are
invalid so misspelled expectations fail early.

Run the Python reference runner from the repository root:

```bash
uv run conformance conformance/cases
```

Use `--spec-dir contracts/v0` when running against a non-standard checkout
layout.

## Implementing A Runner For Another SDK

The case files are the cross-SDK input, and `conformance/case.schema.json`
defines their shared format. The Python runner is the reference implementation
and reference fixture runner for current behavior. It may use the Python
`harness` package for controlled kernel assembly and scenario support in
deterministic runs, but the normative inputs are the fixtures,
`conformance/case.schema.json`, `contracts/v0`, and this document.
Other SDKs should implement their own runner against the same JSON fixtures
instead of importing Python code or depending on Python object layouts.

A non-Python runner should provide the same deterministic fixture runner:

- Load every `*.json` case in sorted order and reject unknown keys.
- Validate fixture fields that map to v0 schemas, plus runner-specific field
  checks. During execution, validate emitted events, snapshots, result messages,
  model requests, tool outputs, resume input, and traces against the v0 schemas.
- Convert `model_steps` and `resume_model_steps` into scripted model responses.
- Emit stream events from `stream_model_steps` when the case requests streaming.
- Provide the standard conformance tools: `echo`, `accept`, `handoff`, `fail`,
  `delayed_echo`, `wait`, `progress`, `parallel_wait`, and `strict_count`.
  The Python runner owns those portable tool behaviors in
  `python/conformance/src/conformance/_standard_tools.py`; harness tool
  fixtures are separate runtime scenario components and must not define
  conformance semantics.
- Assert every `expected_*` field and every `forbidden_*` field present in the
  case. Absence of an expectation means the runner should not assert it.

The runner may use SDK-native classes, async primitives, error types, and test
frameworks. The portable contract is the JSON fixture, emitted event/snapshot
wire shape, final status, trace replay result, and documented behavior.
