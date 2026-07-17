# Kernel v0 Conformance Cases

`conformance/cases` contains portable JSON fixtures for the JHarness runtime. The
repository-local Python runner executes them against the same source tree. The format
is defined by `case.schema.json`; unknown fields are invalid.

The normative behavior-to-fixture map is maintained in
[`coverage.md`](coverage.md).

The v0 rewrite deliberately replaces the former case format. Runners do not
support old case types, tool modes, hooks, separate store/journal expectations,
or programmable scheduler fixtures.

## Case Layout

The v0 case directory is deliberately flat. Runners load only direct
`conformance/cases/*.json` children in sorted filename order; nested category
directories are not part of the runner contract. Behavioral grouping belongs
in [`coverage.md`](coverage.md), so one fixture keeps one stable portable path
without coupling readers to an implementation directory taxonomy.

## Case Kinds

### Scenario

A scenario contains an optional seed checkpoint and one or more invocations. Each
invocation declares:

- a start, continue, or resume fixture request;
- deterministic model steps as zero or more live deltas followed by exactly one
  response, error, or blocking outcome;
- optional approval decisions, controller actions, history rewrite, repository
  failure/delay, limits, and controlled invalid batch-policy behavior;
- exact durable expectations and selected live-event expectations.

Continue and resume requests name either the seed or previous checkpoint as
their source; the fixture never duplicates a generated checkpoint. Later
invocations may therefore consume the previous invocation result, allowing one
fixture to prove suspend/resume behavior without embedding generated ids.

Optional configuration is omitted when unused. Empty limits, approval maps,
action lists, and other no-op configuration are not an alternate fixture form.

### Validation

A validation case supplies an arbitrary JSON value, names a v0 contract schema,
and declares whether structural plus portable semantic validation must accept
it. Targets are message, model response, tool spec/result, state, snapshot,
request, checkpoint, event, or trace. This covers invariants that JSON Schema
alone cannot express.

## Runner Requirements

A runner must:

1. load direct case files in sorted filename order;
2. validate every fixture against `case.schema.json`;
3. resolve v0 schema ids from the supplied spec directory without network I/O;
4. implement the deterministic tools in `tools.contract.json`;
5. execute each invocation once through `Runtime`/`Invocation`, observing live
   events and obtaining the authoritative checkpoint from that same execution;
6. validate all emitted wire values against their v0 schemas;
7. assert every field in `expected`;
8. validate generated traces with deterministic single-pass verification;
9. preserve completion-order freedom only for live parallel `tool_finished`
   observations; durable tool-batch facts remain ordered;
10. when `repository_idempotent` is asserted, replay every successful checkpoint
    by id as a no-op without a duplicate semantic commit, and reject both the
    same id with different content and a new id carrying a stale revision;
11. reject obsolete fixture fields rather than silently ignoring them.

## Durable Expectations

Every invocation expectation names the final derived status, final revision,
fact kinds committed by that invocation, message roles, and metrics. Optional
fields assert pending calls, tool outcome kinds, usage, final content,
suspension, failure/limit reason, event counts/order, model request histories,
and repository/request errors. Trace validity is unconditional and is not a
per-case representation field.

Revision and fact assertions are the primary portable durability evidence. Live
event counts cannot substitute for them.

`usage` asserts the complete accumulated durable model usage. `tool_activity`
filters live tool start/finish events into an exact ordered sequence of
`{kind, tool_call_id}` objects where ordering is itself portable.
`max_active_tools` derives the peak number of started but unfinished calls and
asserts concurrency limits without constraining legal physical completion
order. Completion-order freedom is asserted by omitting `tool_activity`.

`batch_policy` is an optional conformance-only fault injection. It proves that a host
strategy cannot select an empty/non-prefix batch, combine
serial calls, parallelize unsafe tools, or exceed the batch limit. It is not a
runtime wire field.

Approval, history-reducer, and repository delay fields are also
conformance-only. They prove that the one work deadline covers every ordinary
effect and that cancellation leaves the prior durable checkpoint authoritative.

## Standard Tools

The machine-readable standard catalog is
[`tools.contract.json`](tools.contract.json), validated by
[`tools.contract.schema.json`](tools.contract.schema.json). Human behavior notes
are in [`tools.md`](tools.md). Support-package fixtures are not normative.

## Python Runner

Run the local suite from the repository root:

```bash
uv run python -m conformance.cli conformance/cases --spec-dir contracts/v0
```

The runner consumes the canonical cases and contracts directly from this repository.
It must not create or maintain a synchronized fixture copy.
