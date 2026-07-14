# Repository Guidelines

## Scope

This repository is the language-neutral JHarness specification. Normative wire
contracts live in `contracts/v0`, portable behavior fixtures live in
`conformance/cases`, and architecture documentation lives in `docs`.

Runtime implementations do not live here. The Python implementation is maintained
in `Ezio2000/jharness-python`; the planned Go implementation is maintained in
`Ezio2000/jharness-go`.

## Specification Rules

- Keep one flat lifecycle: `Planning`, `ToolsPending`, `Suspended`, `Completed`,
  `Failed`, and `Limited`.
- Keep one runtime/invocation execution, one model invocation operation, one tool
  invocation operation, and one atomic checkpoint boundary.
- Keep schemas explicit and portable; never derive them from a language runtime.
- Do not add language-specific package layouts, provider clients, test doubles, or
  alternate protocol representations to this repository.
- A portable behavior change updates normative documentation, schemas, coverage,
  and matching conformance cases in the same change.
- Replace protocol shapes directly within an unreleased contract family. Released
  specification tags are immutable.

## Development

Python is used only for repository validation and must be managed with `uv`; never
use `pip`.

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python scripts/validate_spec.py
```

## Completion Standard

Do not claim a specification change complete until all schemas resolve without
network access, every JSON document validates, every case is represented in the
coverage matrix, local links resolve, and at least one implementation has an
explicit follow-up path for the new specification release.
