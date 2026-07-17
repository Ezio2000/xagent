# Repository Guidelines

## Scope

This repository is the complete JHarness Python project. It owns four coordinated,
independently installable distributions:

- `jharness-kernel`, providing `jharness.kernel`;
- `jharness-toolkit`, providing `jharness.toolkit`;
- `jharness-models`, providing `jharness.models`;
- `jharness-tools`, providing `jharness.tools`.

Contracts, conformance, examples, benchmarks, tests, documentation, and release
automation live in this repository and use one coordinated version and release tag.

## Design Rules

- Prefer high cohesion, low coupling, and reusable components with one clear owner.
- Keep one flat lifecycle: `Planning`, `ToolsPending`, `Suspended`, `Completed`,
  `Failed`, and `Limited`.
- Keep one runtime/invocation execution, one model invocation operation, one tool
  invocation operation, and one atomic checkpoint boundary.
- Keep public values immutable, extension ports narrow and async, policies pure,
  concurrency bounded, and deadlines monotonic.
- Keep schemas and wire codecs explicit. Do not derive persisted shapes through
  reflection or make external API payloads the durable representation.
- Do not add compatibility distributions, obsolete import forwarding, duplicate
  lifecycle implementations, service locators, or programmable schedulers.
- Do not retain generated build, coverage, synchronization, or migration artifacts.

## Dependency Direction

`jharness.kernel` is the foundation and must not import another JHarness package.
Toolkit, models, and tools may import the public kernel API but must not depend on one
another.

| Distribution | Direct runtime dependencies |
| --- | --- |
| `jharness-kernel` | None |
| `jharness-toolkit` | exact matching `jharness-kernel`, `jsonschema`, `referencing` |
| `jharness-models` | exact matching `jharness-kernel`, `httpx` |
| `jharness-tools` | exact matching `jharness-kernel`, `regex` |

Every distribution owns only its `jharness.<component>` namespace portion and nested
`py.typed` marker. No wheel owns `jharness/__init__.py`.

## Contract Ownership

- Python domain values and runtime behavior live under `packages/jharness-kernel`.
- Portable top-level JSON shapes live under `contracts/v0` and are encoded explicitly
  by `jharness.kernel.wire`.
- External model HTTP/SSE shapes stay in `jharness.models`; tool argument and
  structured-result validation stay in `jharness.toolkit`.
- Any observable durable behavior change updates code, schemas, normative Markdown,
  conformance cases, `conformance/coverage.md`, and focused tests together.
- Replace obsolete shapes and package names directly; do not preserve old paths.

## Development

Python is managed exclusively with `uv`:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q -p no:cacheprovider
uv run python scripts/validate_spec.py
uv run python -m conformance.cli conformance/cases --spec-dir contracts/v0 --quiet
uv run python benchmarks/runtime_smoke.py
uv build --all-packages --out-dir dist
uv run python scripts/verify_distribution.py dist
```

## Completion Standard

Do not claim a change complete until relevant tests, formatting, lint, strict types,
offline schema resolution, conformance, all four package builds, isolated artifact
imports, and the performance smoke benchmark pass. Keep the worktree free of
intermediate artifacts and report the modified directory list after every code change.
