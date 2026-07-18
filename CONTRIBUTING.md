# Contributing

JHarness is one Python project. A pull request should keep runtime code, public API,
portable persistence contracts, conformance evidence, tests, examples, documentation,
and release metadata consistent within the same repository.

## Choose the Owning Area

- Runtime state, messages, model/tool ports, checkpointing, repository behavior, wire
  codecs, and diagnostics belong in `jharness.kernel`.
- Tool registration, function adaptation, JSON Schema validation, retry, and circuit
  breaking belong in `jharness.toolkit`.
- Model transports, profiles, errors, and request/response codecs belong in
  `jharness.models`.
- Reusable filesystem, shell, interaction, and child-agent implementations belong in
  `jharness.tools`.
- Persisted JSON shapes belong in `contracts/v0`; deterministic behavior evidence
  belongs in `conformance/cases` and `conformance/coverage.md`.

Preserve the dependency direction documented in
[`docs/python-package-boundaries.md`](docs/python-package-boundaries.md). Prefer a
small cohesive change over a new abstraction without a distinct invariant. Replace
obsolete internal structures directly; do not add legacy aliases, forwarding packages,
or migration-only runtime branches.

## Validate Locally

Use `uv` from the repository root:

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

Run the complete gate for architectural, dependency, contract, or release changes.
Focused implementation changes may begin with targeted tests, but the pull request must
finish with all affected gates passing. Remove build, test, coverage, and temporary
artifacts before handoff.

Preview generated artifacts with `uv run python scripts/clean_workspace.py`, then add
`--apply` only after reviewing the exact targets. The cleaner never traverses `.git`
or directories named `.venv`, `venv`, or `ENV` at any nesting depth, so project-local
and package-local environments remain outside cleanup ownership.

## Durable Behavior Changes

A change to observable durable behavior must update, in one pull request:

1. the Python implementation;
2. the relevant normative document under `docs` or `contracts/v0`;
3. every affected JSON Schema;
4. matching conformance fixtures and the coverage matrix;
5. focused unit and integration tests;
6. public examples and changelog entries when the user-facing behavior changes.

Schemas must resolve without network access. Codecs must reject unknown fields,
invalid discriminators, non-finite numbers, and broken cross-field invariants at the
trust boundary.

## Releases

The repository publishes four coordinated wheels and four source distributions from
one version and one immutable tag. Follow [`docs/releasing.md`](docs/releasing.md);
never publish a partial set or upload artifacts built from a developer worktree.
