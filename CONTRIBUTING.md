# Contributing

JHarness specification changes begin with the portable behavior, not an
implementation API. A change that affects observable behavior must update the
normative documentation, `contracts/v0`, matching conformance cases, and
`conformance/coverage.md` together.

Validation tooling is managed exclusively with `uv`:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run python scripts/validate_spec.py
```

After a specification release, implementation repositories update their pinned
specification revision in separate pull requests. Do not put Python, Go, or provider
implementation code in this repository.
