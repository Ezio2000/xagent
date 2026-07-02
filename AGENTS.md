# Repository Guidelines

## Project Structure & Module Organization

This repository is a model-neutral agent loop runtime designed for multiple
language SDKs. Cross-language contracts live in `spec/v0`; portable behavior
fixtures live in `conformance/cases`; design notes live in `docs`. The current
reference SDK is Python under `sdks/python`, with package code in
`sdks/python/src/agent_runtime`, conformance tooling in
`sdks/python/src/agent_runtime_conformance`, tests in `sdks/python/tests`, and
runnable examples in `sdks/python/examples`. `sdks/typescript` and `sdks/go` are
reserved for future SDKs.

## Build, Test, and Development Commands

Use `uv` for all Python work, use the Python interpreter selected by `uv`, and
do not use `pip`. Run Python commands from `sdks/python`:

```bash
uv sync                                      # install dependencies
uv run pytest -q -p no:cacheprovider        # run tests
uv run ruff check .                         # lint
uv run ruff format --check .                # check formatting
uv run pyright                              # type-check
uv run agent-runtime-conformance ../../conformance/cases
uv run python examples/basic_tool_loop.py
uv run python examples/pause_resume_trace.py
```

When editing JSON specs or conformance cases, parse all changed contracts:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

for root in ["../../spec/v0", "../../conformance/cases"]:
    for path in sorted(Path(root).glob("*.json")):
        data = json.loads(path.read_text())
        if path.name.endswith(".schema.json"):
            Draft202012Validator.check_schema(data)
print("json ok")
PY
```

## Coding Style & Naming Conventions

Python targets 3.11+ with strict Pyright. Ruff enforces imports, bugbear,
modernization, simplification, and `E/F` rules with a 100-character line limit.
Use 4-space indentation and explicit, model-neutral public types. Keep module
names direct, for example `loop.py`, `models.py`, `messages.py`, `scheduler.py`,
and `tools.py`.

## Testing Guidelines

Use `pytest` and `pytest-asyncio`. Test files should be named `test_*.py`, and
test functions should describe the behavior being protected. Add Python
regression tests for narrow runtime changes and conformance cases for behavior
all SDKs must share. Core changes to checkpointing, resume, limits, streaming,
tool scheduling, event order, or run traces require focused tests.

## Commit & Pull Request Guidelines

Recent commits use concise imperative summaries, for example `Update repository
metadata`. Keep commits focused. Pull requests should include the problem,
implementation summary, validation commands run, and related issues. For
contract changes, call out updated `spec/v0`, conformance cases, and docs.

## Agent-Specific Instructions

This project has no historical compatibility burden. Prefer clean breaking
refactors over shims or legacy aliases. Keep provider adapters, persistence,
approval flows, plugins, UI integrations, queues, and deployment runtime outside
core. Put portable behavior in `spec/v0` and `conformance/cases`, not only in
Python code. Use sub-agent CRs for bounded review; report `Must-Fix`,
`Should-Fix`, and `Looks Good`, and fix credible `Must-Fix` items before
handoff.
