# Repository Guidelines

## Project Structure

This repository is a model-neutral agent runtime monorepo. Cross-language
contracts live in `contracts/v0`; portable behavior fixtures live in
`conformance/cases`; design notes live in `docs`; Python workspace packages live
under `python/packages`.

Python packages:

- `kernel`: the OS-like runtime kernel. It owns the execution loop, scheduler,
  model/tool protocols, events, durable state, snapshots, resume, limits,
  approval/store/journal/hook ports, trace/replay, and the public SDK surface.
- `conformance`: CLI and harness for validating implementations.

Retired runtime packages are not allowed back: `engine`, `protocol`,
`run-state`, `extensions`, and `tracing`. Do not add compatibility shims,
deprecated aliases, or re-export packages for those names.

## Dependency Rules

`kernel` must not depend on any internal runtime package, conformance package,
provider adapter, tool pack, app code, concrete store, UI, queue, or deployment
runtime. External implementations depend on `kernel` and are injected through
kernel ports such as `ModelClient`, `Tool`, `ApprovalPolicy`, `RunStore`,
`RunJournal`, and `RuntimeHook`.

`conformance` may depend on `kernel`, but `kernel` must never import
`conformance`.

## Build, Test, And Development Commands

Python must be managed with `uv`; use the Python interpreter selected by `uv`.
Do not use `pip`.

Run commands from the repository root:

```bash
uv sync
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run conformance conformance/cases
uv run python python/packages/kernel/examples/basic_tool_loop.py
uv run python python/packages/kernel/examples/pause_resume_trace.py
```

When editing JSON contracts or conformance cases, validate changed contracts:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

for root in ["contracts/v0", "conformance/cases"]:
    for path in sorted(Path(root).glob("*.json")):
        data = json.loads(path.read_text())
        if path.name.endswith(".schema.json"):
            Draft202012Validator.check_schema(data)
print("json ok")
PY
```

## Coding Style

Python targets 3.11+ with strict Pyright. Ruff enforces imports, bugbear,
modernization, simplification, and `E/F` rules with a 100-character line limit.
Use explicit, model-neutral public types. Add abstractions only when they reduce
real complexity or enforce a boundary.

## Testing Guidelines

Use `pytest` and `pytest-asyncio`. Add focused regression tests for kernel
changes. Core changes to checkpointing, resume, limits, streaming, tool
scheduling, event order, or run traces require focused tests and, when portable,
conformance cases.

## Commit And PR Guidelines

Keep commits focused. Pull requests should include the problem, implementation
summary, validation commands run, and contract/conformance/doc updates.

## Agent-Specific Instructions

This project has no historical compatibility burden. Prefer clean breaking
refactors over compatibility shims or legacy aliases. Keep the kernel boundary
strict. Put portable behavior in `contracts/v0` and `conformance/cases`, not
only in Python code.
