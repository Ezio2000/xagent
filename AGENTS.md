# Repository Guidelines

## Project Structure

This repository is a model-neutral agent runtime monorepo. Cross-language
contracts live in `contracts/v0`; portable behavior fixtures live in
`conformance/cases`; design notes live in `docs`; Python workspace packages live
under `python`.

Python packages:

- `kernel`: the OS-like runtime kernel. It owns the execution loop, scheduler,
  model/tool protocols, events, durable state, snapshots, resume, limits, and
  approval/store/journal/hook ports.
- `toolkit`: default tool registry, JSON Schema validation, and concrete tool
  invocation glue built on kernel protocols.
- `prompting`: prompt and message construction helpers built on kernel message
  types.
- `modelkit`: model adapter helper facade re-exporting kernel stream
  accumulation and capability helper functions.
- `diagnostics`: public trace objects, trace construction, and deterministic
  replay validation.
- `harness`: the controlled kernel assembly and scenario support package around
  `kernel`. It provides thin scenario builders, model drivers, runtime port
  implementations and fakes, tool registries and fixtures, message fixtures,
  event/timeline/trace observation, and behavior assertions without owning
  kernel runtime semantics or conformance contracts.
- `conformance`: CLI and fixture runner for validating implementations against
  contracts and portable behavior fixtures.

Documentation index:

- `contracts/v0/README.md`: cross-language contract map and schema `$id`
  policy.
- `docs/architecture.md`: kernel architecture and package boundary.
- `docs/harness.md`: controlled kernel assembly and scenario support boundary.
- `docs/python-package-boundaries.md`: Python package split, dependency rules,
  and placement checklist.
- `docs/model-protocol.md`: model adapter protocol.
- `docs/tool-protocol.md`: tool spec, scheduling, approval, and output rules.
- `docs/event-stream.md`: runtime event stream and hook-emitted custom events.
- `docs/state-machine.md`: status transitions, checkpoints, pause, and resume.
- `docs/public-api-audit.md`: Python public API naming and export decisions.
- `conformance/case.schema.json`: portable JSON conformance fixture format.
- `conformance/README.md`: shared conformance case format and runner contract.

Retired runtime packages and imports are not allowed back: `agent_runtime`,
`agent_runtime_conformance`, `engine`, `protocol`, `run_state`, `run-state`,
`extensions`, and `tracing`. Do not add compatibility shims, deprecated aliases,
or re-export packages for those names. Do not reintroduce a top-level `sdks/`
source tree.

## Dependency Rules

`kernel` must not depend on any internal runtime package, conformance package,
provider adapter, tool pack, app code, concrete store, UI, queue, or deployment
runtime. External implementations depend on `kernel` and are injected through
kernel ports such as `ModelClient`, `ToolRegistryProtocol`, `ApprovalPolicy`,
`RunStore`, `RunJournal`, and `RuntimeHook`.

Sibling helper packages may depend on `kernel`, but `kernel` must never import
them. `harness` provides controlled kernel assembly and scenario support and
may compose `kernel`, `toolkit`, `prompting`, and `diagnostics` public APIs.
`conformance` may depend on
`kernel`, `toolkit`, `prompting`, `diagnostics`, and `harness`, but none of
those packages may import `conformance`.

Project and package names must not use retired fragments: `xagent`, `agent_`,
`agent-`, `runtime_`, or `runtime-`. The current Python package names are
`kernel`, `toolkit`, `prompting`, `modelkit`, `diagnostics`, `harness`, and
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
uv run conformance conformance/cases --spec-dir contracts/v0
uv run python examples/python/basic_tool_loop.py
uv run python examples/python/pause_resume_trace.py
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
