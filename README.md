# xagent

Model-neutral agent runtime monorepo.

The Python SDK is now centered on a single OS-like `kernel` package. `kernel`
owns the execution trunk and the extension slots. Provider adapters, tool packs,
storage backends, approval UIs, queues, and product applications live outside the
kernel and are injected through its public ports.

## Package Structure

| Path | Import package | Responsibility |
|---|---|---|
| `contracts/v0` | none | Cross-language JSON Schemas and portable runtime contract docs. |
| `conformance/cases` | none | Shared JSON behavior fixtures that every SDK must pass. |
| `python/packages/kernel` | `kernel` | Runtime kernel: loop, scheduler, model/tool protocols, events, state, snapshots, resume, limits, approval/store/journal/hook ports, trace/replay, and public SDK exports. |
| `python/packages/conformance` | `conformance` | Python conformance CLI, case loader, scripted harness, validators, and assertions. |

## Dependency Rules

| Package | Allowed dependencies | Forbidden dependencies |
|---|---|---|
| `kernel` | Python stdlib and runtime-neutral validation libraries needed by kernel-owned protocols | Any internal runtime package, conformance, provider adapter, tool pack, app, concrete store, UI, queue, or deployment runtime |
| `conformance` | `kernel`, JSON Schema validation libraries | Being imported by `kernel` |

Retired runtime packages must not reappear: `engine`, `protocol`, `run-state`,
`extensions`, and `tracing`. Their responsibilities were folded into `kernel` or
made kernel ports. There are no compatibility shims or legacy aliases.

## Python Development

Use `uv` for all Python work. Do not use `pip`.

```bash
uv sync
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run conformance conformance/cases
```

Run examples from the repository root:

```bash
uv run python python/packages/kernel/examples/basic_tool_loop.py
uv run python python/packages/kernel/examples/pause_resume_trace.py
```

Validate changed contracts:

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

## Public SDK Surface

Applications import the runtime SDK from `kernel`:

```python
from kernel import AgentLoop, Message, ModelResponse, RuntimeContext
```

Extension implementations depend on `kernel` and plug into it:

- model providers implement `ModelClient` or `StreamingModelClient`;
- tools implement `Tool` or expose `ToolSpec` plus execution methods;
- approval systems implement `ApprovalPolicy`;
- persistence systems implement `RunStore` and `RunJournal`;
- lifecycle integrations implement `RuntimeHook`;
- audit systems can consume kernel events or use `RunTrace.from_events`.

`kernel` never imports those concrete implementations. The host application
constructs them and injects them into `AgentLoop`.

## Core Boundary

The kernel provides only reusable runtime infrastructure:

- model-neutral message, model, event, and tool protocols;
- execution loop, bounded scheduling, pause/resume, checkpointing, and event
  ordering;
- durable state, snapshots, limits, run control, and resume inputs;
- approval, hook, store, and journal ports for host-owned implementations;
- trace recording and replay validation for portable diagnostics;
- portable conformance fixtures and a Python conformance runner.

The repository deliberately does not include provider adapters, concrete
storage backends, approval UIs, tool packs, artifact stores, memory systems, MCP
clients, job queues, deployment runtimes, or product applications. Those belong
above the kernel.

## Change Policy

There is no historical compatibility burden. Prefer clean breaking refactors
over compatibility shims, deprecated aliases, or duplicate transitional APIs.
Portable behavior belongs in `contracts/v0` and `conformance/cases`, not only in
Python code.
