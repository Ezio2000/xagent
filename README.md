# Model-Neutral Agent Runtime

Model-neutral agent runtime monorepo.

The Python SDK is now centered on a single OS-like `kernel` package. `kernel`
owns the execution trunk and the extension slots. Provider adapters, tool packs,
storage backends, approval UIs, queues, and product applications live outside the
kernel and are injected through its public ports.

## Repository Index

| Area | Path | Use for |
|---|---|---|
| Contract map | `contracts/v0/README.md` | Cross-language wire shapes, semantic contract ownership, and schema `$id` policy. |
| Architecture notes | `docs/architecture.md` | Kernel boundary and runtime architecture overview. |
| Harness guide | `docs/harness.md` | Workspace-level controlled test harness layers and package boundary. |
| Model protocol | `docs/model-protocol.md` | Model adapter request/response rules. |
| Tool protocol | `docs/tool-protocol.md` | Tool spec, invocation, approval risk, scheduling, and output rules. |
| Event stream | `docs/event-stream.md` | Runtime event names, event ordering, and hook-emitted custom events. |
| State machine | `docs/state-machine.md` | Status meanings, transitions, checkpoints, pause, and resume behavior. |
| Public API audit | `docs/public-api-audit.md` | Exported Python SDK names and public/internal boundary decisions. |
| Conformance guide | `conformance/README.md` | Shared case format and runner expectations for SDK implementations. |
| Conformance case schema | `conformance/case.schema.json` | Portable JSON fixture format consumed by SDK conformance runners. |
| CI workflow | `.github/workflows/ci.yml` | Required repository validation gate. |
| Agent instructions | `AGENTS.md` | Coding-agent operating rules generated from the current repo shape. |

## Repository Structure

| Path | Import package | Responsibility |
|---|---|---|
| `contracts/v0` | none | Cross-language JSON Schemas and portable runtime contract docs. |
| `conformance/case.schema.json` / `conformance/cases` | none | Shared JSON behavior fixture format and cases that every SDK must pass. |
| `docs` | none | Architecture, public API, model/tool/event/state protocol notes. |
| `python/kernel` | `kernel` | Runtime kernel: loop, scheduler, model/tool protocols, events, state, snapshots, resume, limits, approval/store/journal/hook ports, trace payload emission, and public SDK exports. |
| `python/toolkit` | `toolkit` | Default tool registry, JSON Schema validation, and concrete invocation glue. |
| `python/prompting` | `prompting` | Prompt and message construction helpers built on kernel message types. |
| `python/modelkit` | `modelkit` | Model adapter helper facade re-exporting kernel stream accumulation and capability normalization. |
| `python/diagnostics` | `diagnostics` | Public trace objects, trace construction, and deterministic replay validation. |
| `python/harness` | `harness` | Workspace-level controlled test harness: model drivers, fake runtime ports, tool stubs and registry doubles, message fixtures, event/timeline/trace observation, scenario helpers, and behavior assertions. |
| `python/conformance` | `conformance` | Python conformance CLI, case loader, fixture runner, validators, and assertions. |
| `pyproject.toml` / `uv.lock` | none | Root uv workspace, dependency groups, lint, type-check, and test configuration. |

## Dependency Rules

| Package | Allowed dependencies | Forbidden dependencies |
|---|---|---|
| `kernel` | Python stdlib and its own declared third-party dependencies | Any internal runtime package, conformance, provider adapter, tool pack, app, concrete store, UI, queue, or deployment runtime |
| `toolkit` | `kernel`, JSON Schema validation libraries | Agent loop, state machine, provider adapters, conformance runner |
| `prompting` | `kernel` | Runtime state, scheduling, model/provider clients |
| `modelkit` | `kernel` | Runtime loop, tool execution, prompt helpers |
| `diagnostics` | `kernel` | Running agents, invoking tools, provider adapters, persistence backends |
| `harness` | `kernel`, `toolkit`, `prompting`, `diagnostics` | Production runtime semantics, portable conformance contracts, schema validation rules, provider adapters, or app infrastructure |
| `conformance` | `kernel`, `toolkit`, `prompting`, `diagnostics`, `harness`, JSON Schema validation libraries | Being imported by runtime packages |

Hard boundary rules:

- `kernel` must not import any sibling workspace package. External
  implementations depend on `kernel` and are injected through public ports.
- Sibling helper packages may import `kernel`, but `kernel` must never import
  them.
- `harness` is test infrastructure and may compose `kernel`, `toolkit`,
  `prompting`, and `diagnostics` public APIs. Runtime source packages must not
  import `harness`.
- `conformance` may import `kernel`, `toolkit`, `prompting`, `diagnostics`, and
  `harness`; runtime packages must never import `conformance`.
- No top-level `sdks/` source tree. Python packages live under `python`.
- Retired runtime imports must not reappear: `agent_runtime`,
  `agent_runtime_conformance`, `engine`, `protocol`, `run_state`,
  `run-state`, `extensions`, and `tracing`.
- Project and package names must not use the retired fragments `xagent`,
  `agent_`, `agent-`, `runtime_`, or `runtime-`. The current package names are
  `kernel`, `toolkit`, `prompting`, `modelkit`, `diagnostics`, `harness`, and
  `conformance`.
- Do not add compatibility shims, deprecated aliases, or re-export packages for
  old names.
- Portable behavior changes require updates to `contracts/v0`,
  `conformance/cases`, and relevant docs in the same change.

## Python Development

Use `uv` for all Python work. Do not use `pip`.

```bash
uv sync
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run conformance conformance/cases --spec-dir contracts/v0
```

Run examples from the repository root:

```bash
uv run python examples/python/basic_tool_loop.py
uv run python examples/python/pause_resume_trace.py
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
- audit systems can consume kernel events or use `diagnostics.RunTrace`.

`kernel` never imports those concrete implementations. The host application
constructs them and injects them into `AgentLoop`.

## Core Boundary

The kernel provides only reusable runtime infrastructure:

- model-neutral message, model, event, and tool protocols;
- execution loop, bounded scheduling, pause/resume, checkpointing, and event
  ordering;
- durable state, snapshots, limits, run control, and resume inputs;
- approval, hook, store, and journal ports for host-owned implementations;
- immutable trace payload emission for portable diagnostics.

Portable conformance fixtures live under `conformance/cases`; the Python
conformance runner lives in the `conformance` package.

The repository deliberately does not include provider adapters, concrete
storage backends, approval UIs, tool packs, artifact stores, memory systems, MCP
clients, job queues, deployment runtimes, or product applications. Those belong
above the kernel.

## Change Policy

There is no historical compatibility burden. Prefer clean breaking refactors
over compatibility shims, deprecated aliases, or duplicate transitional APIs.
Portable behavior belongs in `contracts/v0` and `conformance/cases`, not only in
Python code.
