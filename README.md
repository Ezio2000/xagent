# agent-runtime

Model-neutral agent loop runtime for building reusable agent SDKs.

The project is organized for multiple language SDKs. The Python SDK is the
reference implementation today; TypeScript and Go directories are reserved for
future SDKs.

External applications should depend on a language SDK package, not on the
repository root. For Python, the package name is `agent-runtime` and the import
package is `agent_runtime`.

## Using The Python SDK

For a published Python package:

```bash
uv add agent-runtime
```

If you want to depend on this repository before a package release, point your
dependency at the Python SDK subdirectory:

```bash
uv add "agent-runtime @ git+https://github.com/Ezio2000/agent-runtime.git@main#subdirectory=sdks/python"
```

For local development against this checkout:

```bash
uv add --editable /path/to/agent-runtime/sdks/python
```

Minimal usage:

```python
import asyncio

from agent_runtime import AgentLoop, Message, ModelRequest, ModelResponse, RuntimeContext


class EchoModel:
    async def complete(self, request: ModelRequest, context: RuntimeContext) -> ModelResponse:
        _ = context
        return ModelResponse.text(request.messages[-1].text)


async def main() -> None:
    agent = AgentLoop(model=EchoModel())
    result = await agent.run([Message.user_text("hello")])
    print(result.final_parts[0].text)


asyncio.run(main())
```

`agent_runtime_conformance` is packaged with the Python SDK for SDK development
and contract validation. Most application code should import only
`agent_runtime`.

## What This Core Provides

`agent-runtime` focuses on the reusable runtime layer:

- Agent loop state machine: `planning`, `executing_tools`, `paused`,
  `completed`, `failed`, and `limit_exceeded`.
- Provider-neutral model protocol: messages, tools, model options, tool choice,
  response format, usage, capabilities, structured provider errors, and
  streaming deltas.
- Event stream: `run_started`, `model_started`, `model_delta`, `model_error`,
  `model_completed`, `tool_started`, `tool_completed`, `state_changed`,
  `pause_requested`, `checkpoint`, `final`, `error`, `run_paused`, and
  `run_completed`.
- Durable checkpoints and snapshots for host-owned persistence and resume.
- Run-control primitives for pausing at durable boundaries, interrupting model
  generation without committing partial output, and pausing for external
  callbacks.
- Strict resume input validation, compact run traces, and deterministic replay
  checks for core runtime behavior.
- Tool scheduling, including conservative parallel execution for explicitly
  safe, read-only, idempotent tools.
- Hooks for observing or rewriting model/tool boundaries.
- Open multimodal message parts for text, image, file, and future content types.

The core deliberately does not include provider adapters, persistence stores,
approval policies, UI rendering, tool packs, plugin systems, or deployment
runtime. Those should layer on top of the SDK through stable protocols.

## Runtime Model

The runtime is a state machine plus an ordered event stream. `checkpoint` is the
durable resume boundary. `model_delta` is live rendering progress only and must
not be required for resume. A paused run is represented by a `paused` snapshot
with pause metadata and a `resume_status`; `run_snapshot(ResumeInput(...))`
restores that status and continues from the durable boundary.

Tool execution is serial by default. If `LoopLimits.max_parallel_tool_calls > 1`,
the scheduler may run a consecutive batch concurrently only when every tool in
that batch declares:

- `parallel_safe: true`
- `read_only: true`
- `idempotent: true`

Parallel batches commit tool observations atomically in model-provided order.
If a timeout interrupts a parallel batch, the next checkpoint remains at the last
fully committed boundary; observed but uncommitted idempotent calls may be rerun
when resuming from that prior non-terminal checkpoint.

## Repository Quick Start

```bash
cd sdks/python
uv sync
uv run python examples/basic_tool_loop.py
uv run python examples/pause_resume_trace.py
```

For tool usage, streaming, multimodal messages, snapshots, pause/resume, and
run traces, see `sdks/python/README.md`,
`sdks/python/examples/basic_tool_loop.py`, and
`sdks/python/examples/pause_resume_trace.py`.

## Model Protocol

Model adapters implement:

```python
async def complete(request, context) -> ModelResponse: ...
```

Streaming adapters may additionally implement `stream()` and advertise
`ModelCapabilities(streaming=True)`:

```python
def stream(request, context) -> AsyncIterator[ModelStreamEvent]: ...
```

`stream()` must return an async iterator directly, usually from an async
generator. The runtime forwards stream progress as `model_delta` events and
commits `AgentState` only after a complete `ModelResponse` exists. If streaming
is not advertised, `stream=True` callers use the normal `complete()` path.

Provider adapters should translate `ModelOptions`, `ToolChoice`,
`ResponseFormat`, multimodal `ContentPart` values, and `ModelCapabilities` into
their concrete provider API. Adapter-specific data must stay in adapter-owned
objects or explicit `metadata` fields; it must not become runtime control flow,
checkpoint state, message wire fields, or trace replay data.
`ToolChoice` and `ResponseFormat` names describe provider-neutral runtime
intentions; they are not tied to any one provider's request shape.

## Project Structure

- `spec/v0`: cross-language contracts for state, messages, events, tools,
  limits, snapshots, resume input, run control, run trace, model requests, model
  responses, model errors, and streaming. Start with `spec/v0/README.md`.
- `conformance/cases`: shared behavior cases every SDK should pass. Case types
  and expectations are documented in `conformance/README.md`.
- `docs`: design notes for architecture, event streams, state machine, model
  protocol, tool protocol, and public API audit decisions.
- `sdks/python`: self-contained reference SDK implementation managed with `uv`,
  including package code, tests, examples, and Python conformance tooling.
- `sdks/typescript`: reserved TypeScript SDK location.
- `sdks/go`: reserved Go SDK location.

The intended package boundary is:

- `sdks/python/src/agent_runtime`: Python runtime SDK used by applications.
- `sdks/python/src/agent_runtime_conformance`: Python conformance CLI and
  shared-case runner for SDK development.
- Future `sdks/typescript` and `sdks/go` implementations should be independently
  publishable packages that follow `spec/v0` and pass `conformance/cases`.

## Development Principles

- Core first: keep the loop, state machine, event stream, message protocol,
  limits, snapshots, hooks, and tool scheduling model-neutral.
- Open for extension: provider adapters, persistence, approvals, plugins, tool
  packs, and UI integrations stay outside core.
- Break cleanly when needed: this project has no historical compatibility burden
  yet. Prefer clear breaking refactors over compatibility shims or duplicate
  transitional APIs.
- No legacy baggage: do not add deprecated aliases, fallback protocols, or
  adapter-specific exceptions unless they are explicitly part of `spec/v0`.
- Spec before surface area: portable behavior belongs in `spec/v0` and
  conformance cases, not only in Python code.
- Performance matters: avoid event-loop blocking, keep parallelism bounded, and
  make hot-path copying intentional.

## Development Workflow

Use `uv` for Python dependency management. Do not rely on global Python
packages.

```bash
cd sdks/python
uv sync
uv run pytest -q -p no:cacheprovider
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run agent-runtime-conformance ../../conformance/cases
uv run python examples/basic_tool_loop.py
uv run python examples/pause_resume_trace.py
```

When JSON spec or conformance files change, also parse them:

```bash
cd sdks/python
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

Core behavior changes should update the Python runtime, focused Python tests,
portable conformance cases, `spec/v0`, and relevant docs together.

## Sub-Agent CR Guidelines

Use sub-agents for bounded code review or verification tasks when parallel
review materially improves confidence. Prompts should include:

- The exact behavior or risk to review.
- The files or modules that define the behavior.
- Whether the agent is read-only or owns a disjoint write scope.
- The validation commands it should run.
- The required report format: `Must-Fix`, `Should-Fix`, and `Looks Good`.

For core runtime changes, use five fixed review lanes:

- Runtime semantics: state transitions, checkpoint/resume safety, pause and
  interrupt behavior, timeout priority, event order, and streaming durability.
- Contracts, docs, and conformance: `spec/v0`, `docs`, portable fixtures, and
  cross-SDK behavior expectations.
- Extensibility boundary: provider neutrality, host-owned integrations, public
  API shape, and whether the design leaves room for future SDKs.
- Code quality: strict typing, data validation, immutability, focused tests,
  maintainability, and local style.
- Historical baggage: compatibility shims, deprecated aliases, transitional
  APIs, duplicate paths, or adapter-specific exceptions that should be removed.

Severity:

- `Must-Fix`: correctness bugs, broken resumability, contract/spec mismatch,
  failing tests, data loss risk, or behavior that invalidates the core runtime
  model.
- `Should-Fix`: maintainability, naming, missing narrow tests, unclear docs, or
  non-blocking design inconsistencies.
- `Looks Good`: confirmed invariants, commands run, and residual risks.

Core CR loop:

1. Run baseline validation before review when the tree is runnable.
2. Start the five review lanes as read-only sub-agents unless a lane owns a
   clearly disjoint write scope.
3. Triage every report into credible `Must-Fix`, credible `Should-Fix`,
   accepted residual risk, stale finding, or rejected finding with reason.
4. Fix every credible `Must-Fix` before handoff. Add or update focused tests
   for runtime semantics, checkpoint/resume, limits, streaming, tool scheduling,
   event order, and contract changes.
5. Rerun local validation after fixes, then rerun targeted lanes for changed or
   subtle risk areas.
6. Repeat until there are no credible `Must-Fix` findings. Repeated
   `Should-Fix` findings may remain only when they are documented as accepted
   residual risk with a concrete reason.
7. Close completed sub-agents after consuming reports and include the final
   validation commands in the handoff.

Do not treat sub-agent output as a substitute for local validation. The final
handoff must be based on the repository state, not only on review reports.
