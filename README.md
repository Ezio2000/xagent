# agent-runtime

Model-neutral agent loop runtime for building reusable agent SDKs.

The project is organized for multiple language SDKs. The Python SDK is the
reference implementation today; TypeScript and Go directories are reserved for
future SDKs.

## What This Core Provides

`agent-runtime` focuses on the reusable runtime layer:

- Agent loop state machine: `planning`, `executing_tools`, `completed`,
  `failed`, and `limit_exceeded`.
- Provider-neutral model protocol: messages, tools, model options, tool choice,
  response format, usage, capabilities, structured provider errors, and
  streaming deltas.
- Event stream: `run_started`, `model_started`, `model_delta`,
  `model_completed`, `tool_started`, `tool_completed`, `state_changed`,
  `checkpoint`, `final`, `error`, and `run_completed`.
- Durable checkpoints and snapshots for host-owned persistence and resume.
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
not be required for resume.

Tool execution is serial by default. If `LoopLimits.max_parallel_tool_calls > 1`,
the scheduler may run a consecutive batch concurrently only when every tool in
that batch declares:

- `parallel_safe: true`
- `read_only: true`
- `idempotent: true`

Parallel batches commit tool observations atomically in model-provided order.
If a timeout interrupts a parallel batch, the next checkpoint remains at the last
fully committed boundary; observed but uncommitted idempotent calls may be rerun
on resume.

## Quick Start

```bash
cd sdks/python
uv sync
uv run python ../../examples/python/basic_tool_loop.py
```

Minimal model:

```python
import asyncio

from agent_runtime import AgentLoop, Message, ModelRequest, ModelResponse
from agent_runtime import RuntimeContext


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

For tool usage, streaming, multimodal messages, and snapshots, see
`sdks/python/README.md` and `examples/python/basic_tool_loop.py`.

## Model Protocol

Model adapters implement:

```python
async def complete(request, context) -> ModelResponse: ...
```

Streaming adapters may additionally implement:

```python
def stream(request, context) -> AsyncIterator[ModelStreamEvent]: ...
```

`stream()` must return an async iterator directly, usually from an async
generator. The runtime forwards stream progress as `model_delta` events and
commits `AgentState` only after a complete `ModelResponse` exists.

Provider adapters should translate `ModelOptions`, `ToolChoice`,
`ResponseFormat`, multimodal `ContentPart` values, and `ModelCapabilities` into
their concrete provider API. Adapter-specific fields belong in `extra` or
`metadata` rather than in runtime control flow.

## Project Structure

- `spec/v0`: cross-language contracts for state, messages, events, tools,
  limits, snapshots, model requests, model responses, and streaming.
- `conformance/cases`: shared behavior cases every SDK should pass.
- `docs`: design notes for architecture, event streams, state machine, model
  protocol, and tool protocol.
- `sdks/python`: reference SDK implementation managed with `uv`.
- `sdks/typescript`: reserved TypeScript SDK location.
- `sdks/go`: reserved Go SDK location.
- `examples/python`: small runnable examples.

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
uv run python ../../examples/python/basic_tool_loop.py
```

When JSON spec or conformance files change, also parse them:

```bash
cd sdks/python
uv run python - <<'PY'
import json
from pathlib import Path

for root in ["../../spec/v0", "../../conformance/cases"]:
    for path in sorted(Path(root).glob("*.json")):
        json.loads(path.read_text())
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

Severity:

- `Must-Fix`: correctness bugs, broken resumability, contract/spec mismatch,
  failing tests, data loss risk, or behavior that invalidates the core runtime
  model.
- `Should-Fix`: maintainability, naming, missing narrow tests, unclear docs, or
  non-blocking design inconsistencies.
- `Looks Good`: confirmed invariants, commands run, and residual risks.

Do not treat sub-agent output as a substitute for local validation. Fix every
credible `Must-Fix` before handoff, rerun validation, request narrow re-review
when the risk is subtle, and close completed sub-agents after consuming reports.
