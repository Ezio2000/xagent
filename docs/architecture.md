# Architecture

The repository is organized around a small Python execution kernel plus
independent sibling packages for optional helper layers.

The `kernel` package owns:

- the lightweight execution loop and state machine;
- model and tool protocol interfaces;
- provider-neutral model options, tool choice, response format, usage, and
  streaming delta contracts;
- loop limits, event stream emission, runtime context, hook slots, and
  serializable run snapshots;
- run-control boundaries for pause, interrupt, external wait, resume, and
  conversation insertion;
- extension ports for checkpoint stores, tool approval decisions, durable event
  journals, cooperative tool cancellation, background task references, artifact
  references, and child-run correlation;
- internal trace recording at runtime boundaries so result traces can be
  serialized and consumed by diagnostics without making diagnostics a kernel
  dependency.

Sibling Python packages own the parts that should be independently importable:

| Package | Owns | May Depend On |
| --- | --- | --- |
| `kernel` | Core execution loop, public protocol/value types, scheduler, events, state, snapshots, resume, limits, hooks, ports, stream accumulation, and capability normalization. | No internal runtime package. |
| `toolkit` | Default `ToolRegistry`, JSON Schema validation, and concrete tool invocation glue. | `kernel` |
| `prompting` | Prompt/message construction conveniences such as `user_text(...)`. | `kernel` |
| `modelkit` | Model adapter helper facade that re-exports kernel stream accumulation and capability discovery helpers for adapter packages. | `kernel` |
| `diagnostics` | Public `RunTrace`, trace construction from events, and deterministic replay validation. | `kernel` |
| `harness` | Reusable test harness helpers such as scripted models and event collection. | `kernel` |
| `conformance` | Cross-SDK fixture runner and schema validation CLI. | `kernel`, `toolkit`, `prompting`, `diagnostics` |

Host applications own:

- HTTP, SSE, WebSocket, or CLI adapters;
- persistence;
- user authentication;
- concrete model clients;
- concrete tools;
- concrete checkpoint stores, approval UIs, queues, sandboxes, provider
  adapters, artifact storage, dashboards, and deployment runtime.

The v0.1 runtime intentionally excludes concrete checkpoint stores, approval
UIs, memory implementations, sandboxing, MCP, subagent schedulers, artifact
stores, job queues, and concrete provider adapters. Those are extension
concerns. The core exposes neutral hooks, context, model protocols, tool
registry protocols, pause metadata, state serialization, and small extension
protocols so host applications and sibling packages can add those behaviors
without changing the loop. Durable progress is represented as `RunSnapshot`;
callback transport, user-message policy, concrete storage backends, concrete
journal backends, worker queues, artifact retention, and retention policy stay
outside the core SDK.

JSON Schema validation is not a `kernel` dependency. The kernel calls
`ToolRegistryProtocol.validate_call(...)` before invoking tools; the default
`toolkit.ToolRegistry` implements that protocol with `jsonschema` validation.
Custom registries must enforce equivalent portable validation if they are used
as production registries.

`RunTrace` is a public diagnostics concept, not a core import surface. The
kernel records compact semantic steps internally and exposes `AgentResult.trace`
as an immutable v0 trace payload mapping. `diagnostics` owns construction into
`RunTrace`, trace-from-events construction, replay validation, and public trace
helpers. The kernel must not define trace object helpers or replay/validation
APIs. Trace data carries compact metadata key summaries, not raw host or
provider metadata values. Raw durable metadata is limited to explicit host-owned
fields such as `RuntimeContext.metadata`, pause metadata, and resume metadata.
Model and tool provider metadata is not copied into durable message history or
trace payloads.

Runtime hooks are invoked inside the active run deadline. Async hooks are
awaited directly. Sync hooks in the Python reference SDK run in a worker thread
so they do not block the event loop; if the run times out or is cancelled, the
runtime stops waiting, but Python cannot forcibly terminate the already-running
thread. Sync hooks should therefore be short and idempotent around host-visible
side effects.

`RunController` is safe to call from other host threads in the Python reference
SDK. Pause, interrupt, and insert requests are protected by a lock and wake the
agent loop with `call_soon_threadsafe`. The controller only synchronizes these
run-control requests; it does not make host persistence, model clients, or tool
implementations thread-safe.
