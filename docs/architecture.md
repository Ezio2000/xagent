# Architecture

The repository is organized around a small Python execution kernel plus
independent sibling packages for optional helper layers.

The `kernel` package owns:

- the lightweight execution loop and state machine;
- model protocol interfaces and tool call/spec/output contracts;
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
| `toolkit` | Tool implementation protocols, tool-facing invocation/context helpers, default `ToolRegistry`, JSON Schema validation, and concrete tool invocation glue. | `kernel` |
| `prompting` | Prompt/message construction conveniences such as `user_text(...)`. | `kernel` |
| `modelkit` | Model adapter helper facade that re-exports kernel stream accumulation and capability discovery helpers for adapter packages. | `kernel` |
| `diagnostics` | Public `RunTrace`, trace construction from events, and deterministic replay validation. | `kernel` |
| `harness` | Workspace-level controlled test harness for exercising runtime packages in repeatable scenarios: model drivers, fake runtime ports, tool stubs and registry doubles, message fixtures, event/timeline/trace observation, test scenario helpers, and behavior assertions. | `kernel`, `toolkit`, `prompting`, `diagnostics` |
| `conformance` | Cross-SDK fixture runner and schema validation CLI. | `kernel`, `toolkit`, `prompting`, `diagnostics`, `harness` |

Host applications own:

- HTTP, SSE, WebSocket, or CLI adapters;
- persistence;
- user authentication;
- concrete model clients;
- concrete tools;
- concrete checkpoint stores, approval UIs, queues, sandboxes, provider
  adapters, artifact storage, dashboards, and deployment runtime.

## Extension Design

The runtime uses a small set of deliberate design patterns rather than a broad
framework hierarchy:

- Ports and adapters: `ModelClient`, `ToolRegistryProtocol`, `ApprovalPolicy`,
  `RunStore`, `RunJournal`, and `RuntimeHook` are host-implemented ports. The
  kernel depends on those protocols and value objects, not on provider SDKs,
  concrete stores, UI layers, or tool packs.
- Strategy: tool scheduling is selected through `ToolSchedulerFactory` and
  `ToolSchedulerProtocol`; retry behavior is selected through
  `RuntimeHook.on_model_error`.
- Observer: runtime events and hooks expose lifecycle observation without
  granting hooks ownership of core event envelopes, sequence numbers, or
  checkpoint placement.
- Snapshot/Memento: `RunSnapshot` and `ResumeInput` are the durable state
  boundary. Hosts persist and reload snapshots, but the kernel owns validation
  of resumable state.
- Null Object: runs without host tools use an internal empty registry so the
  loop follows the same validation and error path as a configured registry.

These patterns support the open/closed boundary: hosts add model providers,
tools, stores, approval policy, journals, schedulers, and observers by injecting
implementations. They should not require kernel changes unless the portable v0
contract itself needs new state, events, or runtime semantics.

The v0 runtime intentionally excludes concrete checkpoint stores, approval
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

`harness` is the controlled test environment around the Python runtime
workspace. It assembles deterministic drivers, fake runtime ports, tool stubs,
message fixtures, observation helpers, and behavior assertions so tests can run
runtime packages in repeatable scenarios. It may compose public APIs from
`kernel`, `toolkit`, `prompting`, and `diagnostics`, but it is not a production
extension layer and must not own runtime semantics, JSON Schema validation
rules, diagnostics replay implementation, provider adapters, or conformance
fixture interpretation.

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
