# Architecture

`agent-runtime` is a small execution kernel for tool-using agents.

The core SDK owns:

- a lightweight state machine;
- model/tool protocol interfaces;
- provider-neutral model options, tool choice, response format, usage, and
  streaming delta contracts;
- loop limits;
- event stream emission;
- runtime context, hook slots, and serializable run snapshots;
- core run-control boundaries for pause, interrupt, external wait, and
  conversation insertion;
- strict resume input validation, compact run traces, and deterministic replay
  validation.

Host applications own:

- HTTP, SSE, WebSocket, or CLI adapters;
- persistence;
- user authentication;
- concrete model clients;
- concrete tools.

The v0.1 runtime intentionally excludes concrete checkpoint stores, approval
UIs, memory implementations, sandboxing, MCP, subagents, and concrete provider
adapters. Those are extension concerns. The core exposes neutral hooks, context,
model protocols, pause metadata, and state serialization so host applications
can add those behaviors without changing the loop. Durable progress is
represented as `RunSnapshot`; storage, callback transport, user-message policy,
trace persistence, and retention policy stay outside the core SDK. `RunTrace`
records semantic runtime steps for replay and conformance, but it is not a log
store, queue, callback transport, or monitoring system. It carries compact
metadata key summaries, not raw host or provider metadata values.
Raw durable metadata is limited to explicit host-owned fields such as
`RuntimeContext.metadata`, pause metadata, and resume metadata. Model and tool
provider metadata is not copied into durable message history or trace payloads.
