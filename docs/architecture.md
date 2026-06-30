# Architecture

`agent-runtime` is a small execution kernel for tool-using agents.

The core SDK owns:

- a lightweight state machine;
- model/tool protocol interfaces;
- provider-neutral model options, tool choice, response format, usage, and
  streaming delta contracts;
- loop limits;
- event stream emission;
- runtime context, hook slots, and serializable run snapshots.

Host applications own:

- HTTP, SSE, WebSocket, or CLI adapters;
- persistence;
- user authentication;
- concrete model clients;
- concrete tools.

The v0.1 runtime intentionally excludes concrete checkpoint stores, approval
UIs, memory implementations, sandboxing, MCP, subagents, and concrete provider
adapters. Those are extension concerns. The core exposes neutral hooks, context,
model protocols, and state serialization so host applications can add those
behaviors without changing the loop. Durable progress is represented as
`RunSnapshot`; storage and retention policy stay outside the core SDK.
