# Agent Runtime v0 Model Streaming

Model streaming is an observation protocol, not a durable state protocol.

SDKs may expose a streaming model client in addition to the canonical
`complete` model call. Streaming clients must return an async iterator directly,
usually by implementing the method as an async generator. They must not require
callers to await a coroutine before receiving the iterator. Streaming clients
yield provider-neutral delta events while the model is producing a response. The
runtime may forward those deltas as `model_delta` events for live rendering.

Durable state is still committed only after a complete `ModelResponse` is
available. SDKs must not append partial assistant messages, partial tool-call
arguments, or partial reasoning to `AgentState`.

If streaming is interrupted before a complete response exists, any observed
`model_delta` events are non-durable UI progress and must not be required for
resume. Timeout or provider-error handling may end the invocation with a
terminal `limit_exceeded` or `failed` checkpoint, but that checkpoint must
preserve the last stable pre-model message history and must not commit a partial
assistant message.

Known `model_delta` payload shapes:

- `text_delta`: `{ "kind": "text_delta", "index": 0, "text_delta": "...", "part_type": "text" }`
- `tool_call_delta`: `{ "kind": "tool_call_delta", "index": 0, "id": "...", "name": "...", "mode": "accept", "arguments_delta": "..." }`
- `reasoning_delta`: `{ "kind": "reasoning_delta", "index": 0, "text_delta": "..." }`
- `usage_delta`: `{ "kind": "usage_delta", "usage": { "input_tokens": 1 } }`

`index` is zero-based within the current streamed response. `id`, `name`, and
`mode`, and `arguments_delta` may arrive across multiple `tool_call_delta`
events for the same index. If `mode` is omitted, the accumulated tool call uses
`execute`. Tool-call arguments are accumulated as JSON object bytes and become
durable only in the final `ModelResponse`.

Tool execution must wait until the complete streamed `ModelResponse` is
available.
