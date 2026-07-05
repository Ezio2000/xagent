# Agent Runtime v0 Model Streaming

Model streaming is an observation protocol, not a durable state protocol.

SDKs may expose a streaming model client in addition to the canonical
`complete` model call. Streaming clients must explicitly advertise streaming
capability. In the Python SDK this is `ModelCapabilities(streaming=True)`.
Streaming methods must return an async iterator directly, usually by
implementing the method as an async generator. They must not require callers to
await a coroutine before receiving the iterator. Streaming clients yield
provider-neutral delta events while the model is producing a response. The
runtime may forward those deltas as `model_delta` events for live rendering.

Durable state is still committed only after a complete `ModelResponse` is
available. SDKs must not append partial assistant messages, partial tool-call
arguments, or partial reasoning to `AgentState`.
`reasoning_delta` events are live progress only. They must not be committed to
durable messages, final response parts, snapshots, or resume input.

If streaming is interrupted before a complete response exists, any observed
`model_delta` events are non-durable UI progress and must not be required for
resume. Timeout or provider-error handling may end the invocation with a
terminal `limit_exceeded` or `failed` checkpoint, but that checkpoint must
preserve the last stable pre-model message history and must not commit a partial
assistant message.
SDKs must not retry a failed streaming model attempt after deltas have been
emitted; hosts may start a new invocation or resume from the last durable
checkpoint if their application policy allows it.

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
`usage_delta` carries the current cumulative usage snapshot for the current
streaming model call. Multiple usage deltas are merged field-by-field; later
non-null token fields replace the same field, and omitted fields do not clear
previously reported fields.

If an SDK exposes a stream-completed event with a `ModelResponse`, that response
provides final completion metadata and fallback content/tool calls when no
matching deltas were emitted. Content and tool-call deltas that were emitted
remain canonical for durable assistant content and tool calls. Usage on the
completed response is merged with prior usage deltas using the same field-level
rules.

Tool execution must wait until the complete streamed `ModelResponse` is
available.
