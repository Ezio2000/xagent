# Model Protocol

The runtime talks to model adapters through a provider-neutral `ModelClient`.
Concrete provider clients live outside the core SDK.

`ModelRequest` contains conversation messages, tool specs, and standardized
call controls:

- `options`: model name, temperature, top-p, output token limit, stop sequences,
  and seed.
- `tool_choice`: auto, none, required, or a specific tool name. Its
  `allow_parallel_tool_calls` flag controls whether the model may return
  multiple tool calls in one response; runtime execution concurrency is still
  governed by `LoopLimits.max_parallel_tool_calls`.
- `response_format`: text, JSON object, or strict JSON schema.

`ModelResponse` contains final content parts, tool calls, finish reason, usage,
model id, response id, and provider metadata. Common finish reasons include
`end_turn`, `tool_calls`, `max_tokens`, `stop_sequence`, `refusal`,
`content_filter`, and `error`; the field remains an open string.

Model adapters may expose capabilities through a `capabilities` value or method.
The core recognizes streaming, tools, tool choice, parallel tool calls,
multimodal input/output, structured output, JSON mode, and usage reporting.

Streaming adapters may additionally expose `stream(request, context)`. The
method must return an async iterator directly, usually from an async generator;
it must not return a coroutine that callers must await to obtain the iterator.
Stream deltas are emitted as `model_delta` events for live rendering only.
Durable `AgentState` is committed after the complete `ModelResponse` is
available.
