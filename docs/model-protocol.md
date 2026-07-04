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

`ToolChoice` and `ResponseFormat` names are provider-neutral runtime semantics,
not bindings to a specific model API. Adapters translate those intentions into
provider-specific request fields or reject unsupported combinations before
calling the provider.

`ModelResponse` contains final content parts, tool calls, finish reason, usage,
model id, response id, and provider metadata. Provider metadata is visible to
hooks and adapters during the current invocation, but it is not copied into
durable assistant messages, checkpoints, or trace payloads. Common finish
reasons include `end_turn`, `tool_calls`, `max_tokens`, `stop_sequence`,
`refusal`, `content_filter`, and `error`; the field remains an open string.
Runtime usage accounting accumulates the standard token fields reported in
`ModelResponse.usage` into `AgentState.total_usage`. A response that omits usage
or omits an individual usage field does not clear previously accumulated values;
only fields explicitly reported by the response are added to their cumulative
field.

Each `ToolCall` includes an open non-empty `mode` string. Core runtimes
recognize `execute`, which waits for the tool's final observation before
continuing, and `accept`, which asks the tool to accept external work and
immediately commits a `ToolAcceptance` or `ToolRejection` result. Provider
adapters may expose model syntax such as `accept(web_search(...))`, but the
normalized core shape remains the original tool name with `mode: "accept"`.

Python adapters may raise `ModelProviderError(ModelErrorInfo(...))` for
structured provider failures. `ModelErrorInfo` is runtime exception detail for
the current SDK invocation; checkpoint state records the portable error message,
not provider metadata or request objects. The portable structured error shape is
specified in `spec/v0/model-error.schema.json`. `ModelErrorInfo.retryable` is
advisory provider metadata. The runtime retries only when host code returns
`ModelErrorDecision(retry=True)` from `RuntimeHook.on_model_error`, the call was
not streaming, and `LoopLimits.max_model_retries` still permits another attempt.
Each failed attempt emits `model_error`; retry opens a fresh `model_started`
attempt in the same planning iteration.
Streaming model failures are not retried because emitted deltas may already
have reached live consumers. If a hook returns `ModelErrorDecision(retry=True)`
for a streaming failure, the runtime still emits `model_error` with
`data.retry == false`.
Conformance fixtures may set `retry_model_errors: true`; that flag is only a
runner instruction to install a hook that returns `retry=error.retryable`, not a
separate runtime retry rule.

Model adapters may expose capabilities through a `capabilities` value or method.
The core recognizes streaming, tools, tool choice, parallel tool calls,
multimodal input/output, structured output, JSON mode, and usage reporting.

Streaming adapters may additionally expose `stream(request, context)` and must
advertise `ModelCapabilities(streaming=True)`. The method must return an async
iterator directly, usually from an async generator; it must not return a
coroutine that callers must await to obtain the iterator. If streaming is not
advertised, `stream=True` callers use the normal `complete()` path.
Stream deltas are emitted as `model_delta` events for live rendering only.
Durable `AgentState` is committed after the complete `ModelResponse` is
available. Reasoning deltas are included in this live-only stream surface; they
are not appended to message history, checkpoints, or final response content.

External inputs that arrive while a model call is in flight use run-control
conversation insertion. The runtime cancels the in-flight model call, appends an
`external` message, checkpoints, and starts planning again. This insertion
mechanism is independent of tool calls and does not require the inserted input
to originate from a tool.
