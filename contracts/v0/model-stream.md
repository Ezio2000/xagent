# Kernel v0 Model Streaming

Model streaming is live observation, not durable state. There is one model
operation:

```text
Model.invoke(request, context, *, stream, emit_delta) -> ModelResponse
```

When `stream=false`, the model returns a complete response and does not call the
sink. When `stream=true`, `emit_delta` may receive only four provider-neutral
delta variants:

- `content`
- `tool_call`
- `reasoning`
- `usage`

There are no started or completed stream items. `model_started` and
`model_finished` remain invocation observation events around `Model.invoke`;
they are not values in the model stream.

The provider adapter owns stream assembly and always returns one complete
`ModelResponse`. Tool-call deltas accumulate id, name, and JSON arguments by
zero-based call index. Usage deltas merge field by field; an omitted value does
not clear a value already reported. There is no tool invocation mode.

Only the returned response can produce a `model_turn` checkpoint. Partial text,
reasoning, calls, and usage never enter snapshot history or metrics. Kernel does
not run a second response accumulator.

Adapters await each `emit_delta` call in stream order. Closing or cancelling the
invocation closes the provider stream before control returns. Pause,
conversation insertion, provider failure, iterator failure, or deadline before
return preserves the last committed checkpoint and discards partial deltas.

Kernel does not retry a partially observed stream. Retry decorators must not
expose deltas from a failed attempt before retrying it.
