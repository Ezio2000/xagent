# agent-runtime Python SDK

The Python SDK implements the v0.1 agent loop runtime.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Minimal Usage

```python
from agent_runtime import (
    AgentLoop,
    ContentPart,
    LoopLimits,
    Message,
    ModelOptions,
    ModelResponse,
    ResponseFormat,
    RuntimeContext,
    RuntimeHook,
    RunSnapshot,
    ToolChoice,
    ToolResult,
    ToolSpec,
)

agent = AgentLoop(model=model_client, tools=[tool])
result = await agent.run(
    [Message.user_text("Use the tool if needed")],
    context=RuntimeContext(run_id="run-1", metadata={"tenant": "acme"}),
)
```

`AgentLoop` accepts provider-neutral model controls. Provider adapters translate
these values to their concrete API shape:

```python
agent = AgentLoop(
    model=model_client,
    tools=[tool],
    model_options=ModelOptions(model="provider-model", temperature=0.2),
    tool_choice=ToolChoice(mode="auto", allow_parallel_tool_calls=True),
    response_format=ResponseFormat(type="json_object"),
)
```

Model adapters receive a `ModelRequest` plus `RuntimeContext`:

```python
async def complete(request, context):
    return ModelResponse.text(request.messages[-1].text)
```

Tools expose a neutral `ToolSpec` and receive the same context:

```python
class EchoTool:
    spec = ToolSpec(
        name="echo",
        description="Return input text.",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )

    async def execute(self, arguments, context):
        return ToolResult.text(arguments["text"])
```

Tool execution is serial by default. Enable simple parallel scheduling with a
runtime limit and explicit tool annotations:

```python
search_tool.spec = ToolSpec(
    name="search",
    description="Search docs",
    input_schema={"type": "object"},
    annotations={
        "parallel_safe": True,
        "read_only": True,
        "idempotent": True,
    },
)

agent = AgentLoop(
    model=model_client,
    tools=[search_tool],
    limits=LoopLimits(max_parallel_tool_calls=4),
)
```

The model decides which tools to request. The runtime decides whether those
calls may run concurrently. Tool observations are committed to message history
in the model-provided order even when executions finish out of order. Parallel
batches checkpoint only after the whole batch is committed; serial tools still
checkpoint one result at a time.
If `stop_on_tool_error=True`, tool execution is serial to preserve fail-fast
semantics.

Hooks subclass `RuntimeHook`. They can observe or rewrite model/tool boundaries.
`on_event` can append custom runtime events:

```python
class ProgressHook(RuntimeHook):
    def on_event(self, event, context, emitter):
        if event.type == "model_started":
            emitter.emit("custom_progress", {"phase": "model"})
```

If a model adapter implements `stream(request, context)`, callers can enable
live model deltas. The method must return an async iterator directly, usually
because it is an async generator.

```python
async for event in agent.run_events(messages, stream=True):
    if event.type == "model_delta":
        render(event.data)
    if event.type == "checkpoint":
        save(event.data)
```

`model_delta` is for live rendering only. Durable resume state is still carried
only by `checkpoint` events after the complete model response is available.

`RunSnapshot.to_dict()` is the durable checkpoint boundary. It contains
`AgentState` plus `RuntimeContext`; context timestamps are wall-clock epoch
seconds so snapshots can cross process restarts. `checkpoint` events carry the
same payload after model commits, durable tool commits, and state transitions.
Storage is host-owned; resume with `run_snapshot` or `run_snapshot_events`.

```python
snapshot = RunSnapshot.from_dict(saved_payload)
result = await agent.run_snapshot(snapshot)
```

## Multimodal Messages

The core message protocol uses content parts for every message. Text is also a
content part; there is no separate legacy `content` field.

```python
message = Message.user(
    [
        ContentPart.text_part("Analyze this image"),
        ContentPart.image_uri(
            "https://example.com/car.png",
            media_type="image/png",
            name="car.png",
        ),
    ]
)
```

The core runtime ships helpers for `text`, `image`, and `file` parts. The
underlying part `type` is open so provider adapters can carry additional
multimodal blocks such as audio, video, citations, or reasoning references.
