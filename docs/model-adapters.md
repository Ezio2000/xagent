# Model Adapters

Concrete adapters live in `jharness.models`. They implement the provider-neutral
`jharness.kernel.Model` protocol by translating kernel requests, complete responses,
and live deltas to and from provider wire APIs.

Provider code does not define runtime semantics. When an endpoint feature cannot be
represented by the kernel protocol, its adapter rejects the feature or keeps the detail
inside provider-local profile and metadata fields. Durable state, events, tool calls,
and stream semantics change only through the owning kernel contracts.

## Package Boundary

`jharness.models` may depend on `jharness.kernel` and `httpx`. Kernel, toolkit, and
ready-to-use tools never depend on models. Applications construct a concrete model
and inject it into `Runtime`:

```python
from jharness.kernel import Runtime
from jharness.models.openai import OpenAIChatCompletionsModel

model = OpenAIChatCompletionsModel(
    base_url="https://api.example.com/v1",
    api_key=api_key,
    model=model_name,
)
runtime = Runtime(model=model)
```

The top-level `jharness.models` package does not flatten provider APIs. Public
adapters are imported from their concrete protocol namespace.

## HTTP Client Ownership

Adapters accept an optional host-owned `httpx.AsyncClient`. A high-throughput host can
inject one client for a compatible event-loop lifecycle, reuse its connection pool, and
close it during host shutdown. When no client is injected, each request owns a
short-lived client. Runtime never closes a host-owned client.

The default transport policy bounds connect time at 10 seconds and every other HTTP
phase at 60 seconds. A caller may replace that policy through the adapter `timeout`
option; an explicit `timeout=None` disables only the HTTP phase timeout. The remaining
`RunContext.deadline` still clamps every request and the runtime's absolute deadline
still cancels the invocation. Omitting `timeout` is different from explicitly passing
`None`.

Streaming parsers accept only CR, LF, or CRLF as SSE line terminators and decode UTF-8
strictly. A line is limited to 262,144 bytes and one event, including comments and all
`data` fields, is limited to 1,048,576 bytes by default. Hosts may configure smaller
positive limits. An over-limit or malformed stream raises the provider's structured
model error and closes the response.

Shared transport code owns only common mechanics:

- client ownership and response lifetime;
- POST and SSE setup;
- success checks and normalized transport context;
- cancellation cleanup;
- the common nested error envelope.

Each provider still owns its URL, headers, request-id locations, retryable statuses,
stream termination rule, and request/response codecs. Provider implementations compose
shared helpers; they do not inherit a behavioral base class.

## Model Invocation and Streaming

Every adapter exposes one `Model.invoke` operation. A non-streaming call decodes one
response into a complete `ModelResponse`. A streaming call incrementally decodes
provider chunks, awaits the ordered delta sink, and returns the same complete response
type when the stream terminates.

Provider transport, payload, iterator, and stream-protocol failures are normalized to
the adapter's `ModelError`. A failure raised by the host-owned delta sink is outside the
provider boundary: it propagates unchanged after the response is closed. This keeps
host backpressure and observation failures distinguishable from provider failures.

The adapter owns the only stream accumulator. It closes response bodies and settles
emitter work before returning, failing, or propagating cancellation. Runtime events
record model start and finish; provider deltas contain only incremental content,
reasoning, tool-call, or usage data.

Historical kernel tool calls contain id, name, and arguments. An adapter encodes that
durable history without consulting the current tool catalog, because a later
invocation may no longer advertise the historical tool name.

Adapters do not own retry loops. Hosts express retry and fallback by decorating the
provider-neutral model boundary so attempt counts, budgets, cancellation, and
observability remain explicit.

Normalized HTTP errors preserve a raw `Retry-After` header in
`ModelError.metadata["retry_after"]` when present; decorators may parse either delay
seconds or an HTTP date. A semantic error delivered inside an HTTP-successful stream
keeps the error payload's code and status instead of being rewritten as status 200.
Provider overload codes, including Anthropic `overloaded_error`, are retryable even
when the enclosing stream used a successful HTTP status.

## OpenAI Chat Completions

The OpenAI-compatible adapter maps the kernel protocol to `POST /chat/completions`:

```python
import os

from jharness.models.openai import (
    OpenAIChatCompletionsModel,
    OpenAIChatCompletionsProfile,
)

model = OpenAIChatCompletionsModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
    profile=OpenAIChatCompletionsProfile(),
)
```

The profile declares endpoint capabilities rather than inferring them from a model
name. Its supported surface includes, when enabled by the selected profile:

- text and multimodal message inputs;
- tool specifications, calls, and `ToolChoice`;
- JSON object and JSON Schema response formats;
- complete responses and streaming content, reasoning, tool-call, and usage deltas;
- provider-local request extensions that cannot overwrite reserved fields;
- provider errors normalized to `ModelError`.

`supports_parallel_tool_calls` states that the model can return multiple calls.
`supports_parallel_tool_call_control` separately states that the adapter may send the
`parallel_tool_calls` request field. Streamed call fragments are accumulated by their
provider call index into one kernel tool call.

The adapter does not implement the Responses API, the legacy text Completions API,
provider-hosted tools, provider-managed conversation state, or provider file uploads.

## Anthropic Messages

The Anthropic adapter maps the kernel protocol to `POST /v1/messages`:

```python
import os

from jharness.models.anthropic import AnthropicModel, AnthropicProfile

model = AnthropicModel(
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model=os.environ["ANTHROPIC_MODEL"],
    profile=AnthropicProfile(),
)
```

This namespace has its own codec, stream decoder, profile, and error type because the
wire protocol uses top-level system instructions, content blocks,
`tool_use`/`tool_result`, `output_config.format`, and named SSE events.

Its supported surface includes, when profile-enabled:

- text, image, and document inputs;
- conversion of kernel system messages;
- tool specifications, historical calls, results, and tool choice;
- JSON object and JSON Schema response formats;
- complete and streaming content, reasoning, thinking-block, tool-call, and usage
  responses;
- provider-local request and header extensions;
- provider errors normalized to `ModelError`.

Message Batches, file upload, token counting, provider-hosted tools, agents, and
provider-managed sessions are outside the adapter. Mid-conversation system messages
are disabled unless a profile explicitly enables them.

## DeepSeek Profiles

DeepSeek factories configure one of the concrete wire-protocol adapters. The factory
name selects the protocol; thinking mode and effort remain explicit options:

```python
from jharness.models.anthropic import AnthropicModel
from jharness.models.deepseek import (
    deepseek_anthropic_profile,
    deepseek_openai_chat_profile,
)
from jharness.models.openai import OpenAIChatCompletionsModel

chat_model = OpenAIChatCompletionsModel(
    base_url="https://api.deepseek.com",
    api_key=api_key,
    model=model_name,
    profile=deepseek_openai_chat_profile(thinking=False),
)

thinking_model = AnthropicModel(
    base_url="https://api.deepseek.com/anthropic",
    api_key=api_key,
    model=model_name,
    profile=deepseek_anthropic_profile(thinking=True, effort="max"),
)
```

OpenAI-format thinking mode does not advertise tool use when the endpoint requires
reasoning content to be returned in later history but that content is only a live
kernel delta. The Anthropic-format profile can preserve thinking blocks as durable
content and therefore supports tools in both modes. Endpoint-specific capability flags
remain conservative; runtime execution concurrency is always bounded independently by
`RunLimits.max_tool_concurrency`.

## Codec Boundary

Wire-format branching belongs in codecs rather than the runtime-facing client flow:

- message and content conversion belongs in a message codec;
- tool schema and tool-call conversion belongs in a tool codec;
- request and complete-response assembly belongs in a model codec;
- SSE and chunk parsing belongs in a stream codec;
- HTTP and provider error normalization belongs at the configured transport boundary.

This keeps kernel provider-neutral and puts endpoint differences in explicit profiles
rather than scattered conditionals.
