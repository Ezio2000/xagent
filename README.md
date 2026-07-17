# JHarness

JHarness is a provider-neutral Python runtime for durable model and tool execution.
Kernel, toolkit, model adapters, ready-to-use tools, portable contracts, conformance,
tests, and release automation are developed in one repository and released with one
coordinated version.

JHarness requires Python 3.11 or newer and publishes four distributions.

## Install

Install only the component you need:

```bash
uv add jharness-kernel
uv add jharness-toolkit
uv add jharness-models
uv add jharness-tools
```

The non-kernel distributions install the exact matching kernel automatically. Install
the complete product with:

```bash
uv add jharness-kernel jharness-toolkit jharness-models jharness-tools
```

| Distribution | Python import | Internal dependency |
| --- | --- | --- |
| `jharness-kernel` | `jharness.kernel` | None |
| `jharness-toolkit` | `jharness.toolkit` | `jharness-kernel` |
| `jharness-models` | `jharness.models` | `jharness-kernel` |
| `jharness-tools` | `jharness.tools` | `jharness-kernel` |

## Quick Start

Configure a model adapter, inject it into `Runtime`, and execute one invocation:

```python
import asyncio
import os

from jharness.kernel import Completed, Message, Runtime
from jharness.models.openai import OpenAIChatCompletionsModel


async def main() -> None:
    model = OpenAIChatCompletionsModel(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.environ["OPENAI_MODEL"],
    )
    checkpoint = await Runtime(model=model).start(
        (Message.user("Say hello in one short sentence."),)
    ).result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")
    print("".join(part.text or "" for part in state.parts))


asyncio.run(main())
```

Model-specific entry points are explicit:

```python
from jharness.models.anthropic import AnthropicModel, AnthropicProfile
from jharness.models.deepseek import deepseek_anthropic_profile, deepseek_openai_chat_profile
from jharness.models.openai import (
    OpenAIChatCompletionsModel,
    OpenAIChatCompletionsProfile,
)
```

Ready-to-use tools implement kernel contracts and are composed externally with the
toolkit:

```python
from pathlib import Path

from jharness.kernel import Runtime
from jharness.toolkit import ToolRegistry
from jharness.tools import GlobTool, GrepTool, ReadTool

root = Path.cwd()
registry = ToolRegistry((ReadTool(root), GlobTool(root), GrepTool(root)))
runtime = Runtime(model=model, tools=registry)
```

`EditTool`, `WriteTool`, and `BashTool` expose destructive capabilities. Hosts must
apply approval, least-privilege credentials, and operating-system isolation. Bash
requires a `bash` executable on `PATH`; child-agent tools require a host-owned
`AgentBackend`.

## Durable State

Every durable boundary produces one immutable `Checkpoint`. Explicit codecs keep
persisted JSON independent from external model wire formats and Python object layout:

```python
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint

payload = encode_checkpoint(checkpoint)
restored = decode_checkpoint(payload)
```

The persistence family is documented in [`contracts/v0`](contracts/v0/README.md), and
[`conformance/cases`](conformance/cases/) verifies the same runtime behavior.

## Repository Layout

| Area | Path |
| --- | --- |
| Installable projects | [`packages`](packages/) |
| Tests | [`tests`](tests/) |
| Examples | [`examples`](examples/) |
| Contracts | [`contracts/v0`](contracts/v0/README.md) |
| Conformance | [`conformance`](conformance/README.md) |
| Architecture | [`docs/architecture.md`](docs/architecture.md) |
| Package boundaries | [`docs/python-package-boundaries.md`](docs/python-package-boundaries.md) |
| Model adapters | [`docs/model-adapters.md`](docs/model-adapters.md) |
| Release process | [`docs/releasing.md`](docs/releasing.md) |

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q -p no:cacheprovider
uv run python scripts/validate_spec.py
uv run python -m conformance.cli conformance/cases --spec-dir contracts/v0 --quiet
uv run python benchmarks/runtime_smoke.py
uv build --all-packages --out-dir dist
uv run python scripts/verify_distribution.py dist
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete change requirements.
