"""Run one invocation against an Anthropic Messages endpoint.

Required environment variables:

  ANTHROPIC_BASE_URL   Example: https://api.anthropic.com
  ANTHROPIC_API_KEY
  ANTHROPIC_MODEL

Optional environment variables:

  ANTHROPIC_PROMPT
"""

from __future__ import annotations

import asyncio
import os

from jharness.kernel import Completed, Message, Runtime
from jharness.models.anthropic import AnthropicModel, AnthropicProfile


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


async def main() -> None:
    model = AnthropicModel(
        base_url=env_required("ANTHROPIC_BASE_URL"),
        api_key=env_required("ANTHROPIC_API_KEY"),
        model=env_required("ANTHROPIC_MODEL"),
        profile=AnthropicProfile(name=os.environ.get("ANTHROPIC_PROFILE_NAME", "anthropic")),
    )
    prompt = os.environ.get("ANTHROPIC_PROMPT", "Say hello in one short sentence.")
    checkpoint = await Runtime(model=model).start((Message.user(prompt),)).result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")
    print("".join(part.text or "" for part in state.parts))


if __name__ == "__main__":
    asyncio.run(main())
