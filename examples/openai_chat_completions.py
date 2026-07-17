"""Run one invocation against an OpenAI Chat Completions endpoint.

Required environment variables:

  OPENAI_CHAT_COMPLETIONS_BASE_URL   Example: https://api.example.com/v1
  OPENAI_CHAT_COMPLETIONS_API_KEY
  OPENAI_CHAT_COMPLETIONS_MODEL

Optional environment variables:

  OPENAI_CHAT_COMPLETIONS_PROMPT
"""

from __future__ import annotations

import asyncio
import os

from jharness.kernel import Completed, Message, Runtime
from jharness.models.openai import OpenAIChatCompletionsModel, OpenAIChatCompletionsProfile


def env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


async def main() -> None:
    model = OpenAIChatCompletionsModel(
        base_url=env_required("OPENAI_CHAT_COMPLETIONS_BASE_URL"),
        api_key=env_required("OPENAI_CHAT_COMPLETIONS_API_KEY"),
        model=env_required("OPENAI_CHAT_COMPLETIONS_MODEL"),
        profile=OpenAIChatCompletionsProfile(
            name=os.environ.get("OPENAI_CHAT_COMPLETIONS_PROFILE_NAME", "openai-chat-completions"),
            supports_image_input=False,
            supports_json_schema=False,
        ),
    )
    prompt = os.environ.get("OPENAI_CHAT_COMPLETIONS_PROMPT", "Say hello in one short sentence.")
    checkpoint = await Runtime(model=model).start((Message.user(prompt),)).result()
    state = checkpoint.snapshot.state
    if not isinstance(state, Completed):
        raise RuntimeError(f"run stopped with {checkpoint.snapshot.status}")
    print("".join(part.text or "" for part in state.parts))


if __name__ == "__main__":
    asyncio.run(main())
