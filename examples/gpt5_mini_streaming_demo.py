#!/usr/bin/env python3
"""Simple streaming demo using GPT-5 mini.

Prerequisites:
- Set your OpenAI API key in the environment: export OPENAI_API_KEY=...

This demo prints text as it streams from the model.
"""

import argparse
import asyncio
import time
from cortex import LLM, GPTModels
from cortex.message import SystemMessage, UserMessage


def main() -> None:
    llm = LLM(model=GPTModels.GPT_5_MINI)

    msg = "You are a concise assistant."
    sys_msg = SystemMessage(content=msg)
    msgs = [UserMessage(content="Write a short limerick about streaming text output.")]

    print(msg)
    print(msgs)

    for delta in llm.call(sys_msg, msgs, streaming=True):
        time.sleep(0.5)
        print(delta, end="", flush=True)

    print()


async def async_main() -> None:
    llm = LLM(model=GPTModels.GPT_5_MINI)

    msg = "You are a concise assistant."
    sys_msg = SystemMessage(content=msg)
    msgs = [UserMessage(content="Write a short limerick about streaming text output.")]

    print(msg)
    print(msgs)

    result = await llm.async_call(sys_msg, msgs, streaming=True)
    async for delta in result:
        await asyncio.sleep(0.2)
        print(delta, end="", flush=True)

    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--async', dest='async_mode', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.async_mode:
        asyncio.run(async_main())
    else:
        main()
