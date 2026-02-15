# Cortex Usage Guide

This guide focuses on how to use Cortex effectively:

- Choose the right abstraction (`LLM`, `llmfunc`, `Agent`, Agent System)
- Create tools and run agents (sync and async)
- Use memory and usage tracking
- Use streaming safely
- Build multi-agent systems with `CoordinatorSystem`

---

## 0) Installation and API keys

Install:

```bash
pip install cortex
```

You must provide provider API keys via environment variables required by the provider SDKs.

OpenAI:

```bash
export OPENAI_API_KEY=... 
```

Anthropic:

```bash
export ANTHROPIC_API_KEY=...
```

---

## 1) Pick the right abstraction

### 1.1 Use `LLM` when you want raw model calls

Use `LLM` directly if you already manage messages and just want a backend-routed call.

You will usually use `LLM` through `Agent` or `llmfunc` rather than directly.

### 1.2 Use `llmfunc` for “one-shot” intelligent functions

Use `llmfunc` when:

- One prompt + one user input (maybe with retries) solves the task.
- You want optional JSON output validation (`result_shape`).

It returns a function you can call from your own code, or wrap as a tool.

### 1.3 Use `Agent` when you need a loop + tools + memory

Use `Agent` when:

- You need tool calling.
- You expect multi-turn conversations.
- You want tool retries, parallel tool execution, and conversation memory.

### 1.4 Use the Agent System for multi-agent applications

Use the Agent System when:

- You want a coordinator + specialists.
- You want a single `system.async_ask(...)` entrypoint.
- You want context-managed memory/usage/whiteboard.

---

## 2) Models and fallbacks

### 2.1 Choosing a model

Cortex exports model enums:

- `GPTModels` (OpenAI)
- `AnthropicModels` (Anthropic)

Example:

```python
from cortex import LLM, GPTModels
llm = LLM(model=GPTModels.GPT_5_MINI)
```

### 2.2 Set an automatic fallback model

If a model fails at runtime, you can fall back:

```python
from cortex import LLM, GPTModels

LLM.set_backup_backend(GPTModels.GPT_5, GPTModels.GPT_4O)
```

This is implemented in `LLM.call(...)` / `LLM.async_call(...)` by catching exceptions, marking the primary failed, switching to backup, and retrying.

---

## 3) Messages (when you need structured input)

In many cases you can pass a string to `Agent.ask(...)` / `Agent.async_ask(...)`.

If you need structured messages:

```python
from cortex.message import UserMessage, DeveloperMessage

msgs = [
  UserMessage(content="Summarize this"),
  DeveloperMessage(content="Keep it under 80 words")
]
```

Then pass `msgs` to `ask(...)` / `async_ask(...)`.

---

## 4) Tools

### 4.1 Function tools (local execution)

Create a tool by wrapping a Python function with `Tool` (alias for `FunctionTool`).

Tool functions may accept 0–3 positional args:

- `func(tool_input)`
- `func(tool_input, context)`
- `func(tool_input, context, agent)`

Example:

```python
from cortex import Tool

def add(args):
    return {"result": float(args["a"]) + float(args["b"])}

add_tool = Tool(
    name="add",
    func=add,
    description="Add two numbers.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
)
```

### 4.2 Hosted tools (provider-side)

Cortex also defines hosted tool dataclasses such as:

- `WebSearchTool`
- `CodeInterpreterTool`
- `FileSearchTool`
- `MCPTool`

These tools are **encoded and sent to the provider** *only if the selected backend supports that tool type*.

Notes:

- **Local execution**: the core `Agent` only executes `FunctionTool` locally.
- **Backend support varies**: for example, `OpenAIBackend` registers encoders for several hosted tool types, while other backends may ignore or reject them.

---

## 5) Agents

### 5.1 Quickstart (async agent + async tools)

In `async` mode, your function tools must be `async def`.

```python
from cortex import Agent, LLM, GPTModels, Tool

async def get_time(args):
    return {"now": "2026-01-01T00:00:00Z"}

agent = Agent(
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    tools=[
        Tool(
            name="get_time",
            func=get_time,
            description="Get the current time.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
    ],
    sys_prompt="You are helpful. Use tools when needed.",
    mode="async",
)

reply = await agent.async_ask("What time is it?")
```

### 5.2 Quickstart (sync agent + sync tools)

In `sync` mode, your function tools must be sync.

```python
from cortex import Agent, LLM, GPTModels, Tool

def add(args):
    return {"result": float(args["a"]) + float(args["b"])}

agent = Agent(
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    tools=[
        Tool(
            name="add",
            func=add,
            description="Add two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        )
    ],
    sys_prompt="You are a math assistant.",
    mode="sync",
)

print(agent.ask("Add 2 and 3"))
```

### 5.3 Memory

Attach memory so follow-ups include history.

Sync memory:

```python
from cortex import AgentMemoryBank

memory = AgentMemoryBank.bank_for("user_id").get_agent_memory("agent_name", k=5)
```

Async memory (used by Agent System):

```python
from cortex import AsyncAgentMemoryBank

memory_bank = AsyncAgentMemoryBank()
```

### 5.3.1 Conversation Summary

By default, agent memory keeps only the last `k` conversation rounds. Older rounds are discarded, which means important context from earlier in the conversation is lost.

**Conversation Summary** solves this by periodically summarizing evicted messages and prepending the summary as a `SystemMessage` when loading memory. The agent always sees the important context, even after many rounds.

#### Enable with one flag

The simplest way to enable summarization — uses a built-in LLM summarizer (`gpt-5-nano`):

```python
from cortex import AgentMemory

memory = AgentMemory(k=5, enable_summary=True)
```

That's it. Every 3 evictions (default), the memory will call a small LLM to update the running summary. The summary is automatically included when the agent loads its history.

#### Control summarization frequency

By default, summarization runs every 3rd eviction. Adjust with `summarize_every_n`:

```python
# Summarize on every eviction (more up-to-date, more LLM calls)
memory = AgentMemory(k=5, enable_summary=True, summarize_every_n=1)

# Summarize every 5th eviction (fewer calls, slightly staler summary)
memory = AgentMemory(k=5, enable_summary=True, summarize_every_n=5)
```

#### Use a custom LLM for summarization

If you want a different model for the summarizer:

```python
from cortex import LLM, GPTModels, AgentMemory

summary_llm = LLM(model=GPTModels.GPT_4O_MINI, temperature=0.2)
memory = AgentMemory(k=5, enable_summary=True, summary_llm=summary_llm)
```

#### Provide your own summarization function

For full control over what gets retained, supply a custom `summary_fn`. The function receives the current summary and the evicted messages, and returns the new summary:

```python
from cortex import AgentMemory

def my_summarizer(current_summary: str, messages: list) -> str:
    """Keep only user questions in the summary."""
    questions = [m.content for m in messages if hasattr(m, 'content') and '?' in m.content]
    new_info = "; ".join(questions)
    if current_summary:
        return f"{current_summary}\n{new_info}"
    return new_info

memory = AgentMemory(k=5, enable_summary=True, summary_fn=my_summarizer)
```

When `summary_fn` is provided, it takes precedence over the default LLM-based summarizer — no LLM calls are made.

#### Async agents

For async agents, use `AsyncAgentMemory` with the same parameters. If you provide a custom `summary_fn`, it must be an `async` function:

```python
from cortex import AsyncAgentMemory

async def my_async_summarizer(current_summary: str, messages: list) -> str:
    # your async logic here
    return current_summary + "\n" + "; ".join(m.content for m in messages)

memory = AsyncAgentMemory(k=5, enable_summary=True, summary_fn=my_async_summarizer)
```

#### Read or set the summary manually

You can inspect or override the summary at any time:

```python
# Read the current summary
print(memory.get_summary())

# Set it manually (e.g. seed with prior context)
memory.set_summary("User is a Python developer working on a web app.")
```

#### Using with memory banks

Memory banks forward summary parameters when creating agent memories:

```python
from cortex import AgentMemoryBank

bank = AgentMemoryBank.bank_for("user_123")
memory = bank.get_agent_memory(
    "assistant",
    k=5,
    enable_summary=True,
    summarize_every_n=2,
)
```

#### Redis-backed memory with summary

The Redis memory classes persist the summary in Redis (under `{key}:summary`), so it survives process restarts:

```python
from cortex.redis_agent_memory import RedisAgentMemory, RedisAgentMemoryBank

# Direct usage
memory = RedisAgentMemory(
    k=5,
    redis_client=redis_client,
    key="user:123:assistant",
    enable_summary=True,
    summarize_every_n=2,
)

# Via memory bank
bank = RedisAgentMemoryBank(redis_client=redis_client)
memory = bank.get_agent_memory(
    "assistant",
    k=5,
    enable_summary=True,
    summary_fn=my_summarizer,
)
```

Because the summary is stored in Redis, a new `RedisAgentMemory` instance pointing at the same key will automatically pick up the existing summary — useful for serverless or multi-process deployments.

#### Wiring memory to an agent

Pass the memory to your agent as usual:

```python
from cortex import Agent, LLM, GPTModels, AgentMemory

memory = AgentMemory(k=5, enable_summary=True)

agent = Agent(
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    sys_prompt="You are a helpful assistant.",
    memory=memory,
    mode="sync",
)

# The agent will automatically use the summary in its context
agent.ask("My name is Alice and I work at Acme Corp.")
agent.ask("What's the weather today?")
# ... many rounds later, the summary retains "Alice", "Acme Corp", etc.
```

#### Reference: the default summary prompt

The built-in summarizer uses `DEFAULT_SUMMARY_PROMPT`, which you can import for reference or reuse:

```python
from cortex import DEFAULT_SUMMARY_PROMPT
print(DEFAULT_SUMMARY_PROMPT)
```

#### Quick reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_summary` | `bool` | `False` | Turn on periodic summarization |
| `summary_fn` | `Callable` | `None` | Custom summarizer; takes precedence over LLM |
| `summary_llm` | `LLM` | `None` | LLM for default summarizer (auto-creates `gpt-5-nano`) |
| `summarize_every_n` | `int` | `3` | Summarize every N-th eviction |

### 5.4 JSON reply mode

If you set `json_reply=True`, the agent parses the final model output via `json.loads(...)`.

Use this when you want structured outputs, and make sure your system prompt enforces valid JSON.

### 5.5 Streaming

You can stream text deltas from the model:

- `agent.ask(..., streaming=True)` returns an iterator of strings
- `await agent.async_ask(..., streaming=True)` returns an async iterator of strings

**Constraint:** Agent streaming currently requires **no tools**.

---

## 6) `llmfunc`

### 6.1 Basic

```python
from cortex import LLM, GPTModels, llmfunc

llm = LLM(model=GPTModels.GPT_4O_MINI, temperature=0.2)

summarize = llmfunc(
    llm,
    prompt="Summarize the user input in 3 bullet points.",
)

print(summarize("Long text..."))
```

### 6.2 Enforce JSON output with `result_shape`

```python
from cortex import LLM, GPTModels, llmfunc

shape = {
  "type": "object",
  "properties": {
    "label": {"type": "string"},
    "confidence": {"type": "number"},
  },
  "required": ["label", "confidence"],
  "additionalProperties": False,
}

classify = llmfunc(
    LLM(model=GPTModels.GPT_4O_MINI),
    prompt="Classify the sentiment.",
    result_shape=shape,
)

result = classify("I love this")
```

### 6.3 Validate with `check_func`

Use `check_func` when JSON parses but still isn’t acceptable.

```python
from cortex import CheckResult

def check(obj):
    if obj["confidence"] < 0 or obj["confidence"] > 1:
        return CheckResult.fail("confidence must be between 0 and 1")
    return CheckResult.ok(obj)
```

### 6.4 Streaming with `llmfunc`

```python
streaming_func = llmfunc(llm, prompt="Write a short poem.", streaming=True)
for delta in streaming_func("About the ocean"):
    print(delta, end="")
```

**Constraint:** streaming is not compatible with `result_shape` or `check_func`.

---

## 7) Agent System (Coordinator + Workers)

The Agent System is the recommended way to build multi-agent apps.

### 7.1 Minimal coordinator-worker setup

```python
from cortex import (
    LLM, GPTModels,
    CoordinatorAgentBuilder, WorkerAgentBuilder,
    CoordinatorSystem,
    AgentSystemContext, AsyncAgentMemoryBank,
)

memory_bank = AsyncAgentMemoryBank()
context = AgentSystemContext(memory_bank=memory_bank)

math_worker = WorkerAgentBuilder(
    name="Math Expert",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You are a math expert.",
    introduction="Solves math problems",
)

coordinator = CoordinatorAgentBuilder(
    name="Coordinator",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You coordinate workers.",
)

system = CoordinatorSystem(
    coordinator_builder=coordinator,
    workers=[math_worker],
    context=context,
)

reply = await system.async_ask("What is 2 + 2?")
```

### 7.2 Workers with tools

Workers can have their own toolsets via `tools_builder`.

### 7.3 Usage tracking

Provide `usage` in `AgentSystemContext` and it will be passed through:

```python
from cortex.message import AgentUsage

usage = AgentUsage()
context = AgentSystemContext(usage=usage, memory_bank=memory_bank)

await system.async_ask("Hello")

print(usage.format())
```

---

## 8) Parallel tool execution

When the model returns multiple tool calls in one response, the agent can run them concurrently.

Configuration:

- `enable_parallel_tools=True | False`
- `max_parallel_tools=<int> | None`

This works in:

- async mode: `asyncio.gather()`
- sync mode: `ThreadPoolExecutor`

---

## 9) Logging

Cortex provides a global logging configuration object.

```python
from cortex import LoggingConfig, set_default_logging_config

set_default_logging_config(
  LoggingConfig(
    print_system_prompt=True,
    print_messages=True,
    print_usage_report=True,
  )
)
```

See `docs/logging_config.md` and `examples/logging_config_example.py`.

---

## 10) Common gotchas

- Streaming does not support tool calling in `LLM`/`Agent`.
- Agent mode must match tool function type:
  - `mode='sync'` requires sync tool functions
  - `mode='async'` requires async tool functions
- If you use `json_reply=True`, enforce JSON in prompts and avoid extra formatting.

---

## References

- Architecture: `docs/cortex_design.md`
- Logging config: `docs/logging_config.md`
- Agent System: `cortex/agent_system/README.md`
- Examples: `examples/`
