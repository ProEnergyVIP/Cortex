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

### 1.4 Use `WorkflowAgent` when you want explicit step orchestration

Use `WorkflowAgent` when:

- You want explicit steps instead of an open-ended agent loop.
- You need deterministic routing between stages.
- You want retries, fallbacks, or parallel branches in one runtime.
- You want orchestration logic to be part of the runtime itself.

### 1.5 Use the composition helpers for custom multi-agent topologies

Use the composition helpers when:

- You want structured handoffs between multiple runtimes.
- You want a shared surface for `Agent`, `WorkflowAgent`, and custom `run_brief(...)` runtimes.
- You want reusable building blocks instead of a preset topology.
- You want to expose child executors as tools and synthesize their results.

This is the recommended layer for custom multi-agent systems.

### 1.6 Use preset agent systems when the topology already fits

Use a preset system when you want a ready-made multi-agent shape:

- Use `CoordinatorSystem` for a flat coordinator-worker pattern.
- Use `HierarchicalAgentSystem` for gateway → manager → specialist orchestration.

Use these when:

- You want a single `system.async_ask(...)` entrypoint.
- You want context-managed memory, usage tracking, and optional whiteboard support.
- The preset topology already matches your application well.

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

That's it. Every 3 evictions (default), the memory will call a small LLM to update the running summary.

When the agent loads its history, `load_memory()` prepends up to two `SystemMessage` entries before the conversation window:

1. **Summary** — compressed context from earlier rounds (if any).
2. **Eviction buffer** — raw text of recently evicted messages that haven't been summarized yet (if any).

This two-layer approach ensures the LLM **never loses visibility** of evicted information, even between summarization runs. The periodic LLM call merely compresses the buffer into the summary for token efficiency.

#### Control summarization frequency

By default, summarization runs every 3rd eviction. Adjust with `summarize_every_n`:

```python
# Summarize on every eviction (more up-to-date, more LLM calls)
memory = AgentMemory(k=5, enable_summary=True, summarize_every_n=1)

# Summarize every 5th eviction (fewer calls, slightly staler summary)
memory = AgentMemory(k=5, enable_summary=True, summarize_every_n=5)
```

> **Note:** Even with `summarize_every_n=5`, no information is lost between summarization runs — the eviction buffer keeps all evicted messages visible to the LLM until they are compressed into the summary.

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

## 7) Agent System

The agent-system package is intentionally layered. The recommended approach is:

- use `Agent` and `WorkflowAgent` as the runtime primitives
- use the composition helpers in `cortex.agent_system` when you want structured handoffs and easy composition
- use preset systems like `CoordinatorSystem` and `HierarchicalAgentSystem` when they already match your topology

### 7.0 Which layer should I use?

Use the lightest layer that solves your problem.

#### Runtime primitives: `Agent`, `WorkflowAgent`

Use these directly when:

- you have a single runtime
- your control flow is simple
- you do not need structured handoffs between multiple nodes

#### Composition helpers

Use the composition layer when:

- you want to build your own topology
- you want structured handoffs and normalized results
- you want one shared surface for `Agent`, `WorkflowAgent`, and custom runtimes
- you want reusable building blocks instead of a rigid system

This is the recommended layer for custom multi-agent systems.

#### Preset systems

Use a preset when the topology already matches what you need:

- `CoordinatorSystem` for a flat coordinator-worker pattern
- `HierarchicalAgentSystem` for gateway → manager → specialist orchestration

### 7.1 Core composition layer

The composition layer is the reusable foundation for custom agent systems.

Key APIs:

- `TaskDesc`
- `TaskResult`
- `TaskExecutorBuilderBase`
- `TaskExecutorBuilder`
- `TaskTextFactory`
- `TaskMetadataFactory`
- `create_task_desc(...)`
- `create_child_task_desc(...)`
- `resolve_task_executor(...)`
- `execute_task_executor(...)`
- `create_task_tool(...)`
- `coerce_task_result(...)`
- `execute_task_tools(...)`
- `synthesize_task_results(...)`

What each piece does:

- `TaskDesc` is the structured handoff contract
- `TaskResult` is the normalized result contract
- `TaskExecutorBuilderBase` is the neutral shared builder foundation
- `TaskExecutorBuilder` wraps `Agent`, `WorkflowAgent`, or custom runtimes behind one builder surface
- `create_task_tool(...)` exposes an executor as a `Tool`
- `execute_task_tools(...)` fans work out to child tools
- `synthesize_task_results(...)` synthesizes child results back upward

The public composition surface is intentionally minimal and centered on `TaskDesc`, `TaskResult`, `TaskExecutor`, and `TaskExecutorBuilder`.

Implementation modules:

- `cortex/agent_system/task_models.py`
- `cortex/agent_system/task_executor.py`
- `cortex/agent_system/task_executor_builders.py`
- `cortex/agent_system/task_executor_adapters.py`
- `cortex/agent_system/task_executor_orchestration.py`
- `cortex/agent_system/task_composition.py`

The composition layer is designed as **building blocks**, not as a graph framework. You can:

- build your own topology
- reuse just the normalization and synthesis helpers
- use the shared builder base for your own presets
- mix composition helpers with preset systems where it makes sense

Minimal example:

```python
from cortex import (
    AgentSystemContext,
    AsyncAgentMemoryBank,
    GPTModels,
    LLM,
    TaskExecutorBuilder,
    create_task_desc,
    execute_task_executor,
)

context = AgentSystemContext(memory_bank=AsyncAgentMemoryBank())

researcher = TaskExecutorBuilder.create_agent(
    name="Researcher",
    role="research",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt="You investigate the task and return a structured result.",
)

reviewer = TaskExecutorBuilder.create_agent(
    name="Reviewer",
    role="review",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt="You review the task and return a structured result.",
)

lead = TaskExecutorBuilder.create_workflow(
    name="Lead",
    role="lead",
    workflow=make_lead_workflow,
)

desc = create_task_desc(
    from_executor="user",
    to_executor="Lead",
    handoff_kind="user_to_lead",
    original_request="Should we approve this rollout?",
    request_summary="Evaluate whether the rollout should be approved.",
    current_understanding="The lead should coordinate a research and review pass.",
    assigned_task="Coordinate the review and return a final recommendation.",
)

result = await execute_task_executor(
    lead,
    desc=desc,
    context=context,
    installed_tools=[researcher.as_tool(), reviewer.as_tool()],
)
```

If you are designing your own multi-layer system, start here first and only move to a preset if your topology matches it naturally.

### 7.2 Coordinator-worker preset

Use `CoordinatorSystem` when a flat coordinator-worker topology already matches your needs.

This preset is best when:

- one coordinator delegates to workers
- workers behave like specialized assistants
- you want simple orchestration with minimal structure overhead
- you do not need explicit structured handoffs between every hop

This preset is primarily `Agent`-builder based.

### 7.3 Minimal coordinator-worker setup

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

Choose this preset when:

- the coordinator mostly delegates to specialists
- workers do not need to reinterpret messages into deeper structured handoffs
- you want the simplest prebuilt orchestration pattern

### 7.4 Workers with tools

Workers can have their own toolsets via `tools_builder`.

The coordinator preset is primarily `Agent`-builder based. If you need explicit per-node step graphs or structured multi-layer handoffs, prefer either:

- the composition helpers
- or the hierarchical preset

### 7.5 WorkflowAgent as a direct runtime

Use `WorkflowAgent` directly when you want explicit orchestration inside a single runtime without introducing a larger multi-agent topology.

Choose it when you need:

- step-by-step control flow
- retries or fallback logic
- deterministic routing between stages
- parallel branches in one workflow

Once you need structured handoffs between multiple runtimes, move up to:

- the composition helpers for a custom topology
- or `HierarchicalAgentSystem` for the built-in multi-layer preset

### 7.6 Usage tracking

Provide `usage` in `AgentSystemContext` and it will be passed through:

```python
from cortex.message import AgentUsage

usage = AgentUsage()
context = AgentSystemContext(usage=usage, memory_bank=memory_bank)

await system.async_ask("Hello")

print(usage.format())
```

### 7.7 Whiteboard (agent communication)

The whiteboard provides a channel-based messaging system for agents to communicate asynchronously. Enable it to let agents share context, post updates, and coordinate across conversation turns.

**Basic usage:**

```python
from cortex import AgentSystemContext, AsyncAgentMemoryBank

# Enable whiteboard (uses in-memory storage by default)
context = AgentSystemContext.create(
    memory_bank=AsyncAgentMemoryBank(),
    enable_whiteboard=True,
)

# With Redis persistence
from cortex.agent_system.core.whiteboard import RedisStorage
from redis.asyncio import Redis

redis_client = Redis(host='localhost', port=6379)
context = AgentSystemContext.create(
    memory_bank=AsyncAgentMemoryBank(),
    enable_whiteboard=True,
    whiteboard_storage=RedisStorage(redis_client),
)
```

When enabled, agents automatically get these tools:
- `whiteboard_post` - Send messages to channels
- `whiteboard_read` - Retrieve messages from channels
- `whiteboard_subscribe` - Subscribe to channel updates
- `whiteboard_list_channels` - Discover available channels

**Example workflow:**

```python
# Coordinator posts a goal
await context.whiteboard.post(
    sender="Coordinator",
    channel="project:acme-merger",
    content={"type": "goal", "description": "Analyze merger risks"}
)

# Workers read and post updates
messages = await context.whiteboard.read(channel="project:acme-merger")

await context.whiteboard.post(
    sender="RiskAnalyst",
    channel="project:acme-merger",
    content={"type": "finding", "risk": "High debt ratio"},
    thread="financial-analysis"
)
```

**Direct whiteboard access:**

```python
from cortex.agent_system.core.whiteboard import Whiteboard, InMemoryStorage, RedisStorage

# In-memory (development/testing)
wb = Whiteboard()

# Redis-backed (production)
wb = Whiteboard(storage=RedisStorage(redis_client))

# Post and read
msg = await wb.post(
    sender="Agent",
    channel="alerts",
    content={"severity": "high", "message": "System overload"}
)
messages = await wb.read(channel="alerts", limit=50)
```

### 7.8 Hierarchical preset

Use `HierarchicalAgentSystem` when each layer should reinterpret, refine, and synthesize work rather than forwarding it verbatim.

Choose this preset when:

- you want gateway triage
- managers should transform tasks before delegating
- specialists should return structured child results
- synthesis should happen at each level

The hierarchy has four layers:

- user
- gateway
- department managers
- specialist workers

Each handoff is structured with a `DelegationBrief`, and each node returns a `NodeResult`. The system is designed around these rules:

- enrich requests when delegating downward
- synthesize findings when reporting upward
- escalate when confidence is too low
- preserve task and conversation threading
- optionally log handoffs to the whiteboard

Core pieces:

- `GatewayNodeBuilder`
- `DepartmentManagerBuilder`
- `SpecialistNodeBuilder`
- `DepartmentSpec`
- `HierarchicalAgentSystem`

Every node role can be backed by:

- `Agent`
- `WorkflowAgent`
- the default preset runtime

### 7.9 Mental model

- `GatewayNodeBuilder` builds the top-level routing node
- `DepartmentManagerBuilder` builds department supervisors
- `SpecialistNodeBuilder` builds specialist workers
- `DepartmentSpec` defines one department and its children
- `HierarchicalAgentSystem.create(...)` assembles the full tree

### 7.10 Every role can be either `Agent` or `WorkflowAgent`

The public API is runtime-symmetric.

For gateway nodes:

- `GatewayNodeBuilder.create_agent(...)`
- `GatewayNodeBuilder.create_workflow(...)`
- `GatewayNodeBuilder.create_default(...)`

For manager nodes:

- `DepartmentManagerBuilder.create_agent(...)`
- `DepartmentManagerBuilder.create_workflow(...)`
- `DepartmentManagerBuilder.create_default(...)`

For specialist nodes:

- `SpecialistNodeBuilder.create_agent(...)`
- `SpecialistNodeBuilder.create_workflow(...)`

This means you can choose the runtime per role:

- use `Agent` for prompt-and-tool-driven behavior
- use `WorkflowAgent` for explicit step orchestration
- use `create_default(...)` for the built-in routing and synthesis runtime

### 7.11 Minimal hierarchical setup

```python
from cortex import (
    AgentSystemContext,
    AsyncAgentMemoryBank,
    DepartmentManagerBuilder,
    DepartmentSpec,
    GatewayNodeBuilder,
    GPTModels,
    HierarchicalAgentSystem,
    LLM,
    SpecialistNodeBuilder,
    Tool,
    build_gateway_prompt,
    build_manager_prompt,
    build_specialist_prompt,
)

memory_bank = AsyncAgentMemoryBank()
context = AgentSystemContext(memory_bank=memory_bank)

lookup_vendor = Tool(
    name="lookup_vendor",
    func=lookup_vendor_func,
    description="Look up vendor information",
    parameters={
        "type": "object",
        "properties": {"company": {"type": "string"}},
        "required": ["company"],
        "additionalProperties": False,
    },
)

gateway = GatewayNodeBuilder.create_agent(
    name="Gateway",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt=build_gateway_prompt(
        organization_context="You route user requests across business departments."
    ),
)

finance_manager = DepartmentManagerBuilder.create_agent(
    name="Finance Manager",
    department="Finance",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt=build_manager_prompt(
        department_name="Finance",
        department_description="Owns procurement, spend, and vendor onboarding decisions.",
    ),
)

vendor_specialist = SpecialistNodeBuilder.create_agent(
    name="Vendor Risk Specialist",
    specialty="Vendor Risk",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt=build_specialist_prompt(
        specialty_name="Vendor Risk",
        specialty_description="Assesses third-party vendor risk.",
    ),
    tools=[lookup_vendor],
)

finance_department = DepartmentSpec.create(
    name="Finance",
    description="Handles procurement and vendor onboarding reviews.",
    manager=finance_manager,
).add_specialist(vendor_specialist)

system = HierarchicalAgentSystem.create(
    gateway_builder=gateway,
    departments=[finance_department],
    context=context,
)

response = await system.async_ask("Should we approve Acme Power Services for onboarding?")
```

### 7.12 Mixed runtime hierarchy

You can mix runtimes freely in one tree.

Typical examples:

- gateway as `Agent`, manager as `WorkflowAgent`, specialists mixed
- gateway as default runtime, managers as `Agent`
- gateway as `WorkflowAgent`, managers as default runtimes, specialists mixed

Workflow-backed gateway and manager factories can accept:

- `context`
- `installed_tools`
- `child_tools`

That allows workflow nodes to orchestrate their child department or specialist tools directly.

### 7.13 Default low-boilerplate path

If you want built-in orchestration behavior:

```python
gateway = GatewayNodeBuilder.create_default()

finance_manager = DepartmentManagerBuilder.create_default(
    name="Finance Manager",
    department="Finance",
)
```

These defaults provide:

- department routing
- specialist fan-out
- synthesis
- confidence-based escalation

### 7.14 Recommended usage guidance

- use `create_default(...)` when you want the framework to handle orchestration
- use `create_agent(...)` when the node behavior is primarily prompt/tool driven
- use `create_workflow(...)` when the node needs explicit steps, routing, or parallel control
- use `DepartmentSpec.create(...).add_specialists(...)` to build larger hierarchies fluently

### 7.15 Example reference

See:

- `examples/hierarchical_agent_system_example.py`

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
