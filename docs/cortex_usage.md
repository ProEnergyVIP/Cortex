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

### 1.4 Use `WorkflowAgent` when you want explicit workflow orchestration

Use `WorkflowAgent` when:

- You want explicit nodes instead of an open-ended agent loop.
- You need deterministic routing between stages.
- You want retries, fallbacks, or parallel branches in one runtime.
- You want orchestration logic to be part of the runtime itself.
- You want explicit shared state and inspectable graph execution.

### 1.5 Use preset agent systems when the topology already fits

Use the preset system when you want a ready-made multi-agent shape:

- Use `CoordinatorSystem` for a flat coordinator-worker pattern.

Use these when:

- You want a single `system.async_ask(...)` entrypoint.
- You want context-managed memory, usage tracking, and optional whiteboard support.
- The coordinator-worker preset topology already matches your application well.

---

### 1.6 Function-first workflow API

The preferred workflow API is function-first:

- `workflow(...)`
- `node(...)` (alias of `function_node(...)`)
- `function_node(...)`
- `router_node(...)`
- `parallel_node(...)`
- `llm_node(...)`
- `edge(...)`

Node functions run with this contract:

- `func(data, context)`
- optional memory-aware signatures:
  - `func(data, context, memory)`
  - `func(data, context, *, memory=...)`

Use explicit graph edges for control flow and return either:

- a `dict` of updates (most common)
- a `WorkflowNodeResult` for explicit routing/stop behavior

For most deterministic Python steps, prefer `node(...)` for readability. Use
`function_node(...)` when you want the explicit name.

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

The agent-system package is intentionally minimal. The recommended approach is:

- use `Agent` and `WorkflowAgent` as the runtime primitives
- use `create_supervisor(...)` for flexible hierarchical agent systems
- use the legacy `CoordinatorSystem` when it already matches your topology

### 7.0 Which layer should I use?

Use the lightest layer that solves your problem.

#### Runtime primitives: `Agent`, `WorkflowAgent`

Use these directly when:

- you have a single runtime
- your control flow is simple
- you do not need structured handoffs between multiple nodes

#### Preset system

Use a preset when the topology already matches what you need:

- `CoordinatorSystem` for a flat coordinator-worker pattern

#### Flexible hierarchical supervisor

Use `create_supervisor(...)` when:

- one parent runtime should manage a set of specialist workers
- you want workers exposed as tools to that parent
- you want the parent runtime to be either an `Agent` or a `WorkflowAgent`
- you want to build arbitrary hierarchical systems instead of using a fixed preset shape

### 7.1 Coordinator-worker preset

Use `CoordinatorSystem` when a flat coordinator-worker topology already matches your needs.

This preset is best when:

- one coordinator delegates to workers
- workers behave like specialized assistants
- you want simple orchestration with minimal structure overhead
- you do not need explicit structured handoffs between every hop

This preset is primarily `Agent`-builder based.

### 7.2 Minimal coordinator-worker setup

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

### 7.3 Workers with tools

Workers can have their own toolsets via `tools_builder`.

The coordinator preset is primarily `Agent`-builder based.

### 7.4 Flexible hierarchical systems with `create_supervisor(...)`

`create_supervisor(...)` is the main helper for building hierarchical agent systems
flexibly in Cortex.

Use it when:

- you want one supervisor to manage multiple specialists
- you want supervisors to call other supervisors
- you want to combine conversational delegation with explicit workflows
- the topology is more custom than a flat coordinator-worker preset

The key idea is that each worker is wrapped as a tool accepting a single `task` string.
The supervisor is expected to rewrite the user request into a complete delegated task for
that worker.

#### 7.4.1 Agent-based supervisor

Use agent mode when a standard prompt-and-tool loop is enough for the parent supervisor.

```python
from cortex import Agent, LLM, GPTModels, create_supervisor

llm = LLM(model=GPTModels.GPT_4O_MINI)

research_worker = Agent(
    name="Research Worker",
    llm=llm,
    sys_prompt="Find relevant facts and summarize them clearly.",
    mode="async",
)

writer_worker = Agent(
    name="Writer Worker",
    llm=llm,
    sys_prompt="Write polished final responses from provided notes.",
    mode="async",
)

supervisor = create_supervisor(
    name="Research Manager",
    workers=[
        {
            "worker": research_worker,
            "description": "Researches facts, background information, and source material.",
        },
        {
            "worker": writer_worker,
            "description": "Writes polished responses from collected notes and findings.",
        },
    ],
    llm=llm,
    instructions=(
        "Delegate research first when factual grounding is needed. "
        "Use the writer to turn notes into a final answer."
    ),
)

reply = await supervisor.async_ask("Prepare a concise explanation of retrieval-augmented generation.")
```

This mode is a good fit when:

- the supervisor mostly needs normal tool-calling behavior
- delegation strategy can live in a system prompt
- you want the simplest flexible hierarchical setup

#### 7.4.2 Workflow-based supervisor

Use workflow mode when the parent supervisor needs explicit routing, state, or staged
execution.

```python
from cortex import (
    Agent,
    LLM,
    GPTModels,
    create_supervisor,
    node,
    workflow,
)

llm = LLM(model=GPTModels.GPT_4O_MINI)

research_worker = Agent(
    name="Research Worker",
    llm=llm,
    sys_prompt="Research the topic and provide concise notes.",
    mode="async",
)

review_worker = Agent(
    name="Review Worker",
    llm=llm,
    sys_prompt="Review drafts for quality, correctness, and missing points.",
    mode="async",
)

def build_supervisor_workflow(*, tools, name, context, memory):
    async def plan_and_delegate(data, context, memory=None):
        topic = str(data.get("input") or "")
        tasks = [
            {"worker": tools[0].name, "task": f"Research this topic thoroughly: {topic}"},
            {"worker": tools[1].name, "task": f"Review the topic and identify coverage gaps: {topic}"},
        ]
        return {"delegations": tasks}

    def finish(data, context, memory=None):
        return {
            "delegations": data.get("delegations", []),
            "summary": "Delegation plan prepared",
        }

    return workflow(
        name=name,
        context=context,
        memory=memory,
        nodes=[
            node("plan_and_delegate", func=plan_and_delegate),
            node("finish", func=finish, is_final=True),
        ],
        edges=[edge("plan_and_delegate", "finish")],
        start_node="plan_and_delegate",
    )

supervisor = create_supervisor(
    name="Workflow Research Manager",
    workers=[
        {
            "worker": research_worker,
            "description": "Finds background information and source material.",
        },
        {
            "worker": review_worker,
            "description": "Reviews outputs for correctness and completeness.",
        },
    ],
    workflow_builder=build_supervisor_workflow,
)

result = await supervisor.async_ask("Explain vector databases for a product manager.")
```

This mode is a good fit when:

- delegation should follow explicit stages
- you want structured workflow state
- the parent supervisor needs deterministic routing or aggregation steps

#### 7.4.3 Worker specification

You can pass workers either directly or as dict specs.

The dict form is recommended because it lets you describe each worker clearly:

```python
{
    "worker": my_worker,
    "name": "Research Worker",  # optional
    "description": "Finds facts and background information.",
}
```

The `description` is especially important because it is what helps the supervisor understand
when to use that worker.

#### 7.4.4 Why this helper matters

`create_supervisor(...)` is meant to make hierarchical agent systems easy to express without
forcing one orchestration style.

You can use it to build:

- a simple manager with specialist workers
- a multi-level tree of supervisors
- workflow parents that call conversational workers
- conversational parents that delegate into workflows
- larger systems that mix all of the above

### 7.5 WorkflowAgent as a direct runtime

Use `WorkflowAgent` directly when you want explicit orchestration inside a single runtime without introducing a larger multi-agent topology.

Choose it when you need:

- node-by-node control flow
- retries or fallback logic
- deterministic routing between stages
- parallel branches in one workflow

`WorkflowAgent` is graph-oriented. Define nodes and edges explicitly.

Example:

```python
from cortex import edge, node, router_node, workflow

def route_request(data, context):
    if data.get("kind") == "math":
        return "solve_math"
    return "answer_directly"

wf = workflow(
    name="Support Workflow",
    nodes=[
        router_node(
            "route_request",
            func=route_request,
            possible_next_nodes=["solve_math", "answer_directly"],
        ),
        node("solve_math", func=lambda data, context: {"_output": "42"}, is_final=True),
        node("answer_directly", func=lambda data, context: {"_output": "done"}, is_final=True),
    ],
    edges=[
        edge("route_request", "solve_math"),
        edge("route_request", "answer_directly"),
    ],
    start_node="route_request",
)
```

### 7.5.1 How to create a `WorkflowAgent`

You can create workflows either with the `workflow(...)` helper or by instantiating
`WorkflowAgent(...)` directly. In most application code, the helper is the cleaner choice.

A workflow usually has:

- a `name`
- a list of `nodes`
- a list of `edges`
- an optional `start_node`
- optional shared `context`
- optional `memory`
- a `max_steps` guard
- optional `state_type`
- optional `state_factory`

The most common node-building pattern is:

- `node(...)` (or `function_node(...)`) for deterministic Python logic
- `router_node(...)` for branching
- `parallel_node(...)` for concurrent branches
- `llm_node(...)` for direct LLM-backed nodes

Example with explicit state updates and routing:

```python
from cortex import WorkflowNodeResult, edge, node, router_node, workflow

def classify(data, context):
    text = str(data.get("input") or "")
    kind = "billing" if "invoice" in text.lower() else "general"
    return WorkflowNodeResult.update_state({"kind": kind}, output=kind)

def route(data, context):
    if data.get("kind") == "billing":
        return "billing_reply"
    return "general_reply"

wf = workflow(
    name="Support Workflow",
    nodes=[
        node("classify", func=classify),
        router_node(
            "route",
            func=route,
            possible_next_nodes=["billing_reply", "general_reply"],
        ),
        node(
            "billing_reply",
            func=lambda data, context: {"_output": "Let me help with billing."},
            is_final=True,
        ),
        node(
            "general_reply",
            func=lambda data, context: {"_output": "How can I help?"},
            is_final=True,
        ),
    ],
    edges=[
        edge("classify", "route"),
        edge("route", "billing_reply"),
        edge("route", "general_reply"),
    ],
    start_node="classify",
)
```

To execute it:

```python
run = await wf.async_run("I have an invoice problem")
print(run.final_output)
print(run.state.to_dict())
print(run.to_dict())
```

Use `async_run(...)` when you want the full trace/state record, and `async_ask(...)` when
you only need the final answer.

### 7.5.1.1 Using a custom workflow state class

Workflows can now use a user-defined state type.

This is useful when you want:

- typed workflow fields
- domain-specific helper methods on state
- cleaner node code than manually reading and writing every value through `state.data`

The simplest pattern is to subclass `WorkflowState` and pass it as `state_type`.

Example:

```python
from dataclasses import dataclass, field

from cortex import WorkflowState, edge, node, workflow


@dataclass
class SupportState(WorkflowState):
    customer_id: str | None = None
    tags: list[str] = field(default_factory=list)


def classify(data, context):
    text = str(data.get("input") or "")
    return {"kind": "billing" if "invoice" in text.lower() else "general"}


def finish(data, context):
    return {"_output": {"kind": data.get("kind")}}


wf = workflow(
    name="Support Workflow",
    nodes=[
        node("classify", func=classify),
        node("finish", func=finish, is_final=True),
    ],
    edges=[edge("classify", "finish")],
    start_node="classify",
    state_type=SupportState,
)
```

When the workflow creates state internally, it will now create `SupportState` instead of
the default `WorkflowState`.

### 7.5.1.2 Seeding custom state before execution

You can still construct state explicitly before a run.

Example:

```python
state = wf.create_state("I need help with an invoice", ticket_id="T-100")
state.customer_id = "cust-123"

run = await wf.async_run(state=state)
print(type(run.state).__name__)
print(run.final_output)
```

This pattern is useful when:

- upstream code already prepared state
- you need to populate custom fields before the first node runs
- you want to reuse the same initialized state shape across tests or applications

### 7.5.1.3 Using `state_factory` for custom initialization

If state creation requires custom logic, use `state_factory` instead of `state_type`.

The factory is called with:

- `user_input`
- `initial_data`

Example:

```python
from dataclasses import dataclass

from cortex import WorkflowState, workflow


@dataclass
class ReviewState(WorkflowState):
    review_type: str | None = None


def build_review_state(*, user_input=None, initial_data=None):
    state = ReviewState(input=user_input, review_type="fast")
    if initial_data:
        state.update(initial_data)
    return state


wf = workflow(
    name="Review Workflow",
    nodes=[...],
    state_factory=build_review_state,
)
```

Use `state_factory` when:

- you need construction logic beyond plain dataclass initialization
- default values depend on configuration or environment
- you want one place to normalize initial workflow data

Do not pass both `state_type` and `state_factory` to the same workflow.

### 7.5.2 Memory pattern with `memory_bank`

When using `AgentSystemContext`, `memory_bank` is shared across agents. For workflows,
create one memory instance per workflow runtime and pass it into `WorkflowAgent`.

```python
from cortex import workflow, node

workflow_memory = context.memory_bank.get_memory("support_workflow")

def step(data, context, memory):
    # memory is injected from workflow.async_run(..., memory=...)
    return {"handled": True}

wf = workflow(
    name="Support Workflow",
    context=context,
    memory=workflow_memory,
    nodes=[node("step", step, is_final=True)],
)
```

### 7.5.3 A practical workflow design pattern

A good default structure for business workflows is:

1. **Ingest**
   - normalize input
   - extract obvious fields into state
2. **Route**
   - choose the next node from deterministic logic
3. **Call specialists**
   - use LLM nodes or parallel deterministic branches
4. **Aggregate**
   - combine outputs into state
5. **Finish**
   - produce the final output

This pattern keeps control flow inspectable and makes retries/fallback behavior easy to
reason about.

### 7.5.4 State design recommendations

For most workflows, a good pattern is:

- put generic workflow values in `state.data`
- add typed fields only for values that are central to the workflow
- add helper methods when they make node logic clearer

Example:

```python
@dataclass
class TriageState(WorkflowState):
    severity: str | None = None

    def mark_urgent(self) -> None:
        self.severity = "urgent"
```

Recommended:

- use `state_type` when a simple subclass is enough
- use `state_factory` when construction itself needs logic
- keep `to_dict()` and `clone()` working if you override them

The `clone()` part matters because `ParallelNode` copies the active state for each branch.
If you subclass `WorkflowState`, the inherited `clone()` implementation is usually enough.

### 7.5.5 Building larger systems from existing agents

A practical way to build complex systems in Cortex is:

- keep each specialist as an `Agent`
- use `WorkflowAgent` as the orchestration layer
- split deterministic routing/state management into function/router nodes
- reserve LLM nodes for work that benefits from model reasoning

A common structure looks like:

```python
intake -> classify -> analyze_in_parallel -> verify -> finalize
```

Where:

- `intake` and `verify` are often `node(...)`
- `classify` is often `router_node(...)` or `llm_node(...)`
- `analysis` stages are often `parallel_node(...)` or `llm_node(...)`
- `finalize` is often a final `node(...)` or `llm_node(...)`

This separation works well because it keeps:

- orchestration deterministic and inspectable
- specialist behavior reusable
- system growth modular instead of monolithic

### 7.5.6 Observability: tracing and inspection

Workflows provide rich tracing and inspection capabilities out of the box.

#### Basic run inspection

```python
run = await wf.async_run("my input")

# Check the result
print(run.status)           # "completed", "paused", "error"
print(run.final_output)     # final output from the workflow
print(run.duration_ms)      # execution time in milliseconds

# Inspect each node execution
for trace in run.traces:
    print(f"{trace.node_name}: {trace.status} in {trace.duration_ms}ms")
    if trace.error:
        print(f"  Error: {trace.error}")
```

#### Customizing what gets captured

```python
from cortex.workflow import WorkflowObservability

obs = WorkflowObservability(
    capture_state=True,           # capture state snapshots
    capture_state_diff=True,      # capture what changed
    include_node_output=True,     # include outputs in traces
    include_input=True,           # include node inputs
    include_timing=True,          # include timing data
    include_errors=True,          # include error details
    max_state_capture_size=5000,  # truncate large values
    redact_patterns=["api_key", "password"],  # redact sensitive data
)

run = await wf.async_run("input", observability=obs)
```

#### Lifecycle hooks

Inject custom logic at key points:

```python
from cortex.workflow import WorkflowHooks

hooks = WorkflowHooks(
    on_node_start=async def(node_name, state, context):
        print(f"Starting: {node_name}"),
    on_node_complete=async def(node_name, state, output, context):
        print(f"Completed: {node_name} -> {output}"),
    on_run_start=async def(run_id, input, context):
        print(f"Run started: {run_id}"),
    on_run_complete=async def(run, context):
        print(f"Run completed with status: {run.status}"),
)

run = await wf.async_run("input", hooks=hooks)
```

#### Event streaming

Stream lifecycle events as they happen:

```python
async for event in wf.async_run_events("my input"):
    # event.type: "run_start", "node_start", "node_complete", "node_error", "run_complete", "pause", "resume"
    print(f"[{event.timestamp}] {event.type}", end="")
    if event.node_name:
        print(f" on {event.node_name}", end="")
    if event.data:
        print(f" -> {event.data.get('output')}", end="")
    print()
```

#### Explainability helpers

The run object provides helpers for understanding execution:

```python
# Visual timeline
timeline = run.to_timeline()
# Returns: list of (node_name, start_time_ms, duration_ms, status)

# Mermaid diagram
mermaid = run.to_mermaid()
# Returns: Mermaid flowchart markup showing the execution path

# Failure explanations
explanations = run.explain_failures()
# Returns: list of {"node": str, "error": str, "suggestion": str}
```

### 7.5.7 Human-in-the-Loop (HITL)

Workflows can pause for human approval and resume later.

#### Basic pause and resume

```python
from cortex.workflow import WorkflowNodeResult

def process_with_approval(data, context):
    amount = data.get("amount", 0)
    if amount > 1000:
        # Pause for approval
        return WorkflowNodeResult.interrupt_run(
            reason="high_value_transaction",
            payload={"amount": amount, "recipient": data.get("recipient")},
            resume_node="after_approval",
        )
    return {"processed": True}

def after_approval(data, context):
    approval = data.get("_human_input", "")
    if "approve" in approval.lower():
        return {"processed": True, "approved": True}
    return {"processed": False, "approved": False}
```

#### Using in a workflow

```python
from cortex import workflow, node, edge

wf = workflow(
    name="Approval Flow",
    nodes=[
        node("process", func=process_with_approval),
        node("after_approval", func=after_approval, is_final=True),
    ],
    edges=[edge("process", "after_approval")],
    start_node="process",
)

# First run - may pause
run = await wf.async_run("process payment of $2000")
if run.pause:
    print(f"Paused: {run.pause.reason}")
    print(f"Payload: {run.pause.payload}")
    # Save run for later resumption
    saved_run = run

# Later - resume with user input
resumed = await wf.async_resume(saved_run, user_input="Approved")
print(resumed.final_output)  # {"processed": True, "approved": True}
```

#### Stateless workflow, stateful runs

Key insight: `WorkflowAgent` is stateless and can be reused across requests.

```python
# Create once at startup
wf = workflow(name="Approval Flow", nodes=[...], edges=[...])

# Use for many requests
async def handle_request(input_data):
    run = await wf.async_run(input_data)
    if run.pause:
        # Store run somewhere (database, cache, etc.)
        await db.save_run(run.run_id, run)
        return {"status": "pending_approval", "run_id": run.run_id}
    return {"result": run.final_output}

async def handle_approval(run_id, user_response):
    run = await db.load_run(run_id)
    resumed = await wf.async_resume(run, user_input=user_response)
    return {"result": resumed.final_output}
```

This pattern allows:
- Single workflow instance shared across all requests
- Pause state persisted externally (database, Redis, etc.)
- Clean separation between workflow definition and execution state

### 7.6 Graph-first builder pattern

For dynamic workflows, you can build incrementally:

```python
from cortex import WorkflowAgent, edge

wf = WorkflowAgent(name="Dynamic Flow", context=context, memory=memory)

@wf.node
def start(data, context):
    return {"stage": "ready"}

@wf.node("finish", is_final=True)
def done(data, context):
    return {"_output": data.get("stage")}

wf.add_edge("start", "finish")
wf.build()
```

Important: after any `add_node(...)` / `add_edge(...)` mutation, call `build()` before running.

### 7.7 Usage tracking

Provide `usage` in `AgentSystemContext` and it will be passed through:

```python
from cortex.message import AgentUsage

usage = AgentUsage()
context = AgentSystemContext(usage=usage, memory_bank=memory_bank)

await system.async_ask("Hello")

print(usage.format())
```

### 7.8 Whiteboard (agent communication)

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
