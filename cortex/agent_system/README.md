# Agent System

The agent-system package gives you **building blocks at multiple layers** for constructing multi-agent applications on top of the core `Agent`, `WorkflowAgent`, `LLM`, and `Tool` primitives.

## Overview

Cortex now exposes three useful layers:

- **Runtime primitives**: `Agent` and `WorkflowAgent`
- **Composable helper layer**: shared task/runner/tool orchestration helpers in `cortex.agent_system`
- **Preset systems**: `CoordinatorSystem` and `HierarchicalAgentSystem`

This layering lets you choose the lightest abstraction that matches your topology instead of forcing everything into one framework.

## Choosing the right layer

### Use runtime primitives directly

Use plain `Agent` or `WorkflowAgent` when:

- you only need one agent
- your orchestration is simple and local
- you do not need structured handoffs between multiple nodes

Use:

- `Agent` when prompt + tools are enough
- `WorkflowAgent` when you want explicit steps, branching, retries, or parallel execution in a single runtime

### Use the composition helpers

Use the composition helpers when:

- you want to build a **custom topology**
- you want structured handoffs and normalized results
- you want to mix `Agent`, `WorkflowAgent`, and custom runtimes behind one surface
- you want to expose child runners as tools and synthesize their results

This is the recommended layer when you want reusable building blocks without committing to a preset topology.

### Use a preset system

Use a preset when the topology already matches your problem:

- `CoordinatorSystem` for a flat coordinator-worker shape
- `HierarchicalAgentSystem` for gateway → manager → specialist orchestration with reinterpretation and synthesis at each layer

## What the agent-system package provides

- **Separation of concerns**: builders define structure, systems manage runtime
- **Composable orchestration**: task briefs, results, runner builders, tool wrappers, and synthesis helpers
- **Memory management**: built-in support for conversation history via `AgentMemoryBank`
- **Usage tracking**: optional tracking of API calls and token usage
- **Preset systems**: prebuilt coordinator-worker and hierarchical topologies

## Core Components

### Base Classes

#### `AgentBuilder`
Base class for building agents. Subclasses define how to construct an agent with:
- `name`: Agent identifier
- `llm`: Language model to use
- `prompt_builder`: Callable that generates the system prompt
- `tools_builder`: Optional callable that loads tools
- `memory_k`: Number of recent messages to include in context

#### `AgentSystem`
Base system class that manages agent lifecycle and provides the `async_ask()` interface.

#### `AgentSystemContext`
Runtime context passed to agents, containing:
- `usage`: Optional `AgentUsage` for tracking API calls
- `memory_bank`: `AsyncAgentMemoryBank` for conversation history
- `llm_primary`: Pre-configured primary LLM (GPT-5-MINI, minimal reasoning)
- `llm_creative`: Pre-configured creative LLM (GPT-5-MINI, medium reasoning)

### Composition layer

The composition layer is the reusable middle layer between raw runtimes and preset systems.

Core concepts:

- `TaskDesc`
- `TaskResult`
- `TaskExecutor`
- `BuiltTaskExecutor`
- `TaskExecutorBuilderBase`
- `TaskExecutorBuilder`

Core helper APIs:

- `create_task_desc(...)`
- `create_child_task_desc(...)`
- `resolve_task_executor(...)`
- `execute_task_executor(...)`
- `create_task_tool(...)`
- `execute_task_tools(...)`
- `synthesize_task_results(...)`
- `coerce_task_result(...)`
- `should_escalate(...)`

Supporting helper types:

- `TaskTextFactory`
- `TaskMetadataFactory`

What each piece does:

- `TaskDesc` is the structured handoff contract
- `TaskResult` is the normalized result contract
- `TaskExecutorBuilderBase` is the neutral shared builder foundation
- `TaskExecutorBuilder` is the ergonomic public builder for composition
- `TaskExecutorBuilder.create_agent(...)` wraps an `Agent`-backed runtime
- `TaskExecutorBuilder.create_workflow(...)` wraps a `WorkflowAgent`-backed runtime
- `create_task_tool(...)` exposes an executor as a tool
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

The composition helpers are intentionally runtime-symmetric. You can combine:

- `Agent`-backed executors
- `WorkflowAgent`-backed executors
- custom runtimes that implement `run_brief(...)`

inside the same topology.

### Minimal composition example

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

### Coordinator System

#### `CoordinatorAgentBuilder`
Builder for coordinator agents that orchestrate worker agents. Coordinators:
- Analyze user requests and delegate to appropriate workers
- Forward messages verbatim without alteration
- Add factual context when needed
- Relay worker responses back to users

#### `WorkerAgentBuilder`
Builder for specialized worker agents. Workers:
- Handle specific domain tasks
- Can have their own tools and memory
- Communicate through the coordinator
- Support optional "thinking" mode for internal reasoning

#### `CoordinatorSystem`
Complete system implementation that:
- Manages a coordinator agent and multiple worker agents
- Exposes workers as tools to the coordinator
- Handles agent construction and caching
- Provides simple `async_ask()` interface

Use this preset when:

- your topology is flat
- a single coordinator delegates to workers
- workers mostly act as specialized assistants
- you do not need structured multi-layer reinterpretation between each hop

### Hierarchical Agent System

The hierarchical system is for multi-layer orchestration where nodes reinterpret and synthesize work instead of forwarding raw messages verbatim.

Core types:

- `GatewayNodeBuilder`
- `DepartmentManagerBuilder`
- `SpecialistNodeBuilder`
- `DepartmentSpec`
- `HierarchicalAgentSystem`

Hierarchy shape:

- user
- gateway
- department managers
- specialists

Each handoff uses a structured `DelegationBrief`, and each node responds with a `NodeResult`.

Use this preset when:

- you want gateway triage
- managers should rewrite or refine tasks before delegating
- specialists should return structured child results
- synthesis should happen at each level

#### Runtime-symmetric builder API

Every role can be backed by either `Agent` or `WorkflowAgent`.

Gateway:

- `GatewayNodeBuilder.create_agent(...)`
- `GatewayNodeBuilder.create_workflow(...)`
- `GatewayNodeBuilder.create_default(...)`

Manager:

- `DepartmentManagerBuilder.create_agent(...)`
- `DepartmentManagerBuilder.create_workflow(...)`
- `DepartmentManagerBuilder.create_default(...)`

Specialist:

- `SpecialistNodeBuilder.create_agent(...)`
- `SpecialistNodeBuilder.create_workflow(...)`

Workflow-backed gateway and manager factories can accept:

- `context`
- `installed_tools`
- `child_tools`

This lets workflow-based orchestrators call their child department or specialist tools directly.

#### Fluent hierarchy construction

Use:

- `DepartmentSpec.create(...)`
- `add_specialist(...)`
- `add_specialists(...)`

to build department trees cleanly.

#### Recommended system entrypoint

Use `HierarchicalAgentSystem.create(...)` to assemble the full hierarchy.

#### Example

See:

- `examples/hierarchical_agent_system_example.py`

## Quick Start: coordinator preset

```python
from cortex import (
    LLM,
    GPTModels,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    CoordinatorSystem,
    AgentSystemContext,
    AsyncAgentMemoryBank,
)

# Create context
memory_bank = AsyncAgentMemoryBank()
context = AgentSystemContext(memory_bank=memory_bank)

# Define workers
math_worker = WorkerAgentBuilder(
    name="Math Expert",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You are a math expert.",
    introduction="Solves mathematical problems",
)

# Define coordinator
coordinator = CoordinatorAgentBuilder(
    name="Coordinator",
    llm=LLM(model=GPTModels.GPT_4O_MINI),
    prompt_builder=lambda ctx: "You coordinate specialized workers.",
)

# Create system
system = CoordinatorSystem(
    coordinator_builder=coordinator,
    workers=[math_worker],
    context=context,
)

# Use it
response = await system.async_ask("What is 2 + 2?")
```

If you want a custom topology instead of a preset, start with the composition-layer example above.

## Runtime choices

### `Agent`

Choose `Agent` when the behavior is mostly:

- prompt-driven
- tool-augmented
- conversational

### `WorkflowAgent`

Choose `WorkflowAgent` when the behavior needs:

- explicit steps
- deterministic routing
- retries or fallback behavior
- parallel branches
- a clear control-flow graph

You can use `WorkflowAgent` directly, through `TaskExecutorBuilder.create_workflow(...)`, or inside hierarchical builders like `GatewayNodeBuilder.create_workflow(...)`.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          Agent System Layers                       │
├────────────────────────────────────────────────────────────────────┤
│ Runtime primitives                                                 │
│ - Agent                                                            │
│ - WorkflowAgent                                                    │
├────────────────────────────────────────────────────────────────────┤
│ Composition helpers                                                │
│ - TaskDesc / TaskResult                                            │
│ - TaskExecutorBuilderBase / TaskExecutorBuilder                    │
│ - create_task_tool / execute_task_tools / synthesize_task_results  │
├────────────────────────────────────────────────────────────────────┤
│ Preset systems                                                     │
│ - CoordinatorSystem                                                │
│ - HierarchicalAgentSystem                                          │
└────────────────────────────────────────────────────────────────────┘
                               │
                               │ share
                               ▼
                  ┌─────────────────────────┐
                  │  AgentSystemContext     │
                  │  - Memory Bank          │
                  │  - Usage Tracking       │
                  │  - Shared LLMs          │
                  └─────────────────────────┘
```

## Key Features

### 1. Memory Management
Each agent maintains its own conversation history through the shared `AsyncAgentMemoryBank`:

```python
memory_bank = AsyncAgentMemoryBank()
context = AgentSystemContext(memory_bank=memory_bank)

# Each agent gets its own memory keyed by agent name
# Memory is automatically managed by the system
```

### 2. Usage Tracking
Track API calls and token usage across all agents:

```python
from cortex.message import AgentUsage

usage = AgentUsage()
context = AgentSystemContext(usage=usage, memory_bank=memory_bank)

# After making calls
print(f"Total tokens: {usage.total_tokens}")
```

### 3. Dynamic Tool Loading
Agent builders and workflow-backed runtimes can load tools dynamically based on context:

```python
def tools_builder(context):
    # Access context to determine which tools to load
    return [tool1, tool2]

worker = WorkerAgentBuilder(
    name="Worker",
    llm=llm,
    prompt_builder=prompt_builder,
    tools_builder=tools_builder,  # Called at agent build time
)
```

### 4. Flexible Prompt Building
Prompts can be static or dynamic:

```python
# Static prompt
def static_prompt(ctx):
    return "You are a helpful assistant."

# Dynamic prompt using context
def dynamic_prompt(ctx):
    # Access context properties
    return f"You are an assistant with access to {len(ctx.tools)} tools."
```

## Examples

See `examples/agent_system_example.py` for comprehensive examples including:
- Basic coordinator-worker setup
- Workers with custom tools
- Usage tracking
- Custom context configuration

See `examples/hierarchical_agent_system_example.py` for the hierarchical preset.

Use the composition helpers when you want to build your own topology instead of following either preset.

## Extending the System

### Creating Custom Builders

Extend `AgentBuilder` to create custom agent builders:

```python
class CustomAgentBuilder(AgentBuilder):
    async def build_agent(self, *, context, **kwargs):
        # Custom agent construction logic
        prompt = await self.build_prompt(context)
        tools = await self.load_tools(context)
        
        memory_bank = await context.get_memory_bank()
        memory = await memory_bank.get_agent_memory(self.name_key, k=self.memory_k)
        
        return Agent(
            name=self.name,
            llm=self.llm,
            tools=tools,
            sys_prompt=prompt,
            memory=memory,
            context=context,
        )
```

### Creating Custom Systems

Extend `AgentSystem` to create custom system implementations:

```python
class CustomSystem(AgentSystem):
    async def get_agent(self):
        # Custom agent retrieval/construction logic
        return await self._builder.build_agent(context=self._context)
```

## Best Practices

1. **Keep prompts focused**: Each worker should have a clear, specific role
2. **Use memory wisely**: Set appropriate `memory_k` values to balance context and cost
3. **Track usage**: Enable usage tracking in production to monitor costs
4. **Separate concerns**: Use builders for structure, systems for runtime management
5. **Test workers independently**: Build and test workers before integrating into a coordinator system

## Migration from Direct Runtime Usage

If you're currently using `Agent` directly:

```python
# Old approach
agent = Agent(
    llm=llm,
    tools=tools,
    sys_prompt=prompt,
    memory=memory,
)
response = await agent.async_ask(message)
```

You can migrate to the Agent System:

```python
# New approach
builder = WorkerAgentBuilder(
    name="My Agent",
    llm=llm,
    prompt_builder=lambda ctx: prompt,
    tools_builder=lambda ctx: tools,
)

system = CoordinatorSystem(
    coordinator_builder=coordinator,
    workers=[builder],
    context=context,
)
response = await system.async_ask(message)
```

Benefits:
- Built-in memory management
- Usage tracking
- Easy to add more workers
- Coordinator handles delegation automatically

If you're currently using `Agent` directly and you want structured multi-executor orchestration, the nearest migration path is usually the composition layer:

- wrap agent runtimes with `TaskExecutorBuilder.create_agent(...)`
- expose child executors with `create_task_tool(...)`
- use `execute_task_tools(...)` and `synthesize_task_results(...)` to build your own orchestration layer
