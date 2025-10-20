# Agent System

A higher-level API for building multi-agent systems on top of the core `Agent`, `LLM`, and `Tool` components.

## Overview

The Agent System provides a structured way to build complex multi-agent applications with:

- **Separation of concerns**: Builders define agent structure, Systems manage runtime
- **Memory management**: Built-in support for conversation history via `AgentMemoryBank`
- **Usage tracking**: Optional tracking of API calls and token usage
- **Coordinator-Worker pattern**: Pre-built implementation for orchestrating specialized agents

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

## Quick Start

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

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CoordinatorSystem                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              CoordinatorAgent                         │  │
│  │  - Analyzes requests                                  │  │
│  │  - Delegates to workers                               │  │
│  │  - Manages conversation flow                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           │ delegates to                     │
│                           ▼                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Worker Agent │  │ Worker Agent │  │ Worker Agent │     │
│  │   (Math)     │  │  (Writing)   │  │  (Research)  │     │
│  │              │  │              │  │              │     │
│  │ - Tools      │  │ - Tools      │  │ - Tools      │     │
│  │ - Memory     │  │ - Memory     │  │ - Memory     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ uses
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
Workers can load tools dynamically based on context:

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

## Migration from Direct Agent Usage

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
