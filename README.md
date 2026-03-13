# Cortex Agent Framework

Cortex is a Python framework for building LLM-powered applications with:

- **Runtime primitives** for direct `Agent` and `WorkflowAgent` usage
- **Function-first workflow and runnable composition helpers**
- **A preset system** for coordinator-worker orchestration
- **A flexible supervisor helper** for hierarchical agent systems

It is designed so you can start with a single runtime, move up to explicit node-oriented workflows when needed, use `create_supervisor(...)` for custom hierarchical agent systems, and use the coordinator preset when that topology fits your problem.

## Features

- **Direct agent runtime**
  - Build prompt-and-tool-driven agents with `Agent`
  - Use local tools and provider-hosted tools

- **Explicit workflow runtime**
  - Build node-oriented flows with `WorkflowAgent`
  - Use deterministic routing, retries, fallbacks, parallel branches, and nested runnables
  - Compose workflows with function-first helpers such as `workflow(...)`, `function_node(...)`, `router_node(...)`, `parallel_node(...)`, `runnable_node(...)`, and `llm_node(...)`

- **Runnable composition**
  - Lazily resolve concrete runnables, runnable builders, agents, and workflows only when needed
  - Wrap plain callables as runnables with `function_runnable(...)`
  - Use `RunnableNode` to embed runnables inside workflow graphs

- **Preset systems**
  - `CoordinatorSystem` for flat coordinator-worker delegation

- **Flexible hierarchical supervision**
  - Use `create_supervisor(...)` to expose worker agents as tools to a parent supervisor
  - Build either an `Agent`-based supervisor or a custom `WorkflowAgent`-based supervisor
  - Create arbitrary hierarchical agent systems by recursively composing workers, supervisors, workflows, and runnables

- **Context and memory**
  - Shared `AgentSystemContext`
  - Async memory banks and optional usage tracking
  - Periodic conversation summarization for long-running agents

- **Optional whiteboard**
  - Channel-based shared messaging for multi-agent coordination
  - In-memory and Redis-backed storage implementations

## Choosing the right layer

- **Use `Agent`**
  - when prompt + tools are enough
  - when the flow is conversational and local

- **Use `WorkflowAgent`**
  - when control flow should be explicit
  - when you need branching, retries, fallbacks, or parallel nodes
  - when you want explicit state and inspectable graph execution

- **Use `create_supervisor(...)`**
  - when you want a supervisor that manages specialist workers as tools
  - when you want a flexible entry point for hierarchical agent systems
  - when the parent runtime may be either a normal `Agent` or a custom `WorkflowAgent`

- **Use a preset system**
  - when your topology already matches coordinator-worker orchestration

## Key surfaces

Top-level exports include:

- `Agent`, `Tool`, `LLM`
- `WorkflowAgent`, `workflow`, `function_node`, `router_node`, `parallel_node`, `runnable_node`, `llm_node`
- `RunnableNode`, `function_runnable`, `resolve_runnable`, `adapt_runnable`, `invoke_runnable`
- `WorkflowNodeResult`, `RunnableNode`, `RouterNode`, `ParallelNode`
- `CoordinatorSystem`, `CoordinatorAgentBuilder`, `WorkerAgentBuilder`, `create_supervisor`

## Hierarchical agent systems with `create_supervisor(...)`

`create_supervisor(...)` is the main Cortex helper for building hierarchical agent systems
flexibly.

It is designed for the case where:

- one parent runtime should coordinate multiple specialist workers
- workers should be available as tools to that parent
- the parent may be a standard `Agent` or a custom `WorkflowAgent`
- you want to keep the handoff contract simple and scalable

Each worker is wrapped as a tool that accepts a single rewritten `task` string. This is an
intentional design choice: the supervisor is expected to understand each worker's role and
delegate a fully rewritten task, rather than forwarding the raw user input unchanged.

There are two supported modes:

- **Agent mode**
  - pass `llm`
  - Cortex builds an `Agent` supervisor with the worker tools attached

- **Workflow mode**
  - pass `workflow_builder`
  - Cortex prepares the worker tools and lets you build any `WorkflowAgent` you want

This makes `create_supervisor(...)` the preferred entry point when you want to create:

- manager-worker systems
- multi-level teams of specialists
- supervisors that call other supervisors
- hybrid systems mixing conversational agents and explicit workflows

For a fuller guide, see:

- `docs/cortex_usage.md`
- `docs/cortex_design.md`
- `cortex/agent_system/README.md`

## Installation

Install Cortex using pip:

```bash
pip install cortex
```