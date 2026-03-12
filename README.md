# Cortex Agent Framework

Cortex is a Python framework for building LLM-powered applications with:

- **Runtime primitives** for direct `Agent` and `WorkflowAgent` usage
- **Function-first workflow and runtime composition helpers**
- **A preset system** for coordinator-worker orchestration

It is designed so you can start with a single runtime, move up to explicit node-oriented workflows when needed, and use the coordinator preset when that topology fits your problem.

## Features

- **Direct agent runtime**
  - Build prompt-and-tool-driven agents with `Agent`
  - Use local tools and provider-hosted tools

- **Explicit workflow runtime**
  - Build node-oriented flows with `WorkflowAgent`
  - Use deterministic routing, retries, fallbacks, parallel branches, and nested runtimes
  - Compose workflows with function-first helpers such as `workflow(...)`, `function_node(...)`, `router_node(...)`, `parallel_node(...)`, `runtime_node(...)`, and `llm_node(...)`

- **Runtime composition**
  - Lazily resolve concrete runtimes, runtime builders, agents, and workflows only when needed
  - Wrap plain callables as runtimes with `function_runtime(...)`
  - Use `RuntimeNode` to embed runtimes inside workflow graphs

- **Preset systems**
  - `CoordinatorSystem` for flat coordinator-worker delegation

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

- **Use a preset system**
  - when your topology already matches coordinator-worker orchestration

## Key surfaces

Top-level exports include:

- `Agent`, `Tool`, `LLM`
- `WorkflowAgent`, `workflow`, `function_node`, `router_node`, `parallel_node`, `runtime_node`, `llm_node`
- `RuntimeNode`, `function_runtime`, `resolve_runtime`, `adapt_runtime`, `invoke_runtime`
- `FunctionStep`, `LLMStep`, `RouterStep`, `ParallelStep`
- `CoordinatorSystem`, `CoordinatorAgentBuilder`, `WorkerAgentBuilder`

For a fuller guide, see:

- `docs/cortex_usage.md`
- `docs/cortex_design.md`
- `cortex/agent_system/README.md`

## Installation

Install Cortex using pip:

```bash
pip install cortex
```