# Cortex Agent Framework

Cortex is a Python framework for building LLM-powered applications across three layers:

- **Runtime primitives** for direct `Agent` and `WorkflowAgent` usage
- **Composable task-executor helpers** for custom multi-agent topologies
- **Preset multi-agent systems** for coordinator-worker and hierarchical orchestration

It is designed so you can start with a single runtime, move up to explicit workflows when needed, and adopt higher-level multi-agent abstractions only when your problem actually requires them.

## Features

- **Direct agent runtime**
  - Build prompt-and-tool-driven agents with `Agent`
  - Use local tools and provider-hosted tools

- **Explicit workflow runtime**
  - Build step-based flows with `WorkflowAgent`
  - Use deterministic routing, retries, fallbacks, and parallel branches

- **Composable multi-agent layer**
  - Use `TaskDesc`, `TaskResult`, `TaskExecutor`, and `TaskExecutorBuilder`
  - Mix `Agent`, `WorkflowAgent`, and custom `run_task(...)` runtimes behind one surface
  - Expose executors as tools and synthesize structured child results

- **Preset systems**
  - `CoordinatorSystem` for flat coordinator-worker delegation
  - `HierarchicalAgentSystem` for gateway -> manager -> specialist orchestration

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
  - when you need branching, retries, or parallel steps

- **Use the task composition layer**
  - when you want structured handoffs between multiple runtimes
  - when you want a shared abstraction for custom topologies

- **Use a preset system**
  - when your topology already matches coordinator-worker or hierarchical orchestration

## Key surfaces

Top-level exports include:

- `Agent`, `Tool`, `LLM`
- `WorkflowAgent`, `FunctionStep`, `LLMStep`, `RouterStep`, `ParallelStep`
- `CoordinatorSystem`, `CoordinatorAgentBuilder`, `WorkerAgentBuilder`
- `HierarchicalAgentSystem`, `GatewayNodeBuilder`, `DepartmentManagerBuilder`, `SpecialistNodeBuilder`
- `TaskExecutorBuilder`, `TaskDesc`, `TaskResult`, `create_task_desc(...)`, `execute_task_executor(...)`

For a fuller guide, see:

- `docs/cortex_usage.md`
- `docs/cortex_design.md`
- `cortex/agent_system/README.md`

## Installation

Install Cortex using pip:

```bash
pip install cortex
```