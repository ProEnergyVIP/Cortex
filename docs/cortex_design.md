# Cortex Design & Architecture

This document describes how Cortex is structured internally and how the major abstractions fit together:

- `LLM` and provider backends (`LLMBackend`)
- Message model (`Message`, `AIMessage`, tool call messages)
- Tools (`FunctionTool` and hosted tools)
- `Agent` (conversation loop, tool execution, memory)
- Agent System (`AgentBuilder`, `AgentSystem`, coordinator/worker)
- Whiteboard (shared multi-agent state)

The goal is to help you understand **how requests flow through the library**, where to extend it, and what invariants the core code assumes.

---

## 1) Mental model: end-to-end request flow

At a high level, Cortex is built around this pipeline:

1. You create an `LLM(model=...)`.
2. You create tools (usually `FunctionTool` / alias `Tool`).
3. You create an `Agent(llm, tools, sys_prompt, memory, context, ...)`.
4. You call `agent.ask(...)` (sync) or `await agent.async_ask(...)` (async).
5. The agent builds a message list:
   - `SystemMessage` from the agent’s `sys_prompt`
   - optional memory/history
   - the user message(s)
6. The agent calls the `LLM`, optionally including tool schemas.
7. The backend returns an `AIMessage`.
8. If the message contains tool calls:
   - the agent executes the tools locally (optionally in parallel)
   - tool results are appended as `ToolMessageGroup`
   - the agent loops and calls the LLM again
9. When a final natural-language (or JSON) answer is produced, it’s returned and saved to memory.

The Agent System adds a layer above this:

- Builders construct agents lazily from a runtime context.
- Systems own an `async_ask(...)` API and choose the agent(s) used to answer.
- `CoordinatorSystem` exposes worker agents as tools to a coordinator agent.

---

## 2) Messages: a common internal representation

Cortex normalizes communication into message dataclasses defined in `cortex/message.py`:

- `SystemMessage(content=...)`
- `DeveloperMessage(content=...)`
- `UserMessage(content=..., images=..., files=...)`
- `AIMessage(content=..., function_calls=..., usage=..., model=...)`
- `ToolMessage(content=..., tool_call_id=...)`
- `ToolMessageGroup(tool_messages=[...])`

### 2.1 Tool calls

Provider tool calls are represented as `FunctionCall` objects:

- `name`
- `arguments` (string or object)
- `id`/`call_id`

The `Agent` looks for `ai_msg.function_calls` to decide whether to run tools.

### 2.2 Usage tracking

Token usage is tracked by:

- `MessageUsage` (per model call)
- `AgentUsage` (accumulator across multiple calls/models)

Agents add usage when `ai_msg.usage` and `ai_msg.model` are present.

---

## 3) LLM and Backends

### 3.1 `LLM`: runtime wrapper with fallback

`cortex/LLM.py` defines `LLM`, which is a lightweight runtime wrapper around a provider backend instance.

Responsibilities:

- Normalize model identifiers (Enums are supported).
- Resolve a backend via `LLMBackend.get_backend(model)`.
- Build an `LLMRequest(system_message, messages, temperature, max_tokens, tools, reasoning_effort)`.
- Provide:
  - `call(..., streaming=False)`
  - `async_call(..., streaming=False)`
- Support a **model backup chain**:
  - `LLM.set_backup_backend(primary_model, backup_model)`
  - On exception, the model is marked failed and retried against its backup.
  - Cycle detection prevents invalid fallback graphs.

**Important invariant:** `LLM.call(..., streaming=True)` and `LLM.async_call(..., streaming=True)` reject tool usage (`tools` must be empty) because streaming + tool calling is not supported.

### 3.2 `LLMBackend`: provider integration + encoding registry

`cortex/backend.py` defines:

- `LLMRequest`
- `LLMBackend`

Backends are registered by model key (exact or wildcard patterns like `gpt-*`).

`LLMBackend` also hosts the **message/tool encoder registry**:

- Message encoders map a `Message` subclass to provider-native payload(s).
- Tool encoders map a `Tool` subclass to provider-native tool schema.

The base class provides:

- `encode_message(msg) -> list[dict]` (always returns a list)
- `encode_tool(tool) -> dict`
- default fallback encoders (best-effort)

Backends implement one or more of:

- `call(req) -> AIMessage`
- `async_call(req) -> AIMessage`
- `stream(req) -> Iterable[str]`
- `async_stream(req) -> AsyncIterable[str]`

### 3.3 OpenAI backend

`cortex/backends/openai.py` implements `OpenAIBackend`.

Notable design choices:

- Uses OpenAI Responses-style fields (`instructions`, `input`, `tools`).
- Registers encoders for:
  - system/developer/user/vision/assistant/tool messages
  - function tools and hosted tools (`WebSearchTool`, `CodeInterpreterTool`, `FileSearchTool`, `MCPTool`)
- Registers itself as a wildcard backend: `LLMBackend.register_backend('gpt-*', OpenAIBackend)`.

### 3.4 Anthropic backend

`cortex/backends/anthropic.py` implements `AnthropicBackend`.

Notable design choices:

- Encodes messages into Anthropic’s `messages.create` schema.
- Implements tool calling via `tool_use` blocks and converts them to `FunctionCall`.
- Provides streaming by iterating provider stream events and emitting text deltas.

---

## 4) Tools

Tools are defined in `cortex/tool.py`.

### 4.1 Tool taxonomy

Cortex distinguishes between:

- **Locally executed tools**: `FunctionTool` (alias: `Tool`)
- **Hosted tools**: provider-native tools that run on the model/provider side (e.g. web search, file search, code interpreter, MCP)

In the core `Agent` loop, only `FunctionTool` is executable locally.

### 4.2 `FunctionTool`

A `FunctionTool` is a dataclass with:

- `name`, `func`, `description`, `parameters` (JSON schema)
- optional: `strict`

Invocation conventions:

- The `func` may take 0–3 positional args:
  - `func(tool_input)`
  - `func(tool_input, context)`
  - `func(tool_input, context, agent)`

Execution:

- Sync: `tool.run(...)`
- Async: `await tool.async_run(...)`

Call limiting:

- Each tool tracks `_called_times`.
- `Agent` enforces `tool_call_limit` and removes tools that exceed it.

### 4.3 Hosted tools

Hosted tools are also `BaseTool` subclasses (e.g. `WebSearchTool`, `MCPTool`).

They are *encoded* and sent to the provider via backend tool encoders, but they are not executed locally by `Agent.run_tool_func`.

---

## 5) `llmfunc`: turning an LLM into a function

`cortex/LLMFunc.py` defines `llmfunc(...)` which constructs a callable (sync or async) that:

- Uses a fixed `SystemMessage(prompt)`
- Converts input into `UserMessage`s
- Calls the LLM
- Optionally enforces JSON output with `result_shape` (by appending a schema-format instruction)
- Optionally validates with a `check_func` that returns `CheckResult`
- Retries up to `max_attempts` if validation fails

Streaming mode:

- `llmfunc(..., streaming=True)` returns a generator/async generator of text deltas
- Streaming is **not compatible** with `result_shape` or `check_func`

---

## 6) Agent: conversation orchestration + tool execution

`cortex/agent.py` defines `Agent`, which owns the conversation loop.

### 6.1 Key responsibilities

- Normalize the input into `Message` objects.
- Load memory/history.
- Call the LLM with tool schemas.
- Execute locally runnable tool calls returned by the model.
- Append tool results to the conversation.
- Repeat until a final answer is produced or `loop_limit` is reached.

### 6.2 Sync vs async mode

`Agent(mode='sync' | 'async')` enforces a consistent tool type:

- In `sync` mode, all `FunctionTool.func` must be sync.
- In `async` mode, all `FunctionTool.func` must be async.

This is validated in `_process_and_validate_tools()`.

### 6.3 Parallel tool execution

When the model returns multiple tool calls:

- Async mode uses `asyncio.gather()` to run them concurrently.
- Sync mode uses a `ThreadPoolExecutor`.

You can configure:

- `enable_parallel_tools` (default `True`)
- `max_parallel_tools` to cap concurrency

### 6.4 Tool loop safety

The agent includes several guardrails:

- Tool call repetition detection (tracks last `MAX_RECENT_CALLS`).
- `tool_call_limit` removes overused tools.
- JSON reply mode (`json_reply=True`) parses final `AIMessage.content` via `json.loads`.

Streaming mode:

- `agent.ask(..., streaming=True)` / `agent.async_ask(..., streaming=True)` currently require **no tools**.

### 6.5 Conversation Summary

Agent memory supports an optional **conversation summary** that retains important context from evicted rounds.

#### How it works

1. When `enable_summary=True`, evicted message rounds are accumulated in an **eviction buffer**.
2. Every `summarize_every_n` evictions, the summarization function is called with `(current_summary, all_buffered_messages)` and the buffer is cleared.
3. The returned string replaces the stored summary.
4. On `load_memory()`, up to two `SystemMessage` prefixes are prepended before the conversation rounds:
   - The **summary** (if any) — compressed context from earlier rounds.
   - The **eviction buffer** (if non-empty) — raw text of recently evicted messages not yet summarized.

This two-layer approach ensures the LLM **never loses visibility** of evicted information, even between summarization runs. The periodic LLM call merely compresses the buffer into the summary for token efficiency.

#### Summarization function resolution

The summary function is resolved lazily on first use, in this order:

1. **Custom `summary_fn`** — if provided, used directly (no LLM calls).
2. **Default LLM summarizer** — built from `summary_llm` (or a lazily-created `LLM(model='gpt-5-nano')`). Uses `DEFAULT_SUMMARY_PROMPT` to instruct the LLM to condense the conversation.

#### Failure handling

Summarization errors are caught and logged as warnings. The agent continues normally — a failed summarization never breaks the conversation loop.

#### In-memory vs Redis storage

- **`AgentMemory` / `AsyncAgentMemory`**: summary is stored as an in-memory string (`_summary` field). Lost on process restart.
- **`RedisAgentMemory` / `AsyncRedisAgentMemory`**: summary is persisted in Redis under `{key}:summary`. Survives restarts and is shared across instances pointing at the same key.

#### Integration points

- `AgentMemory.add_messages()` / `AsyncAgentMemory.add_messages()` — eviction counting and summarization trigger.
- `AgentMemory.load_memory()` / `AsyncAgentMemory.load_memory()` — summary injection.
- `AgentMemoryBank.get_agent_memory(**kwargs)` — forwards `enable_summary`, `summary_fn`, `summary_llm`, `summarize_every_n`.

---

## 7) Agent System: builders + runtime systems

The Agent System lives in `cortex/agent_system/`.

It introduces two key abstractions:

- `AgentBuilder`: builds an agent from context (prompt, tools, memory)
- `AgentSystem`: owns a context and provides a single `async_ask(...)` entrypoint

### 7.1 `AgentSystemContext`

`AgentSystemContext` (pydantic model) is designed as the runtime dependency container.

It commonly carries:

- `usage: AgentUsage | None`
- `memory_bank: AsyncAgentMemoryBank | None`
- `whiteboard: Whiteboard | None`

It also defines cached properties for shared, pre-configured LLMs:

- `llm_primary`
- `llm_creative`

### 7.2 `CoordinatorSystem`: coordinator + worker agents

`CoordinatorSystem`:

- accepts a `CoordinatorAgentBuilder`
- accepts a list of `WorkerAgentBuilder`
- installs each worker as a tool exposed to the coordinator (tool name ends in `_agent`)
- lazily builds and caches the coordinator agent

The coordinator decides which worker tools to call.

### 7.3 Worker installation as tools

A worker builder’s `install(...)` method returns a `Tool` (async `FunctionTool`) that:

- builds a worker agent on demand
- forwards `user_input` verbatim
- optionally appends `DeveloperMessage(context_instructions)`
- can enrich context with whiteboard summaries
- returns the worker agent’s JSON response

---

## 8) Whiteboard: shared multi-agent state (optional)

The whiteboard lives under `cortex/agent_system/core/whiteboard.py` and is wired via `AgentSystemContext.whiteboard`.

Design intent:

- Provide a **shared, topic-scoped coordination state**:
  - mission, focus, progress
  - blockers
  - decisions
  - recent updates/activity
- Allow workers to suggest updates (`whiteboard_suggestion`) while keeping the coordinator in control.

When a coordinator has a whiteboard:

- `CoordinatorAgentBuilder` injects special management tools such as:
  - `update_mission_func`, `update_progress_func`, `manage_blocker_func`, `log_decision_func`, `get_team_status_func`, `clear_topic_func`

Workers (when enabled) may emit suggestions which can be auto-applied by `WorkerAgentBuilder.install()` via `context.whiteboard.apply_suggestion(...)`.

---

## 9) Extension points

### 9.1 Add or modify provider backends

To add a new LLM provider:

1. Implement a subclass of `LLMBackend`.
2. Register message encoders for the message types you want to support.
3. Register tool encoders for the tool types you want to support.
4. Register your backend for model keys via `LLMBackend.register_backend(model_key_or_pattern, BackendClass)`.

### 9.2 Add new hosted tool types

To add a new hosted tool type:

1. Create a new `BaseTool` dataclass.
2. Add a tool encoder in each backend that supports it.

### 9.3 Add agent-system patterns

To add a new multi-agent topology:

- Implement your own `AgentSystem` subclass.
- Use builders (`AgentBuilder`) to construct agents from context.

---

## 10) Design constraints & gotchas

- **Streaming and tools do not mix** in the current `LLM`/`Agent` APIs.
- `Agent(mode=...)` requires tool functions to match the mode (sync vs async).
- Only `FunctionTool` is executed locally by the core `Agent`.
- Worker outputs are expected to be valid JSON when `json_reply=True`.
- `AgentSystemContext.get_memory_bank()` raises if `memory_bank` is not initialized.

---

## References

- Core LLM routing: `cortex/LLM.py`, `cortex/backend.py`
- Agents: `cortex/agent.py`
- Tools: `cortex/tool.py`
- LLM-powered functions: `cortex/LLMFunc.py`
- Agent System: `cortex/agent_system/*`
- Examples: `examples/`
