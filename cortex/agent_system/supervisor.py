"""Helpers for building supervisor agents around a set of worker agents.

The public API is intentionally small: `create_supervisor(...)` either:

- builds a normal `Agent` supervisor from an `llm`, or
- delegates workflow construction to a user-provided `workflow_builder`.

In both cases, each worker is exposed to the supervisor as a tool that accepts a
single delegated `task` string. This keeps the handoff contract simple and lets
the supervisor rewrite the request specifically for each worker.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from asyncio import iscoroutine
from inspect import signature
from typing import Any, Awaitable, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent


@dataclass
class _SupervisorWorker:
    """Normalized internal representation of one worker managed by a supervisor."""

    worker: Callable[..., Any]
    name: Optional[str] = None
    description: Optional[str] = None
    resolved_worker: Any = None


def _default_supervisor_prompt(name: str, worker_descriptions: list[str], instructions: Optional[str] = None) -> str:
    """Build the default system prompt for agent-based supervisors."""

    worker_block = "\n".join(f"- {description}" for description in worker_descriptions) if worker_descriptions else "- No workers configured"
    prompt = f"""You are {name}, a supervisor agent coordinating a team of worker agents.

Your job is to decide whether to answer directly or delegate work to one or more workers.

Available workers:
{worker_block}

Delegation rules:
- Understand each worker's role before delegating.
- When delegating, write a clear task for that worker in your own words.
- You may rewrite, restructure, and specialize the request for the chosen worker.
- Do not forward the raw user message blindly when a more precise task would help.
- Call multiple workers when their work is independent and useful in parallel.
- Synthesize worker outputs into one final answer for the user.
- If no worker is needed, answer directly.
"""
    if instructions:
        prompt = f"{prompt}\nAdditional instructions:\n{instructions}"
    return prompt


def _normalize_supervisor_workers(workers: list[Any]) -> list[_SupervisorWorker]:
    """Normalize raw worker specs into `_SupervisorWorker` objects."""

    return [_normalize_supervisor_worker_spec(worker) for worker in workers]


def _normalize_supervisor_worker_spec(worker: Any) -> _SupervisorWorker:
    """Normalize one worker spec.

    Accepted shapes:
    - a callable returning a runnable / agent / workflow-like object
    - a dict containing `worker`, plus optional `name` and `description`
    - an already normalized `_SupervisorWorker`
    """

    if isinstance(worker, _SupervisorWorker):
        return worker
    if isinstance(worker, dict):
        worker_builder = worker.get("worker")
        if worker_builder is None:
            raise ValueError("Supervisor worker specs must include a 'worker' entry")
        if not callable(worker_builder):
            raise ValueError("Supervisor worker specs must provide a callable 'worker' builder")
        return _SupervisorWorker(
            worker=worker_builder,
            name=worker.get("name"),
            description=worker.get("description"),
        )
    if not callable(worker):
        raise ValueError("create_supervisor(...) workers must be callables that return a runnable")
    return _SupervisorWorker(worker=worker)


def _normalize_supervisor_tool_name(name: Optional[str], index: int) -> str:
    """Create a stable tool name for a worker."""

    raw_name = name or f"worker_{index + 1}"
    return raw_name.strip().lower().replace(" ", "_")


def _describe_supervisor_worker(spec: _SupervisorWorker, tool_name: str) -> str:
    """Create a human-readable description used in the default supervisor prompt."""

    worker_name = spec.name or getattr(spec.worker, "name", None) or getattr(spec.worker, "__name__", None) or tool_name
    description = spec.description or f"Worker agent '{worker_name}' available through tool '{tool_name}'."
    return f"{worker_name}: {description} (tool: {tool_name})"


async def _resolve_supervisor_worker(
    spec: _SupervisorWorker,
    *,
    context: Any = None,
    usage: Any = None,
    parent: Any = None,
) -> Any:
    """Build and cache a worker runnable the first time it is needed."""

    if spec.resolved_worker is not None:
        return spec.resolved_worker

    built = _call_with_supported_kwargs(
        spec.worker,
        context=context,
        usage=usage,
        parent=parent,
    )
    if iscoroutine(built):
        built = await built
    spec.resolved_worker = built
    return built


async def _invoke_worker(
    worker: Any,
    task: Any,
    *,
    context: Any = None,
    usage: Any = None,
    parent: Any = None,
) -> Any:
    """Invoke a resolved worker directly via its async_ask interface."""

    if hasattr(worker, "async_ask"):
        return await worker.async_ask(task, context=context)
    if callable(worker):
        result = worker(task)
        if iscoroutine(result):
            result = await result
        return result
    raise TypeError(f"Worker {worker!r} does not support async_ask or direct invocation")


def _build_supervisor_worker_tool(
    spec: _SupervisorWorker,
    index: int,
    *,
    context: Any = None,
) -> Tool:
    """Wrap a worker as a tool that accepts a single delegated `task` string."""

    tool_name = _normalize_supervisor_tool_name(
        spec.name or getattr(spec.worker, "name", None) or getattr(spec.worker, "__name__", None),
        index,
    )
    description = spec.description or f"Delegate a task to {spec.name or getattr(spec.worker, 'name', getattr(spec.worker, '__name__', tool_name))}."

    async def _run_worker(args, tool_context=None, agent=None):
        task = args["task"]
        worker = await _resolve_supervisor_worker(
            spec,
            context=tool_context if tool_context is not None else context,
            usage=getattr(tool_context, "usage", None) if tool_context is not None else None,
            parent=agent,
        )
        result = await _invoke_worker(
            worker,
            task,
            context=tool_context if tool_context is not None else context,
            usage=getattr(tool_context, "usage", None) if tool_context is not None else None,
            parent=agent,
        )
        return result

    return Tool(
        name=tool_name,
        func=_run_worker,
        description=description,
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The delegated task or message for this worker.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    )


def _build_worker_tools(
    worker_tools: Optional[list[Callable[..., Tool]]],
    *,
    context: Any = None,
    usage: Any = None,
) -> list[Tool]:
    """Build extra supervisor-facing tools from builder callables."""

    built_tools: list[Tool] = []
    for builder in worker_tools or []:
        if not callable(builder):
            raise ValueError("create_supervisor(..., worker_tools=...) expects callables that return Tool objects")
        built = _call_with_supported_kwargs(
            builder,
            context=context,
            usage=usage,
        )
        if iscoroutine(built):
            raise TypeError("create_supervisor(..., worker_tools=...) builders must return Tool objects directly, not coroutines")
        if not isinstance(built, Tool):
            raise TypeError("create_supervisor(..., worker_tools=...) builders must return Tool objects")
        built_tools.append(built)
    return built_tools


def _validate_duplicate_tool_names(tools: list[Tool], *, source: str) -> None:
    """Raise when a tool collection contains duplicate names."""

    tool_names = [tool.name for tool in tools]
    duplicate_tool_names = sorted({name for name in tool_names if tool_names.count(name) > 1})
    if duplicate_tool_names:
        raise ValueError(
            f"{source} generated duplicate tool names: " + ", ".join(duplicate_tool_names)
        )


def build_supervisor_tools(
    workers: list[Any],
    *,
    worker_tools: Optional[list[Callable[..., Tool]]] = None,
    context: Any = None,
    usage: Any = None,
) -> tuple[list[_SupervisorWorker], list[Tool], list[str]]:
    """Build worker tools and prompt descriptions from raw worker specs.

    Returns a tuple of:
    - normalized worker specs
    - all supervisor-facing worker tools
    - human-readable worker descriptions for prompts/docs
    """

    worker_specs = _normalize_supervisor_workers(workers)
    generated_worker_tools = [
        _build_supervisor_worker_tool(
            spec,
            index,
            context=context,
        )
        for index, spec in enumerate(worker_specs)
    ]
    extra_worker_tools = _build_worker_tools(worker_tools, context=context, usage=usage)
    combined_worker_tools = [*generated_worker_tools, *extra_worker_tools]
    _validate_duplicate_tool_names(
        combined_worker_tools,
        source="create_supervisor(...)",
    )
    worker_descriptions = [
        _describe_supervisor_worker(spec, generated_worker_tools[index].name)
        for index, spec in enumerate(worker_specs)
    ]
    return worker_specs, combined_worker_tools, worker_descriptions


def _get_supervisor_worker_key(spec: _SupervisorWorker, index: int) -> str:
    """Return the canonical worker key used in workflow delegation payloads."""

    return _normalize_supervisor_tool_name(
        spec.name or getattr(spec.worker, "name", None) or getattr(spec.worker, "__name__", None),
        index,
    )


async def invoke_supervisor_workers(
    delegations: list[dict[str, Any]],
    *,
    workers: list[_SupervisorWorker],
    context: Any = None,
    usage: Any = None,
    parent: Any = None,
) -> list[dict[str, Any]]:
    """Execute a list of delegated worker tasks and collect structured results.

    This helper is primarily used by custom workflow supervisors. Each delegation
    item must contain:

    - `worker`: normalized worker tool name
    - `task`: the rewritten task for that worker
    """

    worker_map = {
        _get_supervisor_worker_key(spec, index): spec
        for index, spec in enumerate(workers)
    }
    invalid_workers = [item["worker"] for item in delegations if item.get("worker") not in worker_map]
    if invalid_workers:
        raise ValueError(f"Supervisor workflow plan referenced unknown workers: {', '.join(invalid_workers)}")

    async def _run_delegation(item):
        worker_key = item["worker"]
        task = item["task"]
        worker = await _resolve_supervisor_worker(
            worker_map[worker_key],
            context=context,
            usage=usage,
            parent=parent,
        )
        output = await _invoke_worker(
            worker,
            task,
            context=context,
            usage=usage,
            parent=parent,
        )
        worker_name = getattr(worker, "name", None) or worker_key
        return {
            "worker": worker_key,
            "task": task,
            "output": output,
            "runnable_name": worker_name,
        }

    return await asyncio.gather(*[_run_delegation(item) for item in delegations]) if delegations else []

def _call_with_supported_kwargs(func, **kwargs):
    """Call a function with only the keyword arguments it declares."""

    sig = signature(func)
    accepted_kwargs = {}

    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            accepted_kwargs = kwargs
            break
        if name in kwargs:
            accepted_kwargs[name] = kwargs[name]

    return func(**accepted_kwargs)


async def create_supervisor(
    *,
    name: str,
    workers: list[Any],
    worker_tools: Optional[list[Callable[..., Tool]]] = None,
    llm: Any = None,
    sys_prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    workflow_builder: Optional[Callable[..., WorkflowAgent | Awaitable[WorkflowAgent]]] = None,
    context: Any = None,
    memory: Any = None,
    usage: Any = None,
) -> Agent | WorkflowAgent:
    """Create a supervisor over a set of lazily built worker runnables.

    There are two supported modes:

    - Agent supervisor mode: pass `llm` and optionally `sys_prompt`, `instructions`,
      and extra `tools`. This returns a normal `Agent`.
    - Workflow supervisor mode: pass `workflow_builder`. The builder is called after
      worker tools are prepared and must return a `WorkflowAgent` (sync or async).

    Worker inputs must be callables that lazily return a runnable, or dict specs like:

    ```python
    {
        "worker": build_research_agent,
        "name": "Research Worker",  # optional
        "description": "Researches facts and gathers information.",
    }
    ```

    Extra `worker_tools` can also be supplied as callables that return `Tool` objects.
    Those tools are appended after the tools generated from the lazy worker builders.

    In workflow mode, the builder is called with a filtered subset of:

    - `tools` / `worker_tools`: the prepared worker tools
    - `name`: supervisor name
    - `context`: shared context
    - `memory`: optional memory object for supervisor runtime
    - `usage`: shared usage tracker
    - `workers`: normalized internal worker specs
    - `worker_descriptions`: prompt-friendly worker descriptions
    """

    if not workers:
        raise ValueError("create_supervisor(...) requires at least one worker")
    if llm is None and workflow_builder is None:
        raise ValueError("create_supervisor(...) requires either llm or workflow_builder")
    if llm is not None and workflow_builder is not None:
        raise ValueError("create_supervisor(...) accepts either llm or workflow_builder, not both")
    if workflow_builder is not None and not callable(workflow_builder):
        raise ValueError("create_supervisor(..., workflow_builder=...) requires a callable builder")

    worker_specs, worker_tools, worker_descriptions = build_supervisor_tools(
        workers,
        worker_tools=worker_tools,
        context=context,
        usage=usage,
    )
    if llm is not None:
        supervisor_tools = [*(tools or []), *worker_tools]
        _validate_duplicate_tool_names(
            supervisor_tools,
            source="create_supervisor(...)",
        )
        supervisor_agent = Agent(
            name=name,
            llm=llm,
            tools=supervisor_tools,
            sys_prompt=sys_prompt or _default_supervisor_prompt(name, worker_descriptions, instructions=instructions),
            context=context,
            memory=memory,
            json_reply=False,
            mode="async",
        )
        supervisor_agent.usage = usage
        return supervisor_agent

    built = _call_with_supported_kwargs(
        workflow_builder,
        tools=worker_tools,
        worker_tools=worker_tools,
        name=name,
        context=context,
        memory=memory,
        usage=usage,
        workers=worker_specs,
        worker_descriptions=worker_descriptions,
    )
    if iscoroutine(built):
        built = await built
    if isinstance(built, WorkflowAgent):
        return built
    raise TypeError("create_supervisor(..., workflow_builder=...) must return a WorkflowAgent")


__all__ = ["create_supervisor"]
