from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent
from cortex.workflow.runtime import adapt_runnable, invoke_runnable


@dataclass
class _SupervisorWorker:
    worker: Any
    name: Optional[str] = None
    description: Optional[str] = None


def _default_supervisor_prompt(name: str, worker_descriptions: list[str], instructions: Optional[str] = None) -> str:
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
    return [_normalize_supervisor_worker_spec(worker) for worker in workers]


def _normalize_supervisor_worker_spec(worker: Any) -> _SupervisorWorker:
    if isinstance(worker, _SupervisorWorker):
        return worker
    if isinstance(worker, dict):
        runnable = worker.get("worker")
        if runnable is None:
            raise ValueError("Supervisor worker specs must include a 'worker' entry")
        return _SupervisorWorker(
            worker=runnable,
            name=worker.get("name"),
            description=worker.get("description"),
        )
    return _SupervisorWorker(worker=worker)


def _normalize_supervisor_tool_name(name: Optional[str], index: int) -> str:
    raw_name = name or f"worker_{index + 1}"
    return raw_name.strip().lower().replace(" ", "_")


def _describe_supervisor_worker(spec: _SupervisorWorker, tool_name: str) -> str:
    worker_name = spec.name or getattr(spec.worker, "name", None) or tool_name
    description = spec.description or f"Worker agent '{worker_name}' available through tool '{tool_name}'."
    return f"{worker_name}: {description} (tool: {tool_name})"


def _build_supervisor_worker_tool(
    spec: _SupervisorWorker,
    index: int,
    *,
    context: Any = None,
) -> Tool:
    tool_name = _normalize_supervisor_tool_name(spec.name or getattr(spec.worker, "name", None), index)
    description = spec.description or f"Delegate a task to {spec.name or getattr(spec.worker, 'name', tool_name)}."

    async def _run_worker(args, tool_context=None, agent=None):
        task = args["task"]
        invocation = await invoke_runnable(
            spec.worker,
            task,
            context=tool_context if tool_context is not None else context,
            usage=getattr(tool_context, "usage", None) if tool_context is not None else None,
            parent=agent,
        )
        return invocation.output

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


def build_supervisor_tools(
    workers: list[Any],
    *,
    context: Any = None,
) -> tuple[list[_SupervisorWorker], list[Tool], list[str]]:
    worker_specs = _normalize_supervisor_workers(workers)
    worker_tools = [
        _build_supervisor_worker_tool(
            spec,
            index,
            context=context,
        )
        for index, spec in enumerate(worker_specs)
    ]
    worker_descriptions = [
        _describe_supervisor_worker(spec, worker_tools[index].name)
        for index, spec in enumerate(worker_specs)
    ]
    return worker_specs, worker_tools, worker_descriptions


def _get_supervisor_worker_key(spec: _SupervisorWorker, index: int) -> str:
    return _normalize_supervisor_tool_name(spec.name or getattr(spec.worker, "name", None), index)


async def invoke_supervisor_workers(
    delegations: list[dict[str, Any]],
    *,
    workers: list[_SupervisorWorker],
    context: Any = None,
    usage: Any = None,
    parent: Any = None,
) -> list[dict[str, Any]]:
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
        try:
            invocation = await invoke_runnable(
                worker_map[worker_key].worker,
                task,
                context=context,
                usage=usage,
                parent=parent,
            )
        except TypeError as exc:
            if "does not support async_run" not in str(exc):
                raise
            adapted_worker = await adapt_runnable(
                worker_map[worker_key].worker,
                context=context,
                usage=usage,
                parent=parent,
            )
            output = await adapted_worker.async_ask(task, context=context, usage=usage, parent=parent)
            invocation = type("FallbackInvocation", (), {
                "output": output,
                "runnable_name": adapted_worker.name,
            })()
        return {
            "worker": worker_key,
            "task": task,
            "output": invocation.output,
            "runnable_name": invocation.runnable_name,
        }

    return await asyncio.gather(*[_run_delegation(item) for item in delegations]) if delegations else []

def _call_with_supported_kwargs(func, **kwargs):
    sig = signature(func)
    accepted_kwargs = {}

    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            accepted_kwargs = kwargs
            break
        if name in kwargs:
            accepted_kwargs[name] = kwargs[name]

    return func(**accepted_kwargs)


def create_supervisor(
    *,
    name: str,
    workers: list[Any],
    llm: Any = None,
    sys_prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    workflow_builder: Optional[Callable[..., WorkflowAgent]] = None,
    context: Any = None,
    usage: Any = None,
) -> Agent | WorkflowAgent:
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
        context=context,
    )
    if llm is not None:
        supervisor_agent = Agent(
            name=name,
            llm=llm,
            tools=[*(tools or []), *worker_tools],
            sys_prompt=sys_prompt or _default_supervisor_prompt(name, worker_descriptions, instructions=instructions),
            context=context,
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
        usage=usage,
        workers=worker_specs,
        worker_descriptions=worker_descriptions,
    )
    if isinstance(built, WorkflowAgent):
        return built
    raise TypeError("create_supervisor(..., workflow_builder=...) must return a WorkflowAgent")


__all__ = ["create_supervisor"]
