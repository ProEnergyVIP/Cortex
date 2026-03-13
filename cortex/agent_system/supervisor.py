from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent, runnable_node, workflow
from cortex.workflow.engine import WorkflowStateProtocol
from cortex.workflow.runtime import invoke_runnable


@dataclass
class SupervisorWorker:
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


def _normalize_supervisor_worker_spec(worker: Any) -> SupervisorWorker:
    if isinstance(worker, SupervisorWorker):
        return worker
    if isinstance(worker, dict):
        runnable = worker.get("worker")
        if runnable is None:
            raise ValueError("Supervisor worker specs must include a 'worker' entry")
        return SupervisorWorker(
            worker=runnable,
            name=worker.get("name"),
            description=worker.get("description"),
        )
    return SupervisorWorker(worker=worker)


def _normalize_supervisor_tool_name(name: Optional[str], index: int) -> str:
    raw_name = name or f"worker_{index + 1}"
    return raw_name.strip().lower().replace(" ", "_")


def _describe_supervisor_worker(spec: SupervisorWorker, tool_name: str) -> str:
    worker_name = spec.name or getattr(spec.worker, "name", None) or tool_name
    description = spec.description or f"Worker agent '{worker_name}' available through tool '{tool_name}'."
    return f"{worker_name}: {description} (tool: {tool_name})"


def _build_supervisor_worker_tool(
    spec: SupervisorWorker,
    index: int,
    *,
    tool_name_builder: Optional[Callable[[SupervisorWorker, int], str]] = None,
    tool_description_builder: Optional[Callable[[SupervisorWorker, str], str]] = None,
    context: Any = None,
) -> Tool:
    tool_name = tool_name_builder(spec, index) if tool_name_builder is not None else _normalize_supervisor_tool_name(spec.name or getattr(spec.worker, "name", None), index)
    description = tool_description_builder(spec, tool_name) if tool_description_builder is not None else (spec.description or f"Delegate a task to {spec.name or getattr(spec.worker, 'name', tool_name)}.")

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


def create_supervisor(
    *,
    name: str,
    workers: list[Any],
    llm,
    kind: str = "agent",
    sys_prompt: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[list[Tool]] = None,
    context: Any = None,
    usage: Any = None,
    max_steps: int = 8,
    state_type: Optional[type[WorkflowStateProtocol]] = None,
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None,
    worker_tool_name_builder: Optional[Callable[[SupervisorWorker, int], str]] = None,
    worker_tool_description_builder: Optional[Callable[[SupervisorWorker, str], str]] = None,
) -> Agent | WorkflowAgent:
    if kind not in {"agent", "workflow"}:
        raise ValueError("create_supervisor(..., kind=...) must be 'agent' or 'workflow'")
    if not workers:
        raise ValueError("create_supervisor(...) requires at least one worker")

    worker_specs = [_normalize_supervisor_worker_spec(worker) for worker in workers]
    worker_tools = [
        _build_supervisor_worker_tool(
            spec,
            index,
            tool_name_builder=worker_tool_name_builder,
            tool_description_builder=worker_tool_description_builder,
            context=context,
        )
        for index, spec in enumerate(worker_specs)
    ]
    worker_descriptions = [
        _describe_supervisor_worker(spec, worker_tools[index].name)
        for index, spec in enumerate(worker_specs)
    ]

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

    if kind == "agent":
        return supervisor_agent

    return workflow(
        name=name,
        nodes=[
            runnable_node(
                "supervisor",
                runnable=supervisor_agent,
                is_final=True,
            )
        ],
        context=context,
        usage=usage,
        max_steps=max_steps,
        state_type=state_type,
        state_factory=state_factory,
    )


__all__ = ["SupervisorWorker", "create_supervisor"]
