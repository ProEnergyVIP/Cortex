from __future__ import annotations

from dataclasses import dataclass
from inspect import iscoroutine, signature
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from ..task_composition import TaskExecutorBuilder
from .models import TaskCoordinatorSpec, TaskWorkerSpec
from .prompts import TASK_COORDINATOR_PROMPT, TASK_WORKER_PROMPT


@dataclass(slots=True)
class TaskCoordinatorBuilder(TaskExecutorBuilder):
    @classmethod
    def compose_prompt(cls, *, name: str, task_desc: str, worker_catalog: str) -> str:
        return TASK_COORDINATOR_PROMPT.format(
            name=name,
            task_desc=task_desc,
            worker_catalog=worker_catalog,
        )

    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        llm: Any,
        prompt_builder: Callable[..., str] | str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
        role: str = "coordinator",
    ) -> "TaskCoordinatorBuilder":
        own_tools = cls._as_tool_list(tools)

        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Agent:
            prompt = cls.compose_prompt(
                name=name,
                task_desc=_resolve_prompt_builder(prompt_builder, context=context),
                worker_catalog=_worker_catalog_from_tools(installed_tools),
            )
            return Agent(
                llm=llm,
                tools=[*own_tools, *installed_tools],
                sys_prompt=prompt,
                context=context,
                json_reply=True,
                mode="async",
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Task coordinator '{name}'",
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        prompt_builder: Callable[..., str] | str | None = None,
        description: Optional[str] = None,
        role: str = "coordinator",
    ) -> "TaskCoordinatorBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> WorkflowAgent:
            prompt = None
            if prompt_builder is not None:
                prompt = cls.compose_prompt(
                    name=name,
                    task_desc=_resolve_prompt_builder(prompt_builder, context=context),
                    worker_catalog=_worker_catalog_from_tools(installed_tools),
                )
            return _resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=installed_tools,
                prompt=prompt,
                worker_catalog=_worker_catalog_from_tools(installed_tools),
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Task coordinator workflow '{name}'",
        )

    def as_spec(self) -> TaskCoordinatorSpec:
        return TaskCoordinatorSpec(
            name=self.name,
            role=self.role,
            executor=self,
            description=self.description,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class TaskWorkerBuilder(TaskExecutorBuilder):
    @classmethod
    def compose_prompt(cls, *, name: str, task_desc: str) -> str:
        return TASK_WORKER_PROMPT.format(name=name, task_desc=task_desc)

    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        llm: Any,
        prompt_builder: Callable[..., str] | str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
        role: str = "worker",
    ) -> "TaskWorkerBuilder":
        own_tools = cls._as_tool_list(tools)

        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Agent:
            prompt = cls.compose_prompt(name=name, task_desc=_resolve_prompt_builder(prompt_builder, context=context))
            return Agent(
                llm=llm,
                tools=[*own_tools, *installed_tools],
                sys_prompt=prompt,
                context=context,
                json_reply=True,
                mode="async",
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Task worker '{name}'",
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        prompt_builder: Callable[..., str] | str | None = None,
        description: Optional[str] = None,
        role: str = "worker",
    ) -> "TaskWorkerBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> WorkflowAgent:
            prompt = None
            if prompt_builder is not None:
                prompt = cls.compose_prompt(
                    name=name,
                    task_desc=_resolve_prompt_builder(prompt_builder, context=context),
                )
            return _resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=installed_tools,
                prompt=prompt,
                task_prompt=prompt,
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Task worker workflow '{name}'",
        )

    def as_spec(self) -> TaskWorkerSpec:
        return TaskWorkerSpec(
            name=self.name,
            role=self.role,
            executor=self,
            description=self.description,
            metadata=dict(self.metadata),
        )


def _resolve_prompt_builder(prompt_builder: Callable[..., str] | str, *, context: Any) -> str:
    if not callable(prompt_builder):
        return prompt_builder
    try:
        sig = signature(prompt_builder)
    except (TypeError, ValueError):
        value = prompt_builder()
    else:
        if "context" in sig.parameters:
            value = prompt_builder(context=context)
        elif "ctx" in sig.parameters:
            value = prompt_builder(ctx=context)
        elif sig.parameters:
            value = prompt_builder(context)
        else:
            value = prompt_builder()
    if iscoroutine(value):
        raise TypeError("Task coordinator and worker prompt builders must return prompt strings synchronously.")
    if not isinstance(value, str):
        raise TypeError(
            f"Task coordinator and worker prompt builders must return strings, got {type(value)!r}."
        )
    return value


def _worker_catalog_from_tools(installed_tools: list[Tool]) -> str:
    if not installed_tools:
        return "No workers are configured."
    lines = []
    for tool in installed_tools:
        description = getattr(tool, "description", None) or "No description provided."
        lines.append(f"- {tool.name}: {description}")
    return "\n".join(lines)


def _resolve_workflow_runtime(
    workflow: WorkflowAgent | Callable[..., WorkflowAgent],
    *,
    context: Any,
    installed_tools: list[Tool],
    prompt: str | None = None,
    task_prompt: str | None = None,
    worker_catalog: str | None = None,
) -> WorkflowAgent | Any:
    if not callable(workflow) or isinstance(workflow, WorkflowAgent):
        return workflow

    sig = signature(workflow)
    kwargs: dict[str, Any] = {}
    if "context" in sig.parameters:
        kwargs["context"] = context
    elif sig.parameters:
        return workflow(context)
    if "installed_tools" in sig.parameters:
        kwargs["installed_tools"] = installed_tools
    if "child_tools" in sig.parameters:
        kwargs["child_tools"] = installed_tools
    if prompt is not None and "prompt" in sig.parameters:
        kwargs["prompt"] = prompt
    if task_prompt is not None and "task_prompt" in sig.parameters:
        kwargs["task_prompt"] = task_prompt
    if worker_catalog is not None and "worker_catalog" in sig.parameters:
        kwargs["worker_catalog"] = worker_catalog
    return workflow(**kwargs)
