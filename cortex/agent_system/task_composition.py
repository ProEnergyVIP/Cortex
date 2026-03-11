from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from .task_executor_adapters import (
    AgentTaskRunnerAdapter,
    WorkflowTaskRunnerAdapter,
    normalize_task_result as _normalize_task_result,
)
from .task_executor_builders import RuntimeFactory, TaskExecutorBuilderBase
from .task_executor import BuiltTaskExecutor, TaskExecutor
from .task_executor_orchestration import should_escalate, synthesize_task_results as _synthesize_task_results
from .task_models import TaskDesc, TaskResult, child_task_id

TaskTextFactory = str | Callable[[str], str]
TaskMetadataFactory = dict[str, Any] | Callable[[str], dict[str, Any]]


@dataclass(slots=True)
class TaskExecutorBuilder(TaskExecutorBuilderBase):
    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        role: str,
        llm: Any,
        prompt: str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
    ) -> "TaskExecutorBuilder":
        own_tools = cls._as_tool_list(tools)

        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Agent:
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
            description=description or f"Task executor '{name}'",
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        role: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        description: Optional[str] = None,
    ) -> "TaskExecutorBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> WorkflowAgent:
            return cls._resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=installed_tools,
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Workflow task executor '{name}'",
        )

    @classmethod
    def create_runtime(
        cls,
        *,
        name: str,
        role: str,
        runtime: Any,
        description: Optional[str] = None,
    ) -> "TaskExecutorBuilder":
        if callable(runtime) and not isinstance(runtime, (Agent, WorkflowAgent, BuiltTaskExecutor)):
            runtime_factory = runtime
        else:
            def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Any:
                return runtime

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Runtime-backed task executor '{name}'",
        )


def create_task_desc(
    *,
    from_executor: str,
    to_executor: str,
    handoff_kind: str,
    original_request: str,
    request_summary: str,
    current_understanding: str,
    assigned_task: str,
    conversation_id: Optional[str] = None,
    task_id: Optional[str] = None,
    parent_task_id: Optional[str] = None,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    dependencies: Optional[list[str]] = None,
    priority: str = "medium",
    confidence: float = 1.0,
    escalate_if_below: float = 0.6,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskDesc:
    return TaskDesc.new(
        conversation_id=conversation_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        from_node=from_executor,
        to_node=to_executor,
        handoff_level=handoff_kind,
        original_user_request=original_request,
        original_request_summary=request_summary,
        caller_understanding=current_understanding,
        scoped_task=assigned_task,
        expected_output=expected_output,
        constraints=constraints,
        dependencies=dependencies,
        priority=priority,
        confidence=confidence,
        escalation_if_below=escalate_if_below,
        metadata=metadata,
    )


def create_child_task_desc(
    *,
    parent_desc: TaskDesc,
    child_name: str,
    handoff_kind: str,
    assigned_task: str,
    current_understanding: str,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    dependencies: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskDesc:
    merged_constraints = dict(parent_desc.constraints)
    if constraints:
        merged_constraints.update(constraints)

    return TaskDesc.new(
        conversation_id=parent_desc.conversation_id,
        task_id=child_task_id(parent_desc.task_id, child_name),
        parent_task_id=parent_desc.task_id,
        from_node=parent_desc.to_node,
        to_node=child_name,
        handoff_level=handoff_kind,
        original_user_request=parent_desc.original_user_request,
        original_request_summary=parent_desc.original_request_summary,
        caller_understanding=current_understanding,
        scoped_task=assigned_task,
        expected_output=expected_output,
        constraints=merged_constraints,
        dependencies=list(parent_desc.dependencies) + list(dependencies or []),
        priority=parent_desc.priority,
        confidence=parent_desc.confidence,
        escalation_if_below=parent_desc.escalation_if_below,
        metadata={**parent_desc.metadata, **(metadata or {})},
    )


async def execute_task_executor(
    executor: TaskExecutor | BuiltTaskExecutor | TaskExecutorBuilderBase | Agent | WorkflowAgent,
    *,
    desc: TaskDesc,
    context: Any,
    role: Optional[str] = None,
    name: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> TaskResult:
    built = await resolve_task_executor(
        executor,
        context=context,
        role=role,
        name=name,
        installed_tools=installed_tools,
    )
    return await built.run_task(desc, context=context)


async def resolve_task_executor(
    executor: TaskExecutor | BuiltTaskExecutor | TaskExecutorBuilderBase | Agent | WorkflowAgent,
    *,
    context: Any,
    role: Optional[str] = None,
    name: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> BuiltTaskExecutor:
    return await _resolve_task_executor(
        executor,
        context=context,
        installed_tools=installed_tools or [],
        role=role,
        name=name,
    )


def create_task_tool(
    executor: TaskExecutor | BuiltTaskExecutor | TaskExecutorBuilderBase | Agent | WorkflowAgent,
    *,
    name: str,
    role: str,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> Tool:
    async def func(args: dict[str, Any], context: Any):
        desc_value = args["desc"]
        desc = desc_value if isinstance(desc_value, TaskDesc) else TaskDesc.from_dict(desc_value)
        result = await execute_task_executor(
            executor,
            desc=desc,
            context=context,
            role=role,
            name=name,
            installed_tools=installed_tools,
        )
        return result.to_dict()

    return Tool(
        name=tool_name or name.lower().replace(" ", "_") + "_node",
        func=func,
        description=description or f"Delegates task execution to {name}",
        parameters={
            "type": "object",
            "properties": {
                "desc": {
                    "type": "object",
                    "description": "Structured task description for the target executor",
                }
            },
            "required": ["desc"],
            "additionalProperties": False,
        },
    )


def coerce_task_result(
    value: Any,
    *,
    desc: TaskDesc,
    role: str,
    fallback_name: str,
) -> TaskResult:
    return _normalize_task_result(value, brief=desc, role=role, fallback_name=fallback_name)


async def execute_task_tools(
    tools: Iterable[Tool],
    *,
    parent_desc: TaskDesc,
    context: Any,
    handoff_kind: str,
    assigned_task_builder: TaskTextFactory,
    understanding_builder: TaskTextFactory,
    expected_output: Optional[dict[str, Any]] = None,
    metadata_builder: Optional[TaskMetadataFactory] = None,
    role: str = "worker",
    execute_in_parallel: bool = False,
    tool_name_to_executor_name: Optional[Callable[[Tool], str]] = None,
) -> list[TaskResult]:
    active_tools = list(tools)

    async def _run_tool(tool: Tool) -> TaskResult:
        child_name = (
            tool_name_to_executor_name(tool)
            if tool_name_to_executor_name is not None
            else tool.name.removesuffix("_node").replace("_", " ").title()
        )
        child_desc = create_child_task_desc(
            parent_desc=parent_desc,
            child_name=child_name,
            handoff_kind=handoff_kind,
            assigned_task=_resolve_text_builder(assigned_task_builder, child_name),
            current_understanding=_resolve_text_builder(understanding_builder, child_name),
            expected_output=expected_output,
            metadata=_resolve_metadata_builder(metadata_builder, child_name),
        )
        raw_result = await tool.async_run({"desc": child_desc.to_dict()}, context, None)
        return coerce_task_result(raw_result, desc=child_desc, role=role, fallback_name=child_name)

    if execute_in_parallel:
        return list(await asyncio.gather(*[_run_tool(tool) for tool in active_tools]))
    return [await _run_tool(tool) for tool in active_tools]


def synthesize_task_results(
    *,
    desc: TaskDesc,
    role: str,
    from_executor: Optional[str],
    child_results: Iterable[TaskResult],
    summary: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskResult:
    return _synthesize_task_results(
        brief=desc,
        role=role,
        from_node=from_executor,
        child_results=child_results,
        summary=summary,
        metadata=metadata,
    )


async def _resolve_task_executor(
    executor: TaskExecutor | BuiltTaskExecutor | TaskExecutorBuilderBase | Agent | WorkflowAgent,
    *,
    context: Any,
    installed_tools: list[Tool],
    role: Optional[str],
    name: Optional[str],
) -> BuiltTaskExecutor:
    if isinstance(executor, TaskExecutorBuilderBase):
        return await executor.build(context=context, installed_tools=installed_tools)
    if isinstance(executor, BuiltTaskExecutor):
        return executor
    if hasattr(executor, "run_task") and hasattr(executor, "name") and hasattr(executor, "role"):
        return BuiltTaskExecutor(name=getattr(executor, "name"), role=getattr(executor, "role"), runtime=executor)
    if isinstance(executor, Agent):
        executor_name = name or getattr(executor, "name", None) or "Agent Executor"
        executor_role = role or "executor"
        return BuiltTaskExecutor(
            name=executor_name,
            role=executor_role,
            runtime=AgentTaskRunnerAdapter(name=executor_name, role=executor_role, agent=executor),
        )
    if isinstance(executor, WorkflowAgent):
        executor_name = name or executor.name
        executor_role = role or "executor"
        return BuiltTaskExecutor(
            name=executor_name,
            role=executor_role,
            runtime=WorkflowTaskRunnerAdapter(name=executor_name, role=executor_role, workflow=executor),
        )
    raise TypeError(
        f"Unsupported executor type: {type(executor)!r}. Expected TaskExecutorBuilderBase, BuiltTaskExecutor, "
        "Agent, WorkflowAgent, or a run_task-compatible runtime."
    )


def _resolve_text_builder(value: TaskTextFactory, child_name: str) -> str:
    return value(child_name) if callable(value) else value


def _resolve_metadata_builder(
    value: Optional[TaskMetadataFactory],
    child_name: str,
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return value(child_name) if callable(value) else dict(value)


__all__ = [
    "TaskDesc",
    "TaskResult",
    "TaskExecutor",
    "BuiltTaskExecutor",
    "TaskExecutorBuilderBase",
    "TaskExecutorBuilder",
    "RuntimeFactory",
    "TaskTextFactory",
    "TaskMetadataFactory",
    "create_task_desc",
    "create_child_task_desc",
    "resolve_task_executor",
    "execute_task_executor",
    "create_task_tool",
    "coerce_task_result",
    "execute_task_tools",
    "synthesize_task_results",
    "should_escalate",
]
