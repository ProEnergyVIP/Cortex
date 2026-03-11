from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from .task_adapters import AgentTaskRunnerAdapter, WorkflowTaskRunnerAdapter, normalize_task_result as _normalize_task_result
from .task_builders import RuntimeFactory, TaskRunnerBuilderBase
from .task_node import BuiltTaskRunner, TaskRunner
from .task_orchestration import should_escalate, synthesize_task_results
from .task_types import TaskBrief, TaskResult

ChildBriefBuilder = Callable[..., TaskBrief]
ResultSynthesizer = Callable[..., TaskResult]
TaskTextBuilder = str | Callable[[str], str]
TaskMetadataBuilder = dict[str, Any] | Callable[[str], dict[str, Any]]


@dataclass(slots=True)
class TaskRunnerBuilder(TaskRunnerBuilderBase):
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
    ) -> "TaskRunnerBuilder":
        own_tools = cls._as_tool_list(tools)

        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Agent:
            return Agent(
                llm=llm,
                tools=[*own_tools, *installed_tools],
                sys_prompt=prompt,
                context=context,
                mode="async",
            )

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Task runner '{name}'",
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        role: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        description: Optional[str] = None,
    ) -> "TaskRunnerBuilder":
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
            description=description or f"Workflow task runner '{name}'",
        )

    @classmethod
    def create_runtime(
        cls,
        *,
        name: str,
        role: str,
        runtime: Any,
        description: Optional[str] = None,
    ) -> "TaskRunnerBuilder":
        if callable(runtime) and not isinstance(runtime, (Agent, WorkflowAgent, BuiltTaskRunner)):
            runtime_factory = runtime
        else:
            def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> Any:
                return runtime

        return cls(
            name=name,
            role=role,
            runtime_factory=runtime_factory,
            description=description or f"Runtime-backed task runner '{name}'",
        )


def build_task_brief(
    *,
    from_runner: str,
    to_runner: str,
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
) -> TaskBrief:
    return TaskBrief.new(
        conversation_id=conversation_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        from_node=from_runner,
        to_node=to_runner,
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


def build_child_task_brief(
    *,
    parent_brief: TaskBrief,
    child_name: str,
    handoff_kind: str,
    assigned_task: str,
    current_understanding: str,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    dependencies: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskBrief:
    merged_constraints = dict(parent_brief.constraints)
    if constraints:
        merged_constraints.update(constraints)

    return TaskBrief.new(
        conversation_id=parent_brief.conversation_id,
        task_id=f"{parent_brief.task_id}.{child_name.lower().replace(' ', '_')}",
        parent_task_id=parent_brief.task_id,
        from_node=parent_brief.to_node,
        to_node=child_name,
        handoff_level=handoff_kind,
        original_user_request=parent_brief.original_user_request,
        original_request_summary=parent_brief.original_request_summary,
        caller_understanding=current_understanding,
        scoped_task=assigned_task,
        expected_output=expected_output,
        constraints=merged_constraints,
        dependencies=list(parent_brief.dependencies) + list(dependencies or []),
        priority=parent_brief.priority,
        confidence=parent_brief.confidence,
        escalation_if_below=parent_brief.escalation_if_below,
        metadata={**parent_brief.metadata, **(metadata or {})},
    )


async def run_task_runner(
    runner: TaskRunner | BuiltTaskRunner | TaskRunnerBuilder | Agent | WorkflowAgent,
    *,
    brief: TaskBrief,
    context: Any,
    role: Optional[str] = None,
    name: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> TaskResult:
    built = await _build_task_runner(
        runner,
        context=context,
        installed_tools=installed_tools or [],
        role=role,
        name=name,
    )
    return await built.run_brief(brief, context=context)


async def build_task_runner(
    runner: TaskRunner | BuiltTaskRunner | TaskRunnerBuilder | Agent | WorkflowAgent,
    *,
    context: Any,
    role: Optional[str] = None,
    name: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> BuiltTaskRunner:
    return await _build_task_runner(
        runner,
        context=context,
        installed_tools=installed_tools or [],
        role=role,
        name=name,
    )


def task_runner_tool(
    runner: TaskRunner | BuiltTaskRunner | TaskRunnerBuilder | Agent | WorkflowAgent,
    *,
    name: str,
    role: str,
    tool_name: Optional[str] = None,
    description: Optional[str] = None,
    installed_tools: Optional[list[Tool]] = None,
) -> Tool:
    async def func(args: dict[str, Any], context: Any):
        brief_value = args["brief"]
        brief = brief_value if isinstance(brief_value, TaskBrief) else TaskBrief.from_dict(brief_value)
        result = await run_task_runner(
            runner,
            brief=brief,
            context=context,
            role=role,
            name=name,
            installed_tools=installed_tools,
        )
        return result.to_dict()

    return Tool(
        name=tool_name or name.lower().replace(" ", "_") + "_node",
        func=func,
        description=description or f"Delegates work to {name}",
        parameters={
            "type": "object",
            "properties": {
                "brief": {
                    "type": "object",
                    "description": "Structured task brief for the target runner",
                }
            },
            "required": ["brief"],
            "additionalProperties": False,
        },
    )


def normalize_task_result(
    value: Any,
    *,
    brief: TaskBrief,
    role: str,
    fallback_name: str,
) -> TaskResult:
    return _normalize_task_result(value, brief=brief, role=role, fallback_name=fallback_name)


async def run_task_tools(
    tools: Iterable[Tool],
    *,
    parent_brief: TaskBrief,
    context: Any,
    handoff_kind: str,
    assigned_task_builder: TaskTextBuilder,
    understanding_builder: TaskTextBuilder,
    expected_output: Optional[dict[str, Any]] = None,
    metadata_builder: Optional[TaskMetadataBuilder] = None,
    role: str = "worker",
    execute_in_parallel: bool = False,
    tool_name_to_runner_name: Optional[Callable[[Tool], str]] = None,
) -> list[TaskResult]:
    active_tools = list(tools)

    async def _run_tool(tool: Tool) -> TaskResult:
        child_name = (
            tool_name_to_runner_name(tool)
            if tool_name_to_runner_name is not None
            else tool.name.removesuffix("_node").replace("_", " ").title()
        )
        child_brief = build_child_task_brief(
            parent_brief=parent_brief,
            child_name=child_name,
            handoff_kind=handoff_kind,
            assigned_task=_resolve_text_builder(assigned_task_builder, child_name),
            current_understanding=_resolve_text_builder(understanding_builder, child_name),
            expected_output=expected_output,
            metadata=_resolve_metadata_builder(metadata_builder, child_name),
        )
        raw_result = await tool.async_run({"brief": child_brief.to_dict()}, context, None)
        return normalize_task_result(raw_result, brief=child_brief, role=role, fallback_name=child_name)

    if execute_in_parallel:
        return list(await asyncio.gather(*[_run_tool(tool) for tool in active_tools]))
    return [await _run_tool(tool) for tool in active_tools]


def summarize_task_results(
    *,
    brief: TaskBrief,
    role: str,
    from_runner: Optional[str],
    child_results: Iterable[TaskResult],
    summary: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskResult:
    return synthesize_task_results(
        brief=brief,
        role=role,
        from_node=from_runner,
        child_results=child_results,
        summary=summary,
        metadata=metadata,
    )


async def _build_task_runner(
    runner: TaskRunner | BuiltTaskRunner | TaskRunnerBuilder | Agent | WorkflowAgent,
    *,
    context: Any,
    installed_tools: list[Tool],
    role: Optional[str],
    name: Optional[str],
) -> BuiltTaskRunner:
    if isinstance(runner, TaskRunnerBuilder):
        return await runner.build(context=context, installed_tools=installed_tools)
    if isinstance(runner, BuiltTaskRunner):
        return runner
    if hasattr(runner, "run_brief") and hasattr(runner, "name") and hasattr(runner, "role"):
        return BuiltTaskRunner(name=getattr(runner, "name"), role=getattr(runner, "role"), runtime=runner)
    if isinstance(runner, Agent):
        runner_name = name or getattr(runner, "name", None) or "Agent Runner"
        runner_role = role or "runner"
        return BuiltTaskRunner(
            name=runner_name,
            role=runner_role,
            runtime=AgentTaskRunnerAdapter(name=runner_name, role=runner_role, agent=runner),
        )
    if isinstance(runner, WorkflowAgent):
        runner_name = name or runner.name
        runner_role = role or "runner"
        return BuiltTaskRunner(
            name=runner_name,
            role=runner_role,
            runtime=WorkflowTaskRunnerAdapter(name=runner_name, role=runner_role, workflow=runner),
        )
    raise TypeError(
        f"Unsupported runner type: {type(runner)!r}. Expected TaskRunnerBuilder, BuiltTaskRunner, Agent, WorkflowAgent, or a run_brief-compatible runtime."
    )


def _resolve_text_builder(value: TaskTextBuilder, child_name: str) -> str:
    return value(child_name) if callable(value) else value


def _resolve_metadata_builder(
    value: Optional[TaskMetadataBuilder],
    child_name: str,
) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    return value(child_name) if callable(value) else dict(value)


__all__ = [
    "TaskBrief",
    "TaskResult",
    "TaskRunner",
    "BuiltTaskRunner",
    "TaskRunnerBuilder",
    "RuntimeFactory",
    "ChildBriefBuilder",
    "ResultSynthesizer",
    "TaskTextBuilder",
    "TaskMetadataBuilder",
    "build_task_brief",
    "build_child_task_brief",
    "build_task_runner",
    "run_task_runner",
    "task_runner_tool",
    "normalize_task_result",
    "run_task_tools",
    "summarize_task_results",
    "should_escalate",
]
