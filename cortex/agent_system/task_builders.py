from __future__ import annotations

from dataclasses import dataclass, field
from inspect import iscoroutine, signature
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from .task_adapters import AgentTaskRunnerAdapter, WorkflowTaskRunnerAdapter
from .task_node import BuiltTaskRunner
from .task_types import TaskBrief


RuntimeFactory = Callable[[Any], Any]


@dataclass(slots=True)
class TaskRunnerBuilderBase:
    name: str
    role: str
    runtime_factory: RuntimeFactory
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    async def build_task_runner(self, *, context: Any, installed_tools: Optional[list[Tool]] = None) -> BuiltTaskRunner:
        runtime = self._invoke_runtime_factory(context=context, installed_tools=installed_tools or [])
        if iscoroutine(runtime):
            runtime = await runtime

        if isinstance(runtime, BuiltTaskRunner):
            return runtime
        if hasattr(runtime, "run_brief"):
            return BuiltTaskRunner(name=self.name, role=self.role, runtime=runtime)
        if isinstance(runtime, Agent):
            return BuiltTaskRunner(
                name=self.name,
                role=self.role,
                runtime=AgentTaskRunnerAdapter(name=self.name, role=self.role, agent=runtime),
            )
        if isinstance(runtime, WorkflowAgent):
            return BuiltTaskRunner(
                name=self.name,
                role=self.role,
                runtime=WorkflowTaskRunnerAdapter(name=self.name, role=self.role, workflow=runtime),
            )
        raise TypeError(
            f"Unsupported runtime for task runner '{self.name}': {type(runtime)!r}. "
            "Expected Agent, WorkflowAgent, BuiltTaskRunner, or a run_brief-compatible object."
        )

    async def build(self, *, context: Any, installed_tools: Optional[list[Tool]] = None) -> BuiltTaskRunner:
        return await self.build_task_runner(context=context, installed_tools=installed_tools)

    def install(self) -> Tool:
        async def func(args: dict[str, Any], context: Any):
            brief_value = args["brief"]
            brief = brief_value if isinstance(brief_value, TaskBrief) else TaskBrief.from_dict(brief_value)
            runner = await self.build_task_runner(context=context)
            result = await runner.run_brief(brief, context=context)
            return result.to_dict()

        return Tool(
            name=self.tool_name,
            func=func,
            description=self.description or f"Delegates work to {self.name}",
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

    def as_tool(self) -> Tool:
        return self.install()

    def with_metadata(self, **metadata: Any) -> "TaskRunnerBuilderBase":
        self.metadata.update(metadata)
        return self

    @property
    def tool_name(self) -> str:
        return self.name.lower().replace(" ", "_") + "_node"

    def _invoke_runtime_factory(self, *, context: Any, installed_tools: list[Tool]) -> Any:
        sig = signature(self.runtime_factory)
        kwargs: dict[str, Any] = {}
        if "context" in sig.parameters:
            kwargs["context"] = context
        elif sig.parameters:
            return self.runtime_factory(context)
        if "installed_tools" in sig.parameters:
            kwargs["installed_tools"] = installed_tools
        if "child_tools" in sig.parameters:
            kwargs["child_tools"] = installed_tools
        return self.runtime_factory(**kwargs)

    @staticmethod
    def _as_tool_list(tools: Optional[list[Tool]]) -> list[Tool]:
        return list(tools or [])

    @staticmethod
    def _resolve_workflow_runtime(
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        *,
        context: Any,
        installed_tools: Optional[list[Tool]] = None,
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
            kwargs["installed_tools"] = installed_tools or []
        if "child_tools" in sig.parameters:
            kwargs["child_tools"] = installed_tools or []
        return workflow(**kwargs)


__all__ = ["RuntimeFactory", "TaskRunnerBuilderBase"]
