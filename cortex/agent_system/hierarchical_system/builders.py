from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from ..task_builders import TaskRunnerBuilderBase
from .defaults import DefaultGatewayNode, DefaultManagerNode
from .node import BuiltNode


@dataclass(slots=True)
class NodeBuilder(TaskRunnerBuilderBase):
    async def build_node(self, *, context: Any, installed_tools: Optional[list[Tool]] = None) -> BuiltNode:
        return await self.build_task_runner(context=context, installed_tools=installed_tools)


@dataclass(slots=True)
class GatewayNodeBuilder(NodeBuilder):
    role: str = "gateway"

    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        llm: Any,
        prompt: str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
    ) -> "GatewayNodeBuilder":
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
            runtime_factory=runtime_factory,
            description=description or f"Gateway node '{name}'",
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        description: Optional[str] = None,
    ) -> "GatewayNodeBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> WorkflowAgent:
            return cls._resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=installed_tools,
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"Gateway workflow node '{name}'",
        )

    @classmethod
    def create_default(
        cls,
        *,
        name: str = "Gateway",
        description: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ) -> "GatewayNodeBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> DefaultGatewayNode:
            department_catalog = [
                {
                    "name": tool.name.removesuffix("_node").replace("_", " ").title(),
                    "description": getattr(tool, "description", "") or "",
                }
                for tool in installed_tools
            ]
            return DefaultGatewayNode(
                name=name,
                department_catalog=department_catalog,
                department_tools=installed_tools,
                confidence_threshold=confidence_threshold,
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or "Default gateway runtime with department routing and synthesis.",
        )


@dataclass(slots=True)
class DepartmentManagerBuilder(NodeBuilder):
    role: str = "manager"
    department: Optional[str] = None

    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        department: str,
        llm: Any,
        prompt: str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
    ) -> "DepartmentManagerBuilder":
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
            runtime_factory=runtime_factory,
            description=description or f"{department} manager '{name}'",
            department=department,
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        department: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        description: Optional[str] = None,
    ) -> "DepartmentManagerBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> WorkflowAgent:
            return cls._resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=installed_tools,
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"{department} manager workflow '{name}'",
            department=department,
        )

    @classmethod
    def create_default(
        cls,
        *,
        name: str,
        department: str,
        description: Optional[str] = None,
        confidence_threshold: float = 0.6,
        execute_in_parallel: bool = True,
    ) -> "DepartmentManagerBuilder":
        def runtime_factory(*, context: Any, installed_tools: list[Tool]) -> DefaultManagerNode:
            specialist_catalog = [
                {
                    "name": tool.name.removesuffix("_node").replace("_", " ").title(),
                    "description": getattr(tool, "description", "") or "",
                }
                for tool in installed_tools
            ]
            return DefaultManagerNode(
                name=name,
                specialist_catalog=specialist_catalog,
                specialist_tools=installed_tools,
                confidence_threshold=confidence_threshold,
                execute_in_parallel=execute_in_parallel,
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"Default manager runtime for the {department} department.",
            department=department,
        )


@dataclass(slots=True)
class SpecialistNodeBuilder(NodeBuilder):
    role: str = "worker"
    specialty: Optional[str] = None

    @classmethod
    def create_agent(
        cls,
        *,
        name: str,
        specialty: str,
        llm: Any,
        prompt: str,
        tools: Optional[list[Tool]] = None,
        description: Optional[str] = None,
    ) -> "SpecialistNodeBuilder":
        own_tools = cls._as_tool_list(tools)

        def runtime_factory(*, context: Any) -> Agent:
            return Agent(
                llm=llm,
                tools=own_tools,
                sys_prompt=prompt,
                context=context,
                mode="async",
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"{specialty} specialist '{name}'",
            specialty=specialty,
        )

    @classmethod
    def create_workflow(
        cls,
        *,
        name: str,
        specialty: str,
        workflow: WorkflowAgent | Callable[..., WorkflowAgent],
        description: Optional[str] = None,
    ) -> "SpecialistNodeBuilder":
        def runtime_factory(*, context: Any) -> WorkflowAgent:
            return cls._resolve_workflow_runtime(
                workflow,
                context=context,
                installed_tools=None,
            )

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"{specialty} workflow specialist '{name}'",
            specialty=specialty,
        )
