from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from inspect import iscoroutine
from typing import Any, Callable, Optional

from cortex.agent import Agent, Tool
from cortex.workflow import WorkflowAgent

from .adapters import AgentNodeAdapter, WorkflowNodeAdapter
from .defaults import DefaultGatewayNode, DefaultManagerNode
from .models import DelegationBrief
from .node import BuiltNode


RuntimeFactory = Callable[[Any], Any]


@dataclass(slots=True)
class NodeBuilder:
    name: str
    role: str
    runtime_factory: RuntimeFactory
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    async def build_node(self, *, context: Any, installed_tools: Optional[list[Tool]] = None) -> BuiltNode:
        runtime = self._invoke_runtime_factory(context=context, installed_tools=installed_tools or [])
        if iscoroutine(runtime):
            runtime = await runtime

        if isinstance(runtime, BuiltNode):
            return runtime
        if hasattr(runtime, "run_brief"):
            return BuiltNode(name=self.name, role=self.role, runtime=runtime)
        if isinstance(runtime, Agent):
            return BuiltNode(
                name=self.name,
                role=self.role,
                runtime=AgentNodeAdapter(name=self.name, role=self.role, agent=runtime),
            )
        if isinstance(runtime, WorkflowAgent):
            return BuiltNode(
                name=self.name,
                role=self.role,
                runtime=WorkflowNodeAdapter(name=self.name, role=self.role, workflow=runtime),
            )
        raise TypeError(
            f"Unsupported runtime for node '{self.name}': {type(runtime)!r}. "
            "Expected Agent, WorkflowAgent, BuiltNode, or an ExecutionNode-compatible object."
        )

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

    def install(self) -> Tool:
        async def func(args: dict[str, Any], context: Any):
            brief_value = args["brief"]
            brief = brief_value if isinstance(brief_value, DelegationBrief) else DelegationBrief.from_dict(brief_value)
            node = await self.build_node(context=context)
            result = await node.run_brief(brief, context=context)
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
                        "description": "Structured delegation brief for the target node",
                    }
                },
                "required": ["brief"],
                "additionalProperties": False,
            },
        )

    @property
    def tool_name(self) -> str:
        return self.name.lower().replace(" ", "_") + "_node"

    @staticmethod
    def _as_tool_list(tools: Optional[list[Tool]]) -> list[Tool]:
        return list(tools or [])


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
        workflow: WorkflowAgent | Callable[[Any], WorkflowAgent],
        description: Optional[str] = None,
    ) -> "SpecialistNodeBuilder":
        def runtime_factory(*, context: Any) -> WorkflowAgent:
            if callable(workflow) and not isinstance(workflow, WorkflowAgent):
                built = workflow(context)
                if iscoroutine(built):
                    return built
                return built
            return workflow

        return cls(
            name=name,
            runtime_factory=runtime_factory,
            description=description or f"{specialty} workflow specialist '{name}'",
            specialty=specialty,
        )
