from __future__ import annotations

from typing import Any, Optional

from cortex.agent import Tool

from ..task_composition import create_task_tool, resolve_task_executor
from ..core.system import AgentSystem
from .builders import GatewayNodeBuilder
from .models import DelegationBrief, DepartmentSpec, HandoffRecord, NodeResult


class HierarchicalAgentSystem(AgentSystem):
    def __init__(
        self,
        gateway_builder: GatewayNodeBuilder,
        departments: list[DepartmentSpec],
        context: Optional[Any] = None,
    ):
        super().__init__(context)
        self._gateway_builder = gateway_builder
        self._departments = departments
        self._gateway_node = None

    @classmethod
    def create(
        cls,
        *,
        departments: list[DepartmentSpec],
        context: Optional[Any] = None,
        gateway_builder: Optional[GatewayNodeBuilder] = None,
        gateway_name: str = "Gateway",
    ) -> "HierarchicalAgentSystem":
        return cls(
            gateway_builder=gateway_builder or GatewayNodeBuilder.create_default(name=gateway_name),
            departments=departments,
            context=context,
        )

    async def get_agent(self):
        return await self.get_gateway_node()

    async def get_gateway_node(self):
        if self._gateway_node is not None:
            return self._gateway_node

        installed_tools = [await self._build_department_tool(department) for department in self._departments]
        self._gateway_node = await resolve_task_executor(
            self._gateway_builder,
            context=self._context,
            installed_tools=installed_tools,
            role=self._gateway_builder.role,
            name=self._gateway_builder.name,
        )
        return self._gateway_node

    async def async_ask(self, messages: Any) -> Any:
        user_request = self._stringify_user_input(messages)
        gateway = await self.get_gateway_node()
        payload = {
            "request": user_request,
            "task": "Understand the request, route to departments if needed, and produce the final user-facing answer.",
            "context": {
                "handoff_kind": "user_to_gateway",
                "summary": user_request,
                "understanding": "Direct user request requiring gateway triage and coordination.",
            },
        }
        brief = DelegationBrief.new(
            from_node="user",
            to_node=gateway.name,
            payload=payload,
            metadata={"source": "HierarchicalAgentSystem.async_ask"},
        )
        await self._log_handoff(
            HandoffRecord(
                conversation_id=brief.conversation_id,
                task_id=brief.task_id,
                parent_task_id=brief.parent_task_id,
                from_node=brief.from_node,
                to_node=brief.to_node,
                direction="downward",
                summary=brief.scoped_task,
                confidence=brief.confidence,
                status="delegated",
                metadata={"handoff_level": brief.handoff_level},
            )
        )
        result = await gateway.run_task(brief, context=self._context)
        await self._log_result(result)
        if result.status == "needs_escalation" and result.confidence < 0.6:
            return result.summary
        return result.output.get("final_answer", result.summary)

    async def _build_department_tool(self, department: DepartmentSpec) -> Tool:
        manager = department.manager
        specialists = [specialist.install() for specialist in department.specialists]
        built_manager = await resolve_task_executor(
            manager,
            context=self._context,
            installed_tools=specialists,
            role=manager.role,
            name=manager.name,
        )

        class LoggedDepartmentRunner:
            name = built_manager.name
            role = built_manager.role

            async def run_task(inner_self, desc: DelegationBrief, *, context: Any) -> NodeResult:
                brief = desc
                await self._log_handoff(
                    HandoffRecord(
                        conversation_id=brief.conversation_id,
                        task_id=brief.task_id,
                        parent_task_id=brief.parent_task_id,
                        from_node=brief.from_node,
                        to_node=built_manager.name,
                        direction="downward",
                        summary=brief.scoped_task,
                        confidence=brief.confidence,
                        status="delegated",
                        metadata={"department": department.name, "handoff_level": brief.handoff_level},
                    )
                )
                result = await built_manager.run_task(brief, context=context)
                await self._log_result(result)
                return result

        tool = create_task_tool(
            LoggedDepartmentRunner(),
            name=manager.name,
            role=manager.role,
            tool_name=manager.tool_name,
            description=manager.description or department.description,
        )
        return tool

    def department_names(self) -> list[str]:
        return [department.name for department in self._departments]

    def describe_hierarchy(self) -> dict[str, Any]:
        return {
            "gateway": self._gateway_builder.name,
            "departments": [
                {
                    "name": department.name,
                    "description": department.description,
                    "manager": department.manager.name,
                    "specialists": [specialist.name for specialist in department.specialists],
                }
                for department in self._departments
            ],
        }

    async def _log_handoff(self, record: HandoffRecord) -> None:
        whiteboard = getattr(self._context, "whiteboard", None)
        if whiteboard is None:
            return
        await whiteboard.post(
            sender=record.from_node,
            channel="hierarchical_handoffs",
            content=record.to_dict(),
            thread=record.conversation_id,
        )

    async def _log_result(self, result: NodeResult) -> None:
        await self._log_handoff(
            HandoffRecord(
                conversation_id=result.conversation_id,
                task_id=result.task_id,
                parent_task_id=result.parent_task_id,
                from_node=result.from_node,
                to_node=result.to_node,
                direction="upward",
                summary=result.summary,
                confidence=result.confidence,
                status=result.status,
                metadata={
                    "role": result.role,
                    "blockers": list(result.blockers),
                    "assumptions": list(result.assumptions),
                    "escalation_reason": result.escalation_reason,
                },
            )
        )

    @staticmethod
    def _stringify_user_input(messages: Any) -> str:
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            return "\n".join(HierarchicalAgentSystem._stringify_user_input(item) for item in messages)
        content = getattr(messages, "content", None)
        if content is not None:
            return str(content)
        return str(messages)
