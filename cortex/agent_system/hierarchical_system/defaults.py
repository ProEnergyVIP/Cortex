from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from cortex.agent import Tool

from ..composition import run_task_tools, summarize_task_results
from .models import DelegationBrief, NodeResult
from .orchestration import (
    build_manager_brief,
    build_routing_decision,
    build_specialist_brief,
    should_escalate,
)


DepartmentSelector = Callable[[DelegationBrief, list[dict[str, str]]], list[str]]
SpecialistSelector = Callable[[DelegationBrief, list[dict[str, str]]], list[str]]


@dataclass(slots=True)
class DefaultGatewayNode:
    name: str
    department_catalog: list[dict[str, str]]
    department_tools: list[Tool]
    confidence_threshold: float = 0.6
    selector: Optional[DepartmentSelector] = None

    @property
    def role(self) -> str:
        return "gateway"

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        department_names = self._select_departments(brief)
        if not department_names:
            return NodeResult.escalate(
                brief=brief,
                role="gateway",
                from_node=self.name,
                summary="I need clarification before I can route this request to the right department.",
                confidence=0.0,
                escalation_reason="no department matched the request",
                metadata={
                    "clarification_needed": True,
                    "available_departments": [item["name"] for item in self.department_catalog],
                },
                output={
                    "final_answer": "I need clarification before I can route this request to the right department.",
                    "clarification_question": self._clarification_question(),
                },
            )

        routing = build_routing_decision(
            brief=brief,
            departments=department_names,
            rationale=self._routing_rationale(brief, department_names),
            confidence=self._routing_confidence(department_names),
            clarification_question=None,
        )

        available_department_tools = [
            tool
            for department_name in department_names
            if (tool := self._find_department_tool(department_name)) is not None
        ]
        child_results = await run_task_tools(
            available_department_tools,
            parent_brief=brief,
            context=context,
            handoff_kind="gateway_to_manager",
            assigned_task_builder=lambda department_name: self._manager_task(brief, department_name),
            understanding_builder=lambda department_name: self._manager_understanding(brief, department_name),
            expected_output={"type": "department_synthesis"},
            metadata_builder=lambda department_name: {"routing_decision": routing.to_dict(), "department": department_name},
            role="manager",
        )
        for department_name in department_names:
            if self._find_department_tool(department_name) is not None:
                continue
            manager_brief = build_manager_brief(
                parent_brief=brief,
                manager_name=department_name,
                department_name=department_name,
                scoped_task=self._manager_task(brief, department_name),
                caller_understanding=self._manager_understanding(brief, department_name),
                expected_output={"type": "department_synthesis"},
                metadata={"routing_decision": routing.to_dict()},
            )
            child_results.append(
                NodeResult.escalate(
                    brief=manager_brief,
                    role="manager",
                    from_node=department_name,
                    summary=f"Department '{department_name}' is configured but no executable tool is available.",
                    confidence=0.0,
                    escalation_reason="missing department tool",
                )
            )

        synthesized = summarize_task_results(
            brief=brief,
            role="gateway",
            from_runner=self.name,
            child_results=child_results,
            summary=self._gateway_summary(child_results),
            metadata={"routing_decision": routing.to_dict()},
        )
        synthesized.output.setdefault("routing_decision", routing.to_dict())
        synthesized.output["final_answer"] = synthesized.summary
        if should_escalate(synthesized, threshold=self.confidence_threshold):
            synthesized.output.setdefault("clarification_question", self._clarification_question())
        return synthesized

    def _select_departments(self, brief: DelegationBrief) -> list[str]:
        if self.selector is not None:
            selected = list(dict.fromkeys(self.selector(brief, self.department_catalog)))
            return [item for item in selected if self._find_department_tool(item) is not None]

        text = self._searchable_text(brief)
        scored: list[tuple[int, str]] = []
        for department in self.department_catalog:
            haystack = " ".join(
                part.lower()
                for part in [department.get("name", ""), department.get("description", "")]
                if part
            )
            score = sum(1 for token in haystack.split() if token and token in text)
            if score > 0:
                scored.append((score, department["name"]))

        if scored:
            best = max(score for score, _ in scored)
            return [name for score, name in scored if score == best]
        if len(self.department_catalog) == 1:
            return [self.department_catalog[0]["name"]]
        return []

    def _find_department_tool(self, department_name: str) -> Optional[Tool]:
        normalized = department_name.lower().replace(" ", "_") + "_node"
        for tool in self.department_tools:
            if getattr(tool, "name", None) == normalized:
                return tool
        return None

    def _searchable_text(self, brief: DelegationBrief) -> str:
        return " ".join(
            [
                brief.original_user_request.lower(),
                brief.original_request_summary.lower(),
                brief.caller_understanding.lower(),
                brief.scoped_task.lower(),
            ]
        )

    def _routing_rationale(self, brief: DelegationBrief, departments: list[str]) -> str:
        return f"Selected departments {departments} based on the request summary '{brief.original_request_summary}'."

    def _routing_confidence(self, departments: list[str]) -> float:
        if not departments:
            return 0.0
        if len(departments) == 1:
            return 0.82
        return 0.68

    def _manager_task(self, brief: DelegationBrief, department_name: str) -> str:
        return f"Handle the {department_name} portion of the request and return a synthesized department result."

    def _manager_understanding(self, brief: DelegationBrief, department_name: str) -> str:
        return (
            f"The gateway believes the request has a meaningful {department_name} component. "
            f"Original summary: {brief.original_request_summary}"
        )

    def _gateway_summary(self, child_results: list[NodeResult]) -> str:
        if not child_results:
            return "No department results were produced."
        if len(child_results) == 1:
            return child_results[0].summary
        joined = " ".join(result.summary for result in child_results)
        return f"Integrated department review: {joined}"

    def _clarification_question(self) -> str:
        return "Could you clarify which department or business area this request belongs to, or provide a little more context?"


@dataclass(slots=True)
class DefaultManagerNode:
    name: str
    specialist_catalog: list[dict[str, str]]
    specialist_tools: list[Tool]
    confidence_threshold: float = 0.6
    selector: Optional[SpecialistSelector] = None
    execute_in_parallel: bool = True

    @property
    def role(self) -> str:
        return "manager"

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        specialist_names = self._select_specialists(brief)
        if not specialist_names:
            return NodeResult.escalate(
                brief=brief,
                role="manager",
                from_node=self.name,
                summary="No specialist matched this department task well enough to proceed.",
                confidence=0.0,
                escalation_reason="no specialist matched the scoped task",
                metadata={"available_specialists": [item["name"] for item in self.specialist_catalog]},
            )

        available_specialist_tools = [
            tool
            for specialist_name in specialist_names
            if (tool := self._find_specialist_tool(specialist_name)) is not None
        ]
        child_results = await run_task_tools(
            available_specialist_tools,
            parent_brief=brief,
            context=context,
            handoff_kind="manager_to_worker",
            assigned_task_builder=lambda specialist_name: self._specialist_task(brief, specialist_name),
            understanding_builder=lambda specialist_name: self._specialist_understanding(brief, specialist_name),
            expected_output={"type": "specialist_result"},
            role="worker",
            execute_in_parallel=self.execute_in_parallel,
        )
        for specialist_name in specialist_names:
            if self._find_specialist_tool(specialist_name) is not None:
                continue
            specialist_brief = build_specialist_brief(
                parent_brief=brief,
                specialist_name=specialist_name,
                scoped_task=self._specialist_task(brief, specialist_name),
                caller_understanding=self._specialist_understanding(brief, specialist_name),
                expected_output={"type": "specialist_result"},
            )
            child_results.append(
                NodeResult.escalate(
                    brief=specialist_brief,
                    role="worker",
                    from_node=specialist_name,
                    summary=f"Specialist '{specialist_name}' is configured but no executable tool is available.",
                    confidence=0.0,
                    escalation_reason="missing specialist tool",
                )
            )

        synthesized = summarize_task_results(
            brief=brief,
            role="manager",
            from_runner=self.name,
            child_results=child_results,
            summary=self._manager_summary(child_results),
            metadata={"selected_specialists": specialist_names},
        )
        if should_escalate(synthesized, threshold=self.confidence_threshold):
            synthesized.metadata.setdefault("manager_needs_escalation", True)
        return synthesized

    def _select_specialists(self, brief: DelegationBrief) -> list[str]:
        if self.selector is not None:
            selected = list(dict.fromkeys(self.selector(brief, self.specialist_catalog)))
            return [item for item in selected if self._find_specialist_tool(item) is not None]

        text = self._searchable_text(brief)
        scored: list[tuple[int, str]] = []
        for specialist in self.specialist_catalog:
            haystack = " ".join(
                part.lower()
                for part in [specialist.get("name", ""), specialist.get("description", "")]
                if part
            )
            score = sum(1 for token in haystack.split() if token and token in text)
            if score > 0:
                scored.append((score, specialist["name"]))

        if scored:
            best = max(score for score, _ in scored)
            return [name for score, name in scored if score == best]
        return [item["name"] for item in self.specialist_catalog]

    def _find_specialist_tool(self, specialist_name: str) -> Optional[Tool]:
        normalized = specialist_name.lower().replace(" ", "_") + "_node"
        for tool in self.specialist_tools:
            if getattr(tool, "name", None) == normalized:
                return tool
        return None

    def _searchable_text(self, brief: DelegationBrief) -> str:
        return " ".join(
            [
                brief.original_user_request.lower(),
                brief.original_request_summary.lower(),
                brief.caller_understanding.lower(),
                brief.scoped_task.lower(),
            ]
        )

    def _specialist_task(self, brief: DelegationBrief, specialist_name: str) -> str:
        return f"Handle the specialist portion of this request as {specialist_name} and return a concise structured result."

    def _specialist_understanding(self, brief: DelegationBrief, specialist_name: str) -> str:
        return (
            f"The manager believes {specialist_name} is relevant to this task. "
            f"Department task summary: {brief.scoped_task}"
        )

    def _manager_summary(self, child_results: list[NodeResult]) -> str:
        if not child_results:
            return "No specialist results were produced."
        if len(child_results) == 1:
            return child_results[0].summary
        joined = " ".join(result.summary for result in child_results)
        return f"Department synthesis: {joined}"
