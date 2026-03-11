from __future__ import annotations

from typing import Any, Optional

from ..task_executor_orchestration import (
    build_routing_decision,
    should_escalate,
    synthesize_task_results,
)
from ..task_models import child_task_id
from .models import DelegationBrief, NodeResult, RoutingDecision


def build_manager_brief(
    *,
    parent_brief: DelegationBrief,
    manager_name: str,
    department_name: str,
    scoped_task: str,
    caller_understanding: str,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> DelegationBrief:
    return DelegationBrief.new(
        conversation_id=parent_brief.conversation_id,
        task_id=_child_task_id(parent_brief.task_id, manager_name),
        parent_task_id=parent_brief.task_id,
        from_node=parent_brief.to_node,
        to_node=manager_name,
        handoff_level="gateway_to_manager",
        original_user_request=parent_brief.original_user_request,
        original_request_summary=parent_brief.original_request_summary,
        caller_understanding=caller_understanding,
        scoped_task=scoped_task,
        expected_output=expected_output,
        constraints=_merge_constraints(parent_brief.constraints, constraints, {"department": department_name}),
        dependencies=list(parent_brief.dependencies),
        priority=parent_brief.priority,
        confidence=parent_brief.confidence,
        escalation_if_below=parent_brief.escalation_if_below,
        metadata={**parent_brief.metadata, **(metadata or {}), "department": department_name},
    )



def build_specialist_brief(
    *,
    parent_brief: DelegationBrief,
    specialist_name: str,
    scoped_task: str,
    caller_understanding: str,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    dependencies: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> DelegationBrief:
    return DelegationBrief.new(
        conversation_id=parent_brief.conversation_id,
        task_id=_child_task_id(parent_brief.task_id, specialist_name),
        parent_task_id=parent_brief.task_id,
        from_node=parent_brief.to_node,
        to_node=specialist_name,
        handoff_level="manager_to_worker",
        original_user_request=parent_brief.original_user_request,
        original_request_summary=parent_brief.original_request_summary,
        caller_understanding=caller_understanding,
        scoped_task=scoped_task,
        expected_output=expected_output,
        constraints=_merge_constraints(parent_brief.constraints, constraints),
        dependencies=list(parent_brief.dependencies) + list(dependencies or []),
        priority=parent_brief.priority,
        confidence=parent_brief.confidence,
        escalation_if_below=parent_brief.escalation_if_below,
        metadata={**parent_brief.metadata, **(metadata or {})},
    )



def synthesize_results(
    *,
    brief: DelegationBrief,
    role: str,
    from_node: Optional[str],
    child_results: list[NodeResult],
    summary: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> NodeResult:
    return synthesize_task_results(
        brief=brief,
        role=role,
        from_node=from_node,
        child_results=child_results,
        summary=summary,
        metadata=metadata,
    )



def _merge_constraints(*constraint_sets: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in constraint_sets:
        if value:
            merged.update(value)
    return merged



def _child_task_id(parent_task_id: str, node_name: str) -> str:
    return child_task_id(parent_task_id, node_name)


__all__ = [
    "build_manager_brief",
    "build_specialist_brief",
    "build_routing_decision",
    "should_escalate",
    "synthesize_results",
    "RoutingDecision",
]
