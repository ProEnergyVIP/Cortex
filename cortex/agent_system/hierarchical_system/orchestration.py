from __future__ import annotations

from typing import Any, Iterable, Optional

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
    child_results: Iterable[NodeResult],
    summary: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> NodeResult:
    items = list(child_results)
    if not items:
        return NodeResult.escalate(
            brief=brief,
            role=role,  # type: ignore[arg-type]
            from_node=from_node,
            summary="No child results were available to synthesize.",
            confidence=0.0,
            escalation_reason="missing child results",
            metadata=metadata or {},
        )

    confidence = min(item.confidence for item in items)
    blockers = [blocker for item in items for blocker in item.blockers]
    assumptions = [assumption for item in items for assumption in item.assumptions]
    child_summaries = [
        {
            "from_node": item.from_node,
            "status": item.status,
            "summary": item.summary,
            "confidence": item.confidence,
        }
        for item in items
    ]

    if any(item.status in {"needs_escalation", "failed"} for item in items) or confidence < brief.escalation_if_below:
        return NodeResult.escalate(
            brief=brief,
            role=role,  # type: ignore[arg-type]
            from_node=from_node,
            summary=summary or _default_escalation_summary(items),
            confidence=confidence,
            blockers=blockers,
            escalation_reason="one or more child results require escalation",
            metadata={**(metadata or {}), "child_summaries": child_summaries},
            output={"child_results": [item.to_dict() for item in items]},
        )

    combined_output = {
        "child_results": [item.to_dict() for item in items],
        "synthesized_output": {item.from_node: item.output for item in items},
    }
    return NodeResult.complete(
        brief=brief,
        role=role,  # type: ignore[arg-type]
        from_node=from_node,
        summary=summary or _default_summary(items),
        output=combined_output,
        confidence=confidence,
        assumptions=assumptions,
        blockers=blockers,
        child_summaries=child_summaries,
        metadata=metadata or {},
    )



def build_routing_decision(
    *,
    brief: DelegationBrief,
    departments: list[str],
    rationale: str,
    confidence: float,
    clarification_question: Optional[str] = None,
) -> RoutingDecision:
    mode = "direct"
    if len(departments) == 1:
        mode = "single_department"
    elif len(departments) > 1:
        mode = "multi_department"
    return RoutingDecision(
        conversation_id=brief.conversation_id,
        task_id=brief.task_id,
        mode=mode,
        departments=departments,
        rationale=rationale,
        confidence=confidence,
        clarification_question=clarification_question,
    )



def should_escalate(result: NodeResult, *, threshold: float = 0.6) -> bool:
    return result.status in {"needs_escalation", "failed"} or result.confidence < threshold



def _default_summary(items: list[NodeResult]) -> str:
    joined = "; ".join(f"{item.from_node}: {item.summary}" for item in items)
    return f"Synthesized child results: {joined}"



def _default_escalation_summary(items: list[NodeResult]) -> str:
    joined = "; ".join(f"{item.from_node}: {item.escalation_reason or item.summary}" for item in items)
    return f"Escalation required after child review: {joined}"



def _merge_constraints(*constraint_sets: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in constraint_sets:
        if value:
            merged.update(value)
    return merged



def _child_task_id(parent_task_id: str, node_name: str) -> str:
    normalized = node_name.lower().replace(" ", "_")
    return f"{parent_task_id}.{normalized}"
