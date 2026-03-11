from __future__ import annotations

from typing import Any, Iterable, Optional

from .task_types import RoutingDecision, TaskBrief, TaskResult, child_task_id


def build_child_task_brief_for_role(
    *,
    parent_brief: TaskBrief,
    child_name: str,
    handoff_kind: str,
    scoped_task: str,
    caller_understanding: str,
    expected_output: Optional[dict[str, Any]] = None,
    constraints: Optional[dict[str, Any]] = None,
    dependencies: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskBrief:
    merged_constraints = _merge_constraints(parent_brief.constraints, constraints)
    return TaskBrief.new(
        conversation_id=parent_brief.conversation_id,
        task_id=child_task_id(parent_brief.task_id, child_name),
        parent_task_id=parent_brief.task_id,
        from_node=parent_brief.to_node,
        to_node=child_name,
        handoff_level=handoff_kind,
        original_user_request=parent_brief.original_user_request,
        original_request_summary=parent_brief.original_request_summary,
        caller_understanding=caller_understanding,
        scoped_task=scoped_task,
        expected_output=expected_output,
        constraints=merged_constraints,
        dependencies=list(parent_brief.dependencies) + list(dependencies or []),
        priority=parent_brief.priority,
        confidence=parent_brief.confidence,
        escalation_if_below=parent_brief.escalation_if_below,
        metadata={**parent_brief.metadata, **(metadata or {})},
    )


def synthesize_task_results(
    *,
    brief: TaskBrief,
    role: str,
    from_node: Optional[str],
    child_results: Iterable[TaskResult],
    summary: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> TaskResult:
    items = list(child_results)
    if not items:
        return TaskResult.escalate(
            brief=brief,
            role=role,
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
        return TaskResult.escalate(
            brief=brief,
            role=role,
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
    return TaskResult.complete(
        brief=brief,
        role=role,
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
    brief: TaskBrief,
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


def should_escalate(result: TaskResult, *, threshold: float = 0.6) -> bool:
    return result.status in {"needs_escalation", "failed"} or result.confidence < threshold


def _default_summary(items: list[TaskResult]) -> str:
    joined = "; ".join(f"{item.from_node}: {item.summary}" for item in items)
    return f"Synthesized child results: {joined}"


def _default_escalation_summary(items: list[TaskResult]) -> str:
    joined = "; ".join(f"{item.from_node}: {item.escalation_reason or item.summary}" for item in items)
    return f"Escalation required after child review: {joined}"


def _merge_constraints(*constraint_sets: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in constraint_sets:
        if value:
            merged.update(value)
    return merged
