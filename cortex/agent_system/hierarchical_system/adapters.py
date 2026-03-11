from __future__ import annotations

import json
from dataclasses import dataclass
from inspect import iscoroutine
from typing import Any

from cortex.agent import Agent
from cortex.message import DeveloperMessage, UserMessage
from cortex.workflow import WorkflowAgent

from .models import DelegationBrief, NodeResult


@dataclass(slots=True)
class AgentNodeAdapter:
    name: str
    role: str
    agent: Agent
    confidence_threshold: float = 0.6

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        messages = [
            DeveloperMessage(content=_agent_protocol_prompt(self.role, self.confidence_threshold)),
            UserMessage(content=json.dumps(brief.to_dict(), ensure_ascii=False, indent=2)),
        ]
        value = await self.agent.async_ask(messages, usage=getattr(context, "usage", None))
        return normalize_node_result(value, brief=brief, role=self.role, fallback_name=self.name)


@dataclass(slots=True)
class WorkflowNodeAdapter:
    name: str
    role: str
    workflow: WorkflowAgent

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        run = await self.workflow.async_run(user_input=brief.to_dict(), context=context)
        value = run.final_output
        result = normalize_node_result(value, brief=brief, role=self.role, fallback_name=self.name)
        result.metadata.setdefault("workflow_status", run.status)
        result.metadata.setdefault("workflow_error", run.error)
        result.metadata.setdefault("workflow_trace_count", len(run.traces))
        return result


async def maybe_await(value: Any) -> Any:
    if iscoroutine(value):
        return await value
    return value


def normalize_node_result(value: Any, *, brief: DelegationBrief, role: str, fallback_name: str) -> NodeResult:
    if isinstance(value, NodeResult):
        return value
    if isinstance(value, dict):
        return _result_from_mapping(value, brief=brief, role=role, fallback_name=fallback_name)
    if value is None:
        return NodeResult.complete(
            brief=brief,
            role=role,  # type: ignore[arg-type]
            from_node=fallback_name,
            summary="Completed without a structured payload.",
            output={},
            confidence=0.8,
        )
    return NodeResult.complete(
        brief=brief,
        role=role,  # type: ignore[arg-type]
        from_node=fallback_name,
        summary=str(value),
        output={"response": value},
        confidence=0.8,
    )


def _result_from_mapping(value: dict[str, Any], *, brief: DelegationBrief, role: str, fallback_name: str) -> NodeResult:
    if _looks_like_node_result(value):
        return NodeResult.from_dict(value)

    summary = str(value.get("summary") or value.get("response") or value.get("message") or "Completed")
    confidence = float(value.get("confidence", 0.8))
    status = value.get("status", "completed")
    output = value.get("output")
    if not isinstance(output, dict):
        output = {"value": output if output is not None else value}

    return NodeResult(
        conversation_id=value.get("conversation_id", brief.conversation_id),
        task_id=value.get("task_id", brief.task_id),
        parent_task_id=value.get("parent_task_id", brief.parent_task_id),
        from_node=value.get("from_node", fallback_name),
        to_node=value.get("to_node", brief.from_node),
        role=value.get("role", role),
        status=status,
        summary=summary,
        output=output,
        confidence=confidence,
        assumptions=list(value.get("assumptions", [])),
        blockers=list(value.get("blockers", [])),
        child_summaries=list(value.get("child_summaries", [])),
        escalation_reason=value.get("escalation_reason"),
        metadata=dict(value.get("metadata", {})),
    )


def _looks_like_node_result(value: dict[str, Any]) -> bool:
    required = {"summary", "status", "confidence", "role"}
    return required.issubset(value)


def _agent_protocol_prompt(role: str, threshold: float) -> str:
    return (
        f"You are operating as a {role} node in a hierarchical agent system. "
        "You will receive a JSON delegation brief. Follow these rules strictly: "
        "enrich only when delegating downward, synthesize when reporting upward, never pass raw child outputs upward, "
        f"and if your confidence is below {threshold:.1f}, return a structured escalation instead of guessing. "
        "Return JSON matching this shape: "
        '{"role": "gateway|manager|worker", "status": "completed|partial|blocked|needs_escalation|failed", '
        '"summary": "...", "output": {}, "confidence": 0.0, "assumptions": [], "blockers": [], '
        '"child_summaries": [], "escalation_reason": null, "metadata": {}}.'
    )
