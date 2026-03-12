from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4


TaskRole = str
TaskStatus = Literal["completed", "partial", "blocked", "needs_escalation", "failed"]
RoutingMode = str
Direction = Literal["downward", "upward"]


@dataclass(slots=True)
class TaskDesc:
    conversation_id: str
    task_id: str
    parent_task_id: Optional[str]
    from_node: str
    to_node: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.payload = _coerce_task_payload(self.payload)

    @classmethod
    def new(
        cls,
        *,
        from_node: str,
        to_node: str,
        conversation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TaskDesc":
        return cls(
            conversation_id=conversation_id or new_conversation_id(),
            task_id=task_id or new_task_id(),
            parent_task_id=parent_task_id,
            from_node=from_node,
            to_node=to_node,
            payload=payload or {},
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskDesc":
        if "payload" in value:
            return cls(**value)
        return cls.new(
            conversation_id=value.get("conversation_id"),
            task_id=value.get("task_id"),
            parent_task_id=value.get("parent_task_id"),
            from_node=value["from_node"],
            to_node=value["to_node"],
            payload=_coerce_task_payload(value),
            metadata=value.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def original_user_request(self) -> str:
        return self.request

    @property
    def request(self) -> str:
        return str(self.payload.get("request") or "")

    @property
    def original_request_summary(self) -> str:
        return str(self.context.get("summary") or self.request)

    @property
    def caller_understanding(self) -> str:
        return str(self.context.get("understanding") or "")

    @property
    def scoped_task(self) -> str:
        return self.task

    @property
    def task(self) -> str:
        return str(self.payload.get("task") or "")

    @property
    def context(self) -> dict[str, Any]:
        value = self.payload.get("context")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def confidence(self) -> float:
        return _clamp_confidence(self.payload.get("confidence", 1.0))

    @property
    def expected_output(self) -> Optional[dict[str, Any]]:
        value = self.context.get("expected_output")
        return value if isinstance(value, dict) else None

    @property
    def constraints(self) -> dict[str, Any]:
        value = self.context.get("constraints")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def dependencies(self) -> list[str]:
        value = self.context.get("dependencies")
        return list(value) if isinstance(value, list) else []

    @property
    def priority(self) -> Literal["high", "medium", "low"]:
        value = self.context.get("priority", "medium")
        return value if value in {"high", "medium", "low"} else "medium"

    @property
    def handoff_level(self) -> str:
        return str(self.context.get("handoff_kind") or self.metadata.get("handoff_kind") or "delegation")

    @property
    def escalation_if_below(self) -> float:
        return _clamp_confidence(self.context.get("escalate_if_below", self.metadata.get("escalate_if_below", 0.6)))


@dataclass(slots=True)
class TaskResult:
    conversation_id: str
    task_id: str
    parent_task_id: Optional[str]
    from_node: str
    to_node: str
    role: TaskRole
    status: TaskStatus
    summary: str
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    escalation_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)
        if self.confidence < 0.6 and self.status not in {"needs_escalation", "failed"}:
            self.status = "needs_escalation"
            if self.escalation_reason is None:
                self.escalation_reason = "confidence below escalation threshold"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskResult":
        return cls(**value)

    @classmethod
    def complete(
        cls,
        *,
        brief: TaskDesc,
        role: TaskRole,
        summary: str,
        from_node: Optional[str] = None,
        output: Optional[dict[str, Any]] = None,
        confidence: float = 1.0,
        assumptions: Optional[list[str]] = None,
        blockers: Optional[list[str]] = None,
        child_summaries: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TaskResult":
        result_metadata = dict(metadata or {})
        if assumptions:
            result_metadata["assumptions"] = list(assumptions)
        if blockers:
            result_metadata["blockers"] = list(blockers)
        if child_summaries:
            result_metadata["child_summaries"] = list(child_summaries)
        return cls(
            conversation_id=brief.conversation_id,
            task_id=brief.task_id,
            parent_task_id=brief.parent_task_id,
            from_node=from_node or brief.to_node,
            to_node=brief.from_node,
            role=role,
            status="completed",
            summary=summary,
            output=output or {},
            confidence=confidence,
            metadata=result_metadata,
        )

    @classmethod
    def escalate(
        cls,
        *,
        brief: TaskDesc,
        role: TaskRole,
        summary: str,
        from_node: Optional[str] = None,
        output: Optional[dict[str, Any]] = None,
        confidence: float = 0.0,
        blockers: Optional[list[str]] = None,
        escalation_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "TaskResult":
        result_metadata = dict(metadata or {})
        if blockers:
            result_metadata["blockers"] = list(blockers)
        return cls(
            conversation_id=brief.conversation_id,
            task_id=brief.task_id,
            parent_task_id=brief.parent_task_id,
            from_node=from_node or brief.to_node,
            to_node=brief.from_node,
            role=role,
            status="needs_escalation",
            summary=summary,
            output=output or {},
            confidence=confidence,
            escalation_reason=escalation_reason or "confidence below escalation threshold",
            metadata=result_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def assumptions(self) -> list[str]:
        value = self.metadata.get("assumptions")
        return list(value) if isinstance(value, list) else []

    @property
    def blockers(self) -> list[str]:
        value = self.metadata.get("blockers")
        return list(value) if isinstance(value, list) else []

    @property
    def child_summaries(self) -> list[dict[str, Any]]:
        value = self.metadata.get("child_summaries")
        return list(value) if isinstance(value, list) else []


@dataclass(slots=True)
class RoutingDecision:
    conversation_id: str
    task_id: str
    mode: RoutingMode
    departments: list[str]
    rationale: str
    confidence: float
    clarification_question: Optional[str] = None

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HandoffRecord:
    conversation_id: str
    task_id: str
    parent_task_id: Optional[str]
    from_node: str
    to_node: str
    direction: Direction
    summary: str
    confidence: float
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DepartmentSpec:
    name: str
    description: str
    manager: Any
    specialists: list[Any] = field(default_factory=list)

    @classmethod
    def create(cls, *, name: str, description: str, manager: Any) -> "DepartmentSpec":
        return cls(name=name, description=description, manager=manager)

    def add_specialist(self, specialist: Any) -> "DepartmentSpec":
        self.specialists.append(specialist)
        return self

    def add_specialists(self, *specialists: Any) -> "DepartmentSpec":
        self.specialists.extend(specialists)
        return self


def new_conversation_id() -> str:
    return f"conv_{uuid4().hex}"


def new_task_id() -> str:
    return f"task_{uuid4().hex}"


def child_task_id(parent_task_id: str, node_name: str) -> str:
    return f"{parent_task_id}.{node_name.lower().replace(' ', '_')}"


def clamp_confidence(value: float) -> float:
    return _clamp_confidence(value)


def _coerce_task_payload(value: dict[str, Any]) -> dict[str, Any]:
    payload_value = value.get("payload")
    if isinstance(payload_value, dict):
        payload = dict(payload_value)
    else:
        payload = {}

    if "request" not in payload:
        request = value.get("original_user_request", value.get("request"))
        if request:
            payload["request"] = request
    if "task" not in payload:
        task = value.get("scoped_task", value.get("task"))
        if task:
            payload["task"] = task
    if "context" not in payload:
        context = _coerce_task_context(value)
        if context:
            payload["context"] = context
    if "confidence" not in payload and value.get("confidence") is not None:
        payload["confidence"] = _clamp_confidence(value.get("confidence", 1.0))
    return payload


def _coerce_task_context(value: dict[str, Any]) -> dict[str, Any]:
    context_value = value.get("context")
    if isinstance(context_value, dict):
        return dict(context_value)

    context: dict[str, Any] = {}
    summary = value.get("original_request_summary", value.get("request_summary"))
    if summary:
        context["summary"] = summary
    understanding = value.get("caller_understanding", value.get("current_understanding"))
    if understanding:
        context["understanding"] = understanding
    expected_output = value.get("expected_output")
    if isinstance(expected_output, dict):
        context["expected_output"] = expected_output
    constraints = value.get("constraints")
    if isinstance(constraints, dict):
        context["constraints"] = dict(constraints)
    dependencies = value.get("dependencies")
    if isinstance(dependencies, list):
        context["dependencies"] = list(dependencies)
    priority = value.get("priority")
    if priority in {"high", "medium", "low"}:
        context["priority"] = priority
    handoff_kind = value.get("handoff_level", value.get("handoff_kind"))
    if handoff_kind:
        context["handoff_kind"] = handoff_kind
    escalate_if_below = value.get("escalation_if_below", value.get("escalate_if_below"))
    if escalate_if_below is not None:
        context["escalate_if_below"] = _clamp_confidence(escalate_if_below)
    return context


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
