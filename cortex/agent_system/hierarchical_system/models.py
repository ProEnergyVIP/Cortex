from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional
from uuid import uuid4


Priority = Literal["high", "medium", "low"]
HandoffLevel = str
NodeRole = str
NodeStatus = Literal["completed", "partial", "blocked", "needs_escalation", "failed"]
RoutingMode = str
Direction = Literal["downward", "upward"]


@dataclass(slots=True)
class DelegationBrief:
    conversation_id: str
    task_id: str
    parent_task_id: Optional[str]
    from_node: str
    to_node: str
    handoff_level: HandoffLevel
    original_user_request: str
    original_request_summary: str
    caller_understanding: str
    scoped_task: str
    expected_output: Optional[dict[str, Any]] = None
    constraints: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    priority: Priority = "medium"
    confidence: float = 1.0
    escalation_if_below: float = 0.6
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)
        self.escalation_if_below = _clamp_confidence(self.escalation_if_below)

    @classmethod
    def new(
        cls,
        *,
        from_node: str,
        to_node: str,
        handoff_level: HandoffLevel,
        original_user_request: str,
        original_request_summary: str,
        caller_understanding: str,
        scoped_task: str,
        conversation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        parent_task_id: Optional[str] = None,
        expected_output: Optional[dict[str, Any]] = None,
        constraints: Optional[dict[str, Any]] = None,
        dependencies: Optional[list[str]] = None,
        priority: Priority = "medium",
        confidence: float = 1.0,
        escalation_if_below: float = 0.6,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "DelegationBrief":
        return cls(
            conversation_id=conversation_id or new_conversation_id(),
            task_id=task_id or new_task_id(),
            parent_task_id=parent_task_id,
            from_node=from_node,
            to_node=to_node,
            handoff_level=handoff_level,
            original_user_request=original_user_request,
            original_request_summary=original_request_summary,
            caller_understanding=caller_understanding,
            scoped_task=scoped_task,
            expected_output=expected_output,
            constraints=constraints or {},
            dependencies=dependencies or [],
            priority=priority,
            confidence=confidence,
            escalation_if_below=escalation_if_below,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DelegationBrief":
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NodeResult:
    conversation_id: str
    task_id: str
    parent_task_id: Optional[str]
    from_node: str
    to_node: str
    role: NodeRole
    status: NodeStatus
    summary: str
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    assumptions: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    child_summaries: list[dict[str, Any]] = field(default_factory=list)
    escalation_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp_confidence(self.confidence)
        if self.confidence < 0.6 and self.status not in {"needs_escalation", "failed"}:
            self.status = "needs_escalation"
            if self.escalation_reason is None:
                self.escalation_reason = "confidence below escalation threshold"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NodeResult":
        return cls(**value)

    @classmethod
    def complete(
        cls,
        *,
        brief: DelegationBrief,
        role: NodeRole,
        summary: str,
        from_node: Optional[str] = None,
        output: Optional[dict[str, Any]] = None,
        confidence: float = 1.0,
        assumptions: Optional[list[str]] = None,
        blockers: Optional[list[str]] = None,
        child_summaries: Optional[list[dict[str, Any]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "NodeResult":
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
            assumptions=assumptions or [],
            blockers=blockers or [],
            child_summaries=child_summaries or [],
            metadata=metadata or {},
        )

    @classmethod
    def escalate(
        cls,
        *,
        brief: DelegationBrief,
        role: NodeRole,
        summary: str,
        from_node: Optional[str] = None,
        output: Optional[dict[str, Any]] = None,
        confidence: float = 0.0,
        blockers: Optional[list[str]] = None,
        escalation_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "NodeResult":
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
            blockers=blockers or [],
            escalation_reason=escalation_reason or "confidence below escalation threshold",
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
