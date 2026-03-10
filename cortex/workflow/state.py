from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


def _serialize_value(value: Any) -> Any:
    """Recursively normalize workflow objects into dictionary-friendly values."""
    if isinstance(value, WorkflowRun):
        return value.to_dict()
    if isinstance(value, WorkflowState):
        return value.to_dict()
    if isinstance(value, StepTrace):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass
class StepTrace:
    """Execution trace entry for a single workflow step attempt sequence."""

    step_name: str
    status: str = "pending"
    attempt_count: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    next_step: Optional[str] = None
    fallback_step: Optional[str] = None
    state_before: Optional[dict[str, Any]] = None
    state_after: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace into plain Python data."""
        return {
            "step_name": self.step_name,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "duration_ms": self.duration_ms,
            "next_step": self.next_step,
            "fallback_step": self.fallback_step,
            "state_before": _serialize_value(self.state_before),
            "state_after": _serialize_value(self.state_after),
            "metadata": _serialize_value(self.metadata),
            "output": _serialize_value(self.output),
            "error": self.error,
        }


@dataclass
class WorkflowState:
    """Mutable state object shared across workflow steps."""

    input: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    last_output: Any = None
    final_output: Any = None
    current_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a state value with an optional default."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Return whether a state key exists."""
        return key in self.data

    def require(self, key: str) -> Any:
        """Read a state value or raise if it is missing."""
        if key not in self.data:
            raise KeyError(f"Workflow state is missing required key: {key}")
        return self.data[key]

    def set(self, key: str, value: Any) -> Any:
        """Set a state value and return it."""
        self.data[key] = value
        return value

    def update(self, values: Optional[dict[str, Any]]) -> None:
        """Merge a mapping of updates into the workflow state."""
        if values:
            self.data.update(values)

    def set_output(self, value: Any) -> Any:
        """Record the most recent step output."""
        self.last_output = value
        return value

    def set_final_output(self, value: Any) -> Any:
        """Record the final workflow output and mirror it into last output."""
        self.final_output = value
        self.last_output = value
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow state into plain Python data."""
        return {
            "input": _serialize_value(self.input),
            "data": _serialize_value(dict(self.data)),
            "last_output": _serialize_value(self.last_output),
            "final_output": _serialize_value(self.final_output),
            "current_step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "metadata": _serialize_value(dict(self.metadata)),
        }


@dataclass
class WorkflowRun:
    """Full execution record for a workflow invocation."""

    workflow_name: str
    traces: list[StepTrace] = field(default_factory=list)
    state: Optional[WorkflowState] = None
    status: str = "pending"
    final_output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def add_trace(self, trace: StepTrace) -> None:
        """Append a step trace to this run."""
        self.traces.append(trace)

    @property
    def duration_ms(self) -> Optional[float]:
        """Return the total run duration in milliseconds, if complete."""
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000

    def last_trace(self) -> Optional[StepTrace]:
        """Return the most recent trace entry, if any."""
        if not self.traces:
            return None
        return self.traces[-1]

    def failed_trace(self) -> Optional[StepTrace]:
        """Return the most recent failed trace entry, if any."""
        for trace in reversed(self.traces):
            if trace.status == "failed":
                return trace
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow run into plain Python data."""
        return {
            "workflow_name": self.workflow_name,
            "traces": [trace.to_dict() for trace in self.traces],
            "state": self.state.to_dict() if self.state is not None else None,
            "status": self.status,
            "final_output": _serialize_value(self.final_output),
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "duration_ms": self.duration_ms,
        }
