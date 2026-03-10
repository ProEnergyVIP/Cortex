from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class StepTrace:
    step_name: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    next_step: Optional[str] = None
    output: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowState:
    input: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    last_output: Any = None
    final_output: Any = None
    current_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"Workflow state is missing required key: {key}")
        return self.data[key]

    def set(self, key: str, value: Any) -> Any:
        self.data[key] = value
        return value

    def update(self, values: Optional[dict[str, Any]]) -> None:
        if values:
            self.data.update(values)

    def set_output(self, value: Any) -> Any:
        self.last_output = value
        return value

    def set_final_output(self, value: Any) -> Any:
        self.final_output = value
        self.last_output = value
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "data": dict(self.data),
            "last_output": self.last_output,
            "final_output": self.final_output,
            "current_step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "metadata": dict(self.metadata),
        }


@dataclass
class WorkflowRun:
    workflow_name: str
    traces: list[StepTrace] = field(default_factory=list)
    state: Optional[WorkflowState] = None
    status: str = "pending"
    final_output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def add_trace(self, trace: StepTrace) -> None:
        self.traces.append(trace)
