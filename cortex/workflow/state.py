from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .engine import EngineRun, EngineState, EngineTrace


@dataclass
class StepTrace(EngineTrace):
    """Execution trace entry for a single workflow step attempt sequence."""

    @property
    def step_name(self) -> str:
        return self.node_name

    @step_name.setter
    def step_name(self, value: str) -> None:
        self.node_name = value

    @property
    def next_step(self) -> Optional[str]:
        return self.next_node

    @next_step.setter
    def next_step(self, value: Optional[str]) -> None:
        self.next_node = value

    @property
    def fallback_step(self) -> Optional[str]:
        return self.fallback_node

    @fallback_step.setter
    def fallback_step(self, value: Optional[str]) -> None:
        self.fallback_node = value

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["step_name"] = payload.pop("node_name")
        payload["next_step"] = payload.pop("next_node")
        payload["fallback_step"] = payload.pop("fallback_node")
        return payload


@dataclass
class WorkflowState(EngineState):
    """Mutable state object shared across workflow steps."""

    @property
    def current_step(self) -> Optional[str]:
        return self.current_node

    @current_step.setter
    def current_step(self, value: Optional[str]) -> None:
        self.current_node = value

    @property
    def completed_steps(self) -> list[str]:
        return self.completed_nodes

    @completed_steps.setter
    def completed_steps(self, value: list[str]) -> None:
        self.completed_nodes = value

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["current_step"] = payload.pop("current_node")
        payload["completed_steps"] = payload.pop("completed_nodes")
        return payload


@dataclass
class WorkflowRun(EngineRun):
    """Full execution record for a workflow invocation."""

    @property
    def workflow_name(self) -> str:
        return self.engine_name

    @workflow_name.setter
    def workflow_name(self, value: str) -> None:
        self.engine_name = value

    def last_trace(self) -> Optional[StepTrace]:
        if not self.traces:
            return None
        return self.traces[-1]

    def failed_trace(self) -> Optional[StepTrace]:
        for trace in reversed(self.traces):
            if trace.status == "failed":
                return trace
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["workflow_name"] = payload.pop("engine_name")
        return payload
