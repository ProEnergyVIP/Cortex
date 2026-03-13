from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .engine import EngineRun, EngineState, EngineTrace


@dataclass
class NodeTrace(EngineTrace):
    """Execution trace entry for a single workflow node attempt sequence."""


@dataclass
class WorkflowState(EngineState):
    """Mutable state object shared across workflow nodes."""


@dataclass
class WorkflowRun(EngineRun):
    """Full execution record for a workflow invocation."""

    @property
    def workflow_name(self) -> str:
        return self.engine_name

    @workflow_name.setter
    def workflow_name(self, value: str) -> None:
        self.engine_name = value

    def last_trace(self) -> Optional[NodeTrace]:
        if not self.traces:
            return None
        return self.traces[-1]

    def failed_trace(self) -> Optional[NodeTrace]:
        for trace in reversed(self.traces):
            if trace.status == "failed":
                return trace
        return None

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        payload["workflow_name"] = payload.pop("engine_name")
        return payload
