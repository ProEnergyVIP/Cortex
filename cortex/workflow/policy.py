from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .node import FailureStrategy, NodePolicy

__all__ = ["FailureStrategy", "StepPolicy"]


@dataclass(frozen=True)
class StepPolicy(NodePolicy):
    """Runtime policy controlling retries, fallback behavior, and timeouts."""

    fallback_step: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.fallback_node is not None and self.fallback_step is not None and self.fallback_node != self.fallback_step:
            raise ValueError("fallback_node and fallback_step must match when both are provided")
        fallback_value = self.fallback_step if self.fallback_step is not None else self.fallback_node
        object.__setattr__(self, "fallback_step", fallback_value)
        object.__setattr__(self, "fallback_node", fallback_value)
