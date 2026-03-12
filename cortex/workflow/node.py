from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

FailureStrategy = Literal["raise", "fallback"]


@dataclass(frozen=True)
class NodePolicy:
    """Runtime policy controlling retries, fallback behavior, and timeouts."""

    max_retries: int = 0
    failure_strategy: FailureStrategy = "raise"
    fallback_node: Optional[str] = None
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError("NodePolicy.max_retries must be >= 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("NodePolicy.timeout_seconds must be > 0 when provided")
        if self.failure_strategy == "fallback" and not self.fallback_node:
            raise ValueError("NodePolicy with failure_strategy='fallback' requires fallback_node")
        if self.failure_strategy != "fallback" and self.fallback_node is not None:
            raise ValueError("fallback_node can only be set when failure_strategy='fallback'")


class EngineNode(Protocol):
    name: str
    next_node: Optional[str]
    policy: NodePolicy

    async def run(self, state, *, context: Any = None, workflow: Any = None):
        ...

    def declared_next_nodes(self) -> set[str]:
        ...

    def is_terminal(self) -> bool:
        ...
