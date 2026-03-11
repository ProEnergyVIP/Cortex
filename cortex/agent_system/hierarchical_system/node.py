from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .models import DelegationBrief, NodeResult


@runtime_checkable
class ExecutionNode(Protocol):
    name: str
    role: str

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        ...


NodeFactory = Callable[[Any], Awaitable[ExecutionNode] | ExecutionNode]
ResultNormalizer = Callable[[Any, DelegationBrief, str], NodeResult]


@dataclass(slots=True)
class BuiltNode:
    name: str
    role: str
    runtime: ExecutionNode

    async def run_brief(self, brief: DelegationBrief, *, context: Any) -> NodeResult:
        return await self.runtime.run_brief(brief, context=context)
