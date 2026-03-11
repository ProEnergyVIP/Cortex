from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..task_node import BuiltTaskRunner as BuiltNode
from ..task_node import TaskRunner as ExecutionNode
from .models import DelegationBrief, NodeResult

NodeFactory = Callable[[Any], Awaitable[ExecutionNode] | ExecutionNode]
ResultNormalizer = Callable[[Any, DelegationBrief, str], NodeResult]

__all__ = [
    "ExecutionNode",
    "NodeFactory",
    "ResultNormalizer",
    "BuiltNode",
]
