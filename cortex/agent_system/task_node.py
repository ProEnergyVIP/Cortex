from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .task_types import TaskBrief, TaskResult


@runtime_checkable
class TaskRunner(Protocol):
    name: str
    role: str

    async def run_brief(self, brief: TaskBrief, *, context: Any) -> TaskResult:
        ...


TaskRunnerFactory = Callable[[Any], Awaitable[TaskRunner] | TaskRunner]
ResultNormalizer = Callable[[Any, TaskBrief, str], TaskResult]


@dataclass(slots=True)
class BuiltTaskRunner:
    name: str
    role: str
    runtime: TaskRunner

    async def run_brief(self, brief: TaskBrief, *, context: Any) -> TaskResult:
        return await self.runtime.run_brief(brief, context=context)
