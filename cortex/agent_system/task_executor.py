from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .task_models import TaskDesc, TaskResult


@runtime_checkable
class TaskExecutor(Protocol):
    name: str
    role: str

    async def run_brief(self, brief: TaskDesc, *, context: Any) -> TaskResult:
        ...


TaskExecutorFactory = Callable[[Any], Awaitable[TaskExecutor] | TaskExecutor]
ResultNormalizer = Callable[[Any, TaskDesc, str], TaskResult]


@dataclass(slots=True)
class BuiltTaskExecutor:
    name: str
    role: str
    runtime: TaskExecutor

    async def run_brief(self, brief: TaskDesc, *, context: Any) -> TaskResult:
        return await self.runtime.run_brief(brief, context=context)
