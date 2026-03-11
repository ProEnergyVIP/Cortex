from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cortex.agent import Agent
from cortex.workflow import WorkflowAgent

from ..task_executor import BuiltTaskExecutor
from ..task_executor_builders import TaskExecutorBuilderBase

TaskCoordinatorRuntime = TaskExecutorBuilderBase | BuiltTaskExecutor | Agent | WorkflowAgent
TaskWorkerRuntime = TaskExecutorBuilderBase | BuiltTaskExecutor | Agent | WorkflowAgent


@dataclass(slots=True)
class TaskCoordinatorSpec:
    name: str
    executor: TaskCoordinatorRuntime
    role: str = "coordinator"
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskWorkerSpec:
    name: str
    executor: TaskWorkerRuntime
    role: str = "worker"
    description: str | None = None
    tool_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
