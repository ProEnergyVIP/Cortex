from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from cortex.agent import Agent, Tool
from cortex.message import Message

from ..core.system import AgentSystem
from ..task_composition import create_task_desc, create_task_tool, execute_task_executor, resolve_task_executor
from ..task_executor import BuiltTaskExecutor
from ..task_models import TaskDesc, TaskResult
from .builders import TaskCoordinatorBuilder, TaskWorkerBuilder
from .models import TaskCoordinatorRuntime, TaskCoordinatorSpec, TaskWorkerRuntime, TaskWorkerSpec


class TaskCoordinatorSystem(AgentSystem):
    def __init__(
        self,
        coordinator: TaskCoordinatorSpec | TaskCoordinatorBuilder | TaskCoordinatorRuntime,
        workers: Optional[Iterable[TaskWorkerSpec | TaskWorkerBuilder | TaskWorkerRuntime]] = None,
        context: Optional[Any] = None,
        default_expected_output: Optional[dict[str, Any]] = None,
    ):
        super().__init__(context)
        self._coordinator = _normalize_coordinator_spec(coordinator)
        self._workers = [_normalize_worker_spec(worker) for worker in (workers or [])]
        self._default_expected_output = default_expected_output or {"type": "final_answer"}

    async def get_agent(self) -> Agent:
        raise NotImplementedError(
            "TaskCoordinatorSystem is built on task executors and may use Agent or WorkflowAgent runtimes. "
            "Use async_ask(), async_run_task(), or build_coordinator_executor() instead of get_agent()."
        )

    async def build_coordinator_executor(self) -> BuiltTaskExecutor:
        worker_tools = self._build_worker_tools()
        return await resolve_task_executor(
            self._coordinator.executor,
            context=self._context,
            role=self._coordinator.role,
            name=self._coordinator.name,
            installed_tools=worker_tools,
        )

    async def async_run_task(self, desc: TaskDesc) -> TaskResult:
        worker_tools = self._build_worker_tools()
        return await execute_task_executor(
            self._coordinator.executor,
            desc=desc,
            context=self._context,
            role=self._coordinator.role,
            name=self._coordinator.name,
            installed_tools=worker_tools,
        )

    async def async_ask(
        self,
        messages: str | Message | list[Message],
        *,
        request_summary: Optional[str] = None,
        current_understanding: Optional[str] = None,
        assigned_task: Optional[str] = None,
        expected_output: Optional[dict[str, Any]] = None,
        constraints: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        user_input = _coerce_user_input(messages)
        desc = create_task_desc(
            from_executor="user",
            to_executor=self._coordinator.name,
            handoff_kind="user_to_coordinator",
            original_request=user_input,
            request_summary=request_summary or _default_request_summary(user_input),
            current_understanding=current_understanding or "Interpret the request, decide whether to answer directly or delegate to workers, and produce the best final response.",
            assigned_task=assigned_task or "Handle the user's request by delegating clearly structured sub-tasks when helpful, then synthesize the best final response.",
            expected_output=expected_output or self._default_expected_output,
            constraints=constraints,
            metadata=metadata,
        )
        result = await self.async_run_task(desc)
        return _stringify_response_payload(
            result.output.get("final_answer"),
            fallback=result.output.get("clarification_question"),
            default=result.summary,
        )

    def _build_worker_tools(self) -> list[Tool]:
        return [
            create_task_tool(
                worker.executor,
                name=worker.name,
                role=worker.role,
                tool_name=worker.tool_name,
                description=worker.description,
            )
            for worker in self._workers
        ]


def _normalize_coordinator_spec(
    coordinator: TaskCoordinatorSpec | TaskCoordinatorBuilder | TaskCoordinatorRuntime,
) -> TaskCoordinatorSpec:
    if isinstance(coordinator, TaskCoordinatorSpec):
        return coordinator
    if isinstance(coordinator, TaskCoordinatorBuilder):
        return coordinator.as_spec()
    return TaskCoordinatorSpec(
        name=getattr(coordinator, "name", None) or "Coordinator",
        role=getattr(coordinator, "role", None) or "coordinator",
        executor=coordinator,
        description=getattr(coordinator, "description", None),
        metadata=_extract_metadata(coordinator),
    )


def _normalize_worker_spec(worker: TaskWorkerSpec | TaskWorkerBuilder | TaskWorkerRuntime) -> TaskWorkerSpec:
    if isinstance(worker, TaskWorkerSpec):
        return worker
    if isinstance(worker, TaskWorkerBuilder):
        return worker.as_spec()
    return TaskWorkerSpec(
        name=getattr(worker, "name", None) or "Worker",
        role=getattr(worker, "role", None) or "worker",
        executor=worker,
        description=getattr(worker, "description", None),
        metadata=_extract_metadata(worker),
    )


def _extract_metadata(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _coerce_user_input(messages: str | Message | list[Message]) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Message):
        return str(messages.content)
    if isinstance(messages, list):
        parts = [str(message.content) for message in messages if getattr(message, "content", None)]
        return "\n".join(parts)
    raise TypeError(f"Unsupported message input: {type(messages)!r}")


def _default_request_summary(user_input: str) -> str:
    collapsed = " ".join(user_input.split())
    if len(collapsed) <= 160:
        return collapsed
    return collapsed[:157].rstrip() + "..."


def _stringify_response_payload(value: Any, *, fallback: Any, default: str) -> str:
    chosen = value if value is not None else fallback
    if chosen is None:
        return default
    if isinstance(chosen, str):
        return chosen
    return json.dumps(chosen, ensure_ascii=False)
