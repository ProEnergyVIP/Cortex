from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from asyncio import iscoroutine
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Optional

from cortex.LLMFunc import CheckResult, llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage
from cortex.tool import BaseTool

from .policy import StepPolicy
from .state import WorkflowState
from .types import InputBuilder, PromptBuilder, RouterFunction, StepFunction


async def _resolve_callable(value, *args):
    if callable(value):
        sig = signature(value)
        count = len(sig.parameters)
        if count == 0:
            result = value()
        else:
            result = value(*args[:count])
        if iscoroutine(result):
            return await result
        return result
    return value


class WorkflowStepError(Exception):
    def __init__(self, message: str, *, trace_data: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.trace_data = trace_data or {}


@dataclass
class StepResult:
    updates: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    next_step: Optional[str] = None
    stop: bool = False
    final_output: Any = None
    trace_data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def next(cls, step_name: str, *, output: Any = None, updates: Optional[dict[str, Any]] = None) -> "StepResult":
        return cls(
            updates=updates or {},
            output=output,
            next_step=step_name,
        )

    @classmethod
    def finish(cls, output: Any = None, *, updates: Optional[dict[str, Any]] = None, final_output: Any = None) -> "StepResult":
        resolved_final_output = output if final_output is None else final_output
        return cls(
            updates=updates or {},
            output=output,
            stop=True,
            final_output=resolved_final_output,
        )

    @classmethod
    def update_state(
        cls,
        updates: dict[str, Any],
        *,
        output: Any = None,
        next_step: Optional[str] = None,
    ) -> "StepResult":
        return cls(
            updates=updates,
            output=output,
            next_step=next_step,
        )

    def apply(self, state: WorkflowState) -> None:
        state.update(self.updates)
        if self.output is not None:
            state.set_output(self.output)
        if self.final_output is not None:
            state.set_final_output(self.final_output)


class Step(ABC):
    def __init__(self, name: str, next_step: Optional[str] = None, policy: Optional[StepPolicy] = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Step name must be a non-empty string")
        self.name = name
        self.next_step = next_step
        self.policy = policy or StepPolicy()

    def declared_next_steps(self) -> set[str]:
        if self.next_step is None:
            return set()
        return {self.next_step}

    def is_terminal(self) -> bool:
        return False

    @abstractmethod
    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        raise NotImplementedError


class FunctionStep(Step):
    def __init__(
        self,
        name: str,
        func: StepFunction,
        next_step: Optional[str] = None,
        output_key: Optional[str] = None,
        policy: Optional[StepPolicy] = None,
    ):
        super().__init__(name=name, next_step=next_step, policy=policy)
        self.func = func
        self.output_key = output_key

    @classmethod
    def final(
        cls,
        name: str,
        func: StepFunction,
        output_key: Optional[str] = None,
        policy: Optional[StepPolicy] = None,
    ) -> "FunctionStep":
        return cls(name=name, func=func, next_step=None, output_key=output_key, policy=policy)

    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        result = self.func(state, context, workflow)
        if iscoroutine(result):
            result = await result

        if isinstance(result, StepResult):
            if result.next_step is None:
                result.next_step = self.next_step
            return result

        updates = {}
        if self.output_key is not None:
            updates[self.output_key] = result

        return StepResult(updates=updates, output=result, next_step=self.next_step)


class RouterStep(FunctionStep):
    def __init__(
        self,
        name: str,
        func: RouterFunction,
        next_step: Optional[str] = None,
        output_key: Optional[str] = None,
        possible_next_steps: Optional[list[str]] = None,
        policy: Optional[StepPolicy] = None,
    ):
        super().__init__(name=name, func=func, next_step=next_step, output_key=output_key, policy=policy)
        self.possible_next_steps = set(possible_next_steps or [])

    def declared_next_steps(self) -> set[str]:
        return super().declared_next_steps().union(self.possible_next_steps)

    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        result = self.func(state, context, workflow)
        if iscoroutine(result):
            result = await result

        if isinstance(result, StepResult):
            return result

        if result is None:
            return StepResult(next_step=self.next_step)

        return StepResult(output=result, next_step=str(result))


class WorkflowStep(Step):
    def __init__(
        self,
        name: str,
        workflow_agent,
        *,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        next_step: Optional[str] = None,
        policy: Optional[StepPolicy] = None,
        is_final: bool = False,
    ):
        super().__init__(name=name, next_step=next_step, policy=policy)
        self.workflow_agent = workflow_agent
        self.input_builder = input_builder
        self.output_key = output_key
        self.is_final = is_final

        if self.is_final and self.next_step is not None:
            raise ValueError("Final WorkflowSteps cannot declare next_step")
        if not hasattr(self.workflow_agent, "async_run"):
            raise ValueError("WorkflowStep requires workflow_agent to define async_run")

    def is_terminal(self) -> bool:
        return self.is_final

    @classmethod
    def final(
        cls,
        name: str,
        workflow_agent,
        *,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        policy: Optional[StepPolicy] = None,
    ) -> "WorkflowStep":
        return cls(
            name=name,
            workflow_agent=workflow_agent,
            input_builder=input_builder,
            output_key=output_key,
            policy=policy,
            is_final=True,
        )

    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        if self.input_builder is not None:
            child_input = await _resolve_callable(self.input_builder, state, context, workflow)
        else:
            child_input = state.input

        child_run = await self.workflow_agent.async_run(child_input, context=context)
        result = child_run.final_output

        updates = {}
        if self.output_key is not None:
            updates[self.output_key] = result

        final_output = result if self.is_final else None
        return StepResult(
            updates=updates,
            output=result,
            next_step=self.next_step,
            stop=self.is_final,
            final_output=final_output,
            trace_data={
                "child_workflow_name": child_run.workflow_name,
                "child_run": child_run,
            },
        )


class ParallelStep(Step):
    def __init__(
        self,
        name: str,
        steps: list[Step],
        *,
        next_step: Optional[str] = None,
        output_key: Optional[str] = None,
        merge_strategy: str = "error",
        policy: Optional[StepPolicy] = None,
        is_final: bool = False,
    ):
        super().__init__(name=name, next_step=next_step, policy=policy)
        self.steps = steps
        self.output_key = output_key
        self.merge_strategy = merge_strategy
        self.is_final = is_final

        if not self.steps:
            raise ValueError("ParallelStep requires at least one child step")
        child_step_names = [step.name for step in self.steps]
        if len(set(child_step_names)) != len(child_step_names):
            raise ValueError("ParallelStep child step names must be unique")
        invalid_child_steps = [step.name for step in self.steps if step.next_step is not None or step.is_terminal()]
        if invalid_child_steps:
            formatted = ", ".join(invalid_child_steps)
            raise ValueError(
                "ParallelStep child steps cannot declare next_step or be terminal: "
                f"{formatted}"
            )
        if self.merge_strategy not in {"error", "last_write_wins"}:
            raise ValueError("ParallelStep merge_strategy must be 'error' or 'last_write_wins'")
        if self.is_final and self.next_step is not None:
            raise ValueError("Final ParallelSteps cannot declare next_step")

    @classmethod
    def final(
        cls,
        name: str,
        steps: list[Step],
        *,
        output_key: Optional[str] = None,
        merge_strategy: str = "error",
        policy: Optional[StepPolicy] = None,
    ) -> "ParallelStep":
        return cls(
            name=name,
            steps=steps,
            output_key=output_key,
            merge_strategy=merge_strategy,
            policy=policy,
            is_final=True,
        )

    def is_terminal(self) -> bool:
        return self.is_final

    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        async def _run_child(step: Step):
            branch_state = WorkflowState(
                input=state.input,
                data=dict(state.data),
                last_output=state.last_output,
                final_output=state.final_output,
                current_step=step.name,
                completed_steps=list(state.completed_steps),
                metadata=dict(state.metadata),
            )
            try:
                result = await step.run(branch_state, context=context, workflow=workflow)
            except Exception as e:
                raise WorkflowStepError(
                    f"ParallelStep '{self.name}' branch '{step.name}' failed: {e}",
                    trace_data={
                        "parallel_failed_branch": step.name,
                        "parallel_failed_branch_state": branch_state.to_dict(),
                        "parallel_failed_branch_error": str(e),
                    },
                ) from e
            return step.name, result, branch_state

        branch_results = await asyncio.gather(*[_run_child(step) for step in self.steps])

        merged_updates: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        branch_trace: dict[str, Any] = {}
        update_sources: dict[str, str] = {}

        for step_name, result, branch_state in branch_results:
            for key, value in result.updates.items():
                if key in merged_updates and self.merge_strategy == "error":
                    previous_step = update_sources[key]
                    raise ValueError(
                        f"ParallelStep '{self.name}' received conflicting updates for key '{key}' "
                        f"from steps '{previous_step}' and '{step_name}'"
                    )
                merged_updates[key] = value
                update_sources[key] = step_name
            outputs[step_name] = result.output
            branch_trace[step_name] = {
                "output": result.output,
                "updates": dict(result.updates),
                "state_after": branch_state.to_dict(),
            }

        if self.output_key is not None:
            merged_updates[self.output_key] = outputs

        final_output = outputs if self.is_final else None
        return StepResult(
            updates=merged_updates,
            output=outputs,
            next_step=self.next_step,
            stop=self.is_final,
            final_output=final_output,
            trace_data={
                "parallel_branches": branch_trace,
                "parallel_merge_strategy": self.merge_strategy,
                "parallel_update_sources": dict(update_sources),
            },
        )


class LLMStep(Step):
    def __init__(
        self,
        name: str,
        *,
        llm,
        prompt: str | PromptBuilder,
        tools: Optional[list[BaseTool]] = None,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        result_shape: Optional[dict] = None,
        check_func: Optional[Callable[[Any], CheckResult]] = None,
        max_attempts: int = 3,
        llm_args: Optional[dict[str, Any]] = None,
        next_step: Optional[str] = None,
        response_key: Optional[str] = None,
        is_final: bool = False,
        policy: Optional[StepPolicy] = None,
    ):
        super().__init__(name=name, next_step=next_step, policy=policy)
        self.llm = llm
        self.prompt = prompt
        self.tools = tools or []
        self.input_builder = input_builder
        self.output_key = output_key
        self.result_shape = result_shape
        self.check_func = check_func
        self.max_attempts = max_attempts
        self.llm_args = llm_args or {}
        self.response_key = response_key
        self.is_final = is_final

        if self.tools and (self.result_shape or self.check_func):
            raise ValueError("LLMStep with tools does not support result_shape or check_func in this version")
        if self.is_final and self.next_step is not None:
            raise ValueError("Final LLMSteps cannot declare next_step")

    def is_terminal(self) -> bool:
        return self.is_final

    @classmethod
    def final(
        cls,
        name: str,
        *,
        llm,
        prompt: str | PromptBuilder,
        tools: Optional[list[BaseTool]] = None,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        result_shape: Optional[dict] = None,
        check_func: Optional[Callable[[Any], CheckResult]] = None,
        max_attempts: int = 3,
        llm_args: Optional[dict[str, Any]] = None,
        response_key: Optional[str] = None,
        policy: Optional[StepPolicy] = None,
    ) -> "LLMStep":
        return cls(
            name=name,
            llm=llm,
            prompt=prompt,
            tools=tools,
            input_builder=input_builder,
            output_key=output_key,
            result_shape=result_shape,
            check_func=check_func,
            max_attempts=max_attempts,
            llm_args=llm_args,
            response_key=response_key,
            is_final=True,
            policy=policy,
        )

    async def _build_prompt(self, state: WorkflowState, context: Any = None, workflow: Any = None) -> str:
        return await _resolve_callable(self.prompt, state, context, workflow)

    async def _build_input(self, state: WorkflowState, context: Any = None, workflow: Any = None) -> list[Message]:
        if self.input_builder is not None:
            built = await _resolve_callable(self.input_builder, state, context, workflow)
        else:
            built = state.input

        if isinstance(built, list):
            return built
        if isinstance(built, Message):
            return [built]
        return [UserMessage(content=str(built))]

    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        prompt = await self._build_prompt(state, context=context, workflow=workflow)
        messages = await self._build_input(state, context=context, workflow=workflow)

        if self.tools:
            agent = Agent(
                llm=self.llm,
                tools=self.tools,
                sys_prompt=prompt,
                context=context,
                json_reply=False,
                mode="async",
            )
            result = await agent.async_ask(messages, usage=getattr(workflow, "usage", None))
        else:
            func = llmfunc(
                self.llm,
                prompt=prompt,
                result_shape=self.result_shape,
                check_func=self.check_func,
                max_attempts=self.max_attempts,
                llm_args=self.llm_args,
                async_mode=True,
            )
            result = await func(messages, usage=getattr(workflow, "usage", None))

        if self.response_key and isinstance(result, dict):
            result = result.get(self.response_key)

        updates = {}
        if self.output_key is not None:
            updates[self.output_key] = result

        final_output = result if self.is_final else None
        return StepResult(
            updates=updates,
            output=result,
            next_step=self.next_step,
            stop=self.is_final,
            final_output=final_output,
        )
