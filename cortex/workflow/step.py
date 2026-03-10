from __future__ import annotations

from abc import ABC, abstractmethod
from asyncio import iscoroutine
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable, Optional

from cortex.LLMFunc import CheckResult, llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage
from cortex.tool import BaseTool

from .state import WorkflowState


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


@dataclass
class StepResult:
    updates: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    next_step: Optional[str] = None
    stop: bool = False
    final_output: Any = None

    def apply(self, state: WorkflowState) -> None:
        state.update(self.updates)
        if self.output is not None:
            state.last_output = self.output
        if self.final_output is not None:
            state.final_output = self.final_output


class Step(ABC):
    def __init__(self, name: str, next_step: Optional[str] = None):
        self.name = name
        self.next_step = next_step

    @abstractmethod
    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        raise NotImplementedError


class FunctionStep(Step):
    def __init__(self, name: str, func: Callable, next_step: Optional[str] = None, output_key: Optional[str] = None):
        super().__init__(name=name, next_step=next_step)
        self.func = func
        self.output_key = output_key

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
    async def run(self, state: WorkflowState, *, context: Any = None, workflow: Any = None) -> StepResult:
        result = self.func(state, context, workflow)
        if iscoroutine(result):
            result = await result

        if isinstance(result, StepResult):
            return result

        if result is None:
            return StepResult(next_step=self.next_step)

        return StepResult(output=result, next_step=str(result))


class LLMStep(Step):
    def __init__(
        self,
        name: str,
        *,
        llm,
        prompt: str | Callable,
        tools: Optional[list[BaseTool]] = None,
        input_builder: Optional[Callable] = None,
        output_key: Optional[str] = None,
        result_shape: Optional[dict] = None,
        check_func: Optional[Callable[[Any], CheckResult]] = None,
        max_attempts: int = 3,
        llm_args: Optional[dict[str, Any]] = None,
        next_step: Optional[str] = None,
        response_key: Optional[str] = None,
        is_final: bool = False,
    ):
        super().__init__(name=name, next_step=next_step)
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

    async def _build_prompt(self, state: WorkflowState, context: Any = None, workflow: Any = None) -> str:
        return await _resolve_callable(self.prompt, state, context, workflow)

    async def _build_input(self, state: WorkflowState, context: Any = None, workflow: Any = None):
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
