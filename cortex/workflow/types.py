from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeAlias, Union

from cortex.message import Message

StepValue: TypeAlias = Any
StepUpdates: TypeAlias = dict[str, Any]
WorkflowMessageInput: TypeAlias = Union[Message, list[Message], Any]

PromptBuilderResult: TypeAlias = str
PromptBuilder: TypeAlias = Callable[..., Union[PromptBuilderResult, Awaitable[PromptBuilderResult]]]
InputBuilder: TypeAlias = Callable[..., Union[WorkflowMessageInput, Awaitable[WorkflowMessageInput]]]
StepFunctionResult: TypeAlias = Any
StepFunction: TypeAlias = Callable[..., Union[StepFunctionResult, Awaitable[StepFunctionResult]]]
RouterFunctionResult: TypeAlias = Any
RouterFunction: TypeAlias = Callable[..., Union[RouterFunctionResult, Awaitable[RouterFunctionResult]]]
