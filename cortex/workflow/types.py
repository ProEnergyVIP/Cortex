from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeAlias, Union

from cortex.message import Message

NodeValue: TypeAlias = Any
NodeUpdates: TypeAlias = dict[str, Any]
WorkflowMessageInput: TypeAlias = Union[Message, list[Message], Any]

PromptBuilderResult: TypeAlias = str
PromptBuilder: TypeAlias = Callable[..., Union[PromptBuilderResult, Awaitable[PromptBuilderResult]]]
InputBuilder: TypeAlias = Callable[..., Union[WorkflowMessageInput, Awaitable[WorkflowMessageInput]]]
NodeFunctionResult: TypeAlias = Any
NodeFunction: TypeAlias = Callable[..., Union[NodeFunctionResult, Awaitable[NodeFunctionResult]]]
RouterFunctionResult: TypeAlias = Any
RouterFunction: TypeAlias = Callable[..., Union[RouterFunctionResult, Awaitable[RouterFunctionResult]]]
