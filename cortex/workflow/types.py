from __future__ import annotations

from typing import Any, Awaitable, Callable, TypeAlias, Union

from cortex.message import Message

NodeValue: TypeAlias = Any
NodeUpdates: TypeAlias = dict[str, Any]
WorkflowMessageInput: TypeAlias = Union[Message, list[Message], Any]

PromptBuilderResult: TypeAlias = str
WorkflowCallbackResult: TypeAlias = Any
WorkflowContextCallback: TypeAlias = Callable[..., Union[WorkflowCallbackResult, Awaitable[WorkflowCallbackResult]]]
PromptBuilder: TypeAlias = Callable[..., Union[PromptBuilderResult, Awaitable[PromptBuilderResult]]]
InputBuilder: TypeAlias = WorkflowContextCallback
NodeFunctionResult: TypeAlias = Any
NodeFunction: TypeAlias = WorkflowContextCallback
RouterFunctionResult: TypeAlias = Any
RouterFunction: TypeAlias = WorkflowContextCallback
