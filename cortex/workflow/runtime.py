from __future__ import annotations

from asyncio import iscoroutine
from inspect import signature
from typing import Any, Optional, Protocol, runtime_checkable

from cortex.message import AgentUsage


@runtime_checkable
class RuntimeLike(Protocol):
    name: Optional[str]
    context: Any
    usage: Optional[AgentUsage]

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        ...


async def _call_with_supported_kwargs(func, **kwargs):
    sig = signature(func)
    accepted_kwargs = {}

    for name, param in sig.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL):
            continue
        if param.kind == param.VAR_KEYWORD:
            accepted_kwargs = kwargs
            break
        if name in kwargs:
            accepted_kwargs[name] = kwargs[name]

    result = func(**accepted_kwargs)
    if iscoroutine(result):
        return await result
    return result


async def resolve_runtime(
    runtime: Any,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RuntimeLike:
    current = runtime

    while True:
        if isinstance(current, RuntimeLike):
            return current
        if not callable(current):
            raise TypeError(
                "Runtime must be a concrete runtime with async_ask(...) or a callable that returns one"
            )
        current = await _call_with_supported_kwargs(
            current,
            context=context,
            usage=usage,
            parent=parent,
        )
