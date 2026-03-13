from __future__ import annotations

from dataclasses import dataclass
from asyncio import iscoroutine
from inspect import signature
from typing import Any, Optional, Protocol, runtime_checkable

from cortex.message import AgentUsage


@runtime_checkable
class AskCapableRunnableLike(Protocol):
    name: Optional[str]
    context: Any
    usage: Optional[AgentUsage]

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        ...


@runtime_checkable
class RunnableLike(AskCapableRunnableLike, Protocol):
    pass


@runtime_checkable
class RunResultLike(Protocol):
    final_output: Any
    engine_name: Optional[str]


class RunnableInvocation:
    output: Any
    runnable_name: Optional[str]
    run: Optional[RunResultLike] = None


@dataclass
class RunnableAdapter:
    runnable: AskCapableRunnableLike

    @property
    def name(self) -> Optional[str]:
        return getattr(self.runnable, "name", None)

    @property
    def context(self) -> Any:
        return getattr(self.runnable, "context", None)

    @property
    def usage(self) -> Optional[AgentUsage]:
        return getattr(self.runnable, "usage", None)

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        return await self.runnable.async_ask(user_input, **kwargs)

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        if isinstance(self.runnable, RunCapableRunnableLike):
            return await self.runnable.async_run(user_input, **kwargs)
        raise TypeError("Adapted runnable does not support async_run(...)")


@runtime_checkable
class RunCapableRunnableLike(AskCapableRunnableLike, Protocol):
    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        ...


@dataclass
class FunctionRunnable:
    ask_func: Any
    run_func: Any = None
    name: Optional[str] = None
    context: Any = None
    usage: Optional[AgentUsage] = None

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        return await _call_with_supported_kwargs(
            self.ask_func,
            user_input=user_input,
            context=kwargs.get("context", self.context),
            usage=kwargs.get("usage", self.usage),
            parent=kwargs.get("parent"),
        )

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        if self.run_func is None:
            raise TypeError("FunctionRunnable does not support async_run(...)")
        return await _call_with_supported_kwargs(
            self.run_func,
            user_input=user_input,
            context=kwargs.get("context", self.context),
            usage=kwargs.get("usage", self.usage),
            runnable=self,
            parent=kwargs.get("parent"),
        )


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


async def resolve_runnable(
    runnable: Any,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RunnableLike:
    current = runnable

    while True:
        if isinstance(current, RunnableLike):
            return current
        if not callable(current):
            raise TypeError(
                "Runnable must be a concrete runnable with async_ask(...) or a callable that returns one"
            )
        return FunctionRunnable(
            ask_func=current,
            name=getattr(current, "__name__", None),
            context=context,
            usage=usage,
        )


async def adapt_runnable(
    runnable: Any,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RunnableAdapter:
    resolved_runnable = await resolve_runnable(
        runnable,
        context=context,
        usage=usage,
        parent=parent,
    )
    return RunnableAdapter(runnable=resolved_runnable)


def get_runnable_name(runnable: Any) -> Optional[str]:
    return getattr(runnable, "name", None)


def get_run_name(run: RunResultLike) -> Optional[str]:
    return run.engine_name


def get_run_output(run: Any) -> Any:
    if isinstance(run, RunResultLike):
        return run.final_output
    return getattr(run, "final_output", run)


async def invoke_runnable(
    runnable: Any,
    user_input: Any = None,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RunnableInvocation:
    adapted_runnable = await adapt_runnable(
        runnable,
        context=context,
        usage=usage,
        parent=parent,
    )

    if isinstance(adapted_runnable.runnable, RunCapableRunnableLike):
        runnable_run = await adapted_runnable.async_run(user_input, context=context)
        runnable_name = (
            get_run_name(runnable_run)
            if isinstance(runnable_run, RunResultLike) or hasattr(runnable_run, "final_output")
            else adapted_runnable.name
        )
        return RunnableInvocation(
            output=get_run_output(runnable_run),
            runnable_name=runnable_name,
            run=runnable_run if isinstance(runnable_run, RunResultLike) else None,
        )

    output = await adapted_runnable.async_ask(user_input, context=context)
    return RunnableInvocation(
        output=output,
        runnable_name=adapted_runnable.name,
    )


def function_runnable(
    *,
    ask,
    run=None,
    name: Optional[str] = None,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
) -> FunctionRunnable:
    return FunctionRunnable(
        ask_func=ask,
        run_func=run,
        name=name,
        context=context,
        usage=usage,
    )
