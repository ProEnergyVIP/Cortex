from __future__ import annotations

from dataclasses import dataclass
from asyncio import iscoroutine
from inspect import signature
from typing import Any, Optional, Protocol, runtime_checkable

from cortex.message import AgentUsage


@runtime_checkable
class AskCapableRuntimeLike(Protocol):
    name: Optional[str]
    context: Any
    usage: Optional[AgentUsage]

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        ...


@runtime_checkable
class RuntimeLike(AskCapableRuntimeLike, Protocol):
    pass


@runtime_checkable
class RunResultLike(Protocol):
    final_output: Any
    engine_name: Optional[str]


@runtime_checkable
class WorkflowRunResultLike(RunResultLike, Protocol):
    workflow_name: Optional[str]


@runtime_checkable
class RunCapableRuntimeLike(AskCapableRuntimeLike, Protocol):
    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        ...


@dataclass
class RuntimeInvocation:
    output: Any
    runtime_name: Optional[str]
    run: Optional[RunResultLike] = None


@dataclass
class RuntimeAdapter:
    runtime: AskCapableRuntimeLike

    @property
    def name(self) -> Optional[str]:
        return getattr(self.runtime, "name", None)

    @property
    def context(self) -> Any:
        return getattr(self.runtime, "context", None)

    @property
    def usage(self) -> Optional[AgentUsage]:
        return getattr(self.runtime, "usage", None)

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        return await self.runtime.async_ask(user_input, **kwargs)

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        if isinstance(self.runtime, RunCapableRuntimeLike):
            return await self.runtime.async_run(user_input, **kwargs)
        raise TypeError("Adapted runtime does not support async_run(...)")


@dataclass
class FunctionRuntime:
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
            runtime=self,
            parent=kwargs.get("parent"),
        )

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        if self.run_func is None:
            raise TypeError("FunctionRuntime does not support async_run(...)")
        return await _call_with_supported_kwargs(
            self.run_func,
            user_input=user_input,
            context=kwargs.get("context", self.context),
            usage=kwargs.get("usage", self.usage),
            runtime=self,
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


async def adapt_runtime(
    runtime: Any,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RuntimeAdapter:
    resolved_runtime = await resolve_runtime(
        runtime,
        context=context,
        usage=usage,
        parent=parent,
    )
    return RuntimeAdapter(runtime=resolved_runtime)


def get_runtime_name(runtime: Any) -> Optional[str]:
    return getattr(runtime, "name", None)


def get_run_name(run: RunResultLike) -> Optional[str]:
    if isinstance(run, WorkflowRunResultLike):
        return run.workflow_name
    return getattr(run, "workflow_name", None) or run.engine_name


def get_run_output(run: Any) -> Any:
    if isinstance(run, RunResultLike):
        return run.final_output
    return getattr(run, "final_output", run)


async def invoke_runtime(
    runtime: Any,
    user_input: Any = None,
    *,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RuntimeInvocation:
    adapted_runtime = await adapt_runtime(
        runtime,
        context=context,
        usage=usage,
        parent=parent,
    )

    if isinstance(adapted_runtime.runtime, RunCapableRuntimeLike):
        runtime_run = await adapted_runtime.async_run(user_input, context=context)
        runtime_name = (
            get_run_name(runtime_run)
            if isinstance(runtime_run, RunResultLike) or hasattr(runtime_run, "final_output")
            else adapted_runtime.name
        )
        return RuntimeInvocation(
            output=get_run_output(runtime_run),
            runtime_name=runtime_name,
            run=runtime_run if isinstance(runtime_run, RunResultLike) else None,
        )

    output = await adapted_runtime.async_ask(user_input, context=context)
    return RuntimeInvocation(
        output=output,
        runtime_name=adapted_runtime.name,
    )
