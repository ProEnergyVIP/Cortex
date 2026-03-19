"""Runnable adaptation utilities used by workflow nodes.

The workflow layer accepts agents, workflows, and plain functions. This module normalizes
those shapes into a small runnable interface so `RunnableNode` can invoke them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from asyncio import iscoroutine
from inspect import signature
from typing import Any, Optional, Protocol, runtime_checkable

from cortex.message import AgentUsage


@runtime_checkable
class AskCapableRunnableLike(Protocol):
    """Protocol for runnable-like objects that can answer asynchronously."""

    name: Optional[str]

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        ...


@runtime_checkable
class RunnableLike(AskCapableRunnableLike, Protocol):
    """Marker protocol for the minimum runnable surface workflows require."""

    pass


@runtime_checkable
class RunResultLike(Protocol):
    """Protocol for structured results returned by `async_run(...)`."""

    final_output: Any
    engine_name: Optional[str]


@dataclass
class RunnableInvocation:
    """Normalized output returned by `invoke_runnable(...)`."""

    output: Any
    runnable_name: Optional[str]
    run: Optional[RunResultLike] = None


@dataclass
class RunnableAdapter:
    """Thin adapter that gives resolved runnables a uniform interface."""

    runnable: AskCapableRunnableLike

    @property
    def name(self) -> Optional[str]:
        return getattr(self.runnable, "name", None)

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        """Forward `async_ask(...)` to the wrapped runnable."""

        return await _call_with_supported_kwargs(
            self.runnable.async_ask,
            user_input=user_input,
            **kwargs,
        )

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        """Forward `async_run(...)` when the wrapped runnable supports it."""

        if isinstance(self.runnable, RunCapableRunnableLike):
            return await _call_with_supported_kwargs(
                self.runnable.async_run,
                user_input=user_input,
                **kwargs,
            )
        raise TypeError("Adapted runnable does not support async_run(...)")


@runtime_checkable
class RunCapableRunnableLike(AskCapableRunnableLike, Protocol):
    """Protocol for runnable-like objects that expose `async_run(...)`."""

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        ...


@dataclass
class FunctionRunnable:
    """Simple runnable wrapper around plain callables."""

    ask_func: Any
    run_func: Any = None
    name: Optional[str] = None

    async def async_ask(self, user_input: Any = None, **kwargs) -> Any:
        """Invoke the ask function with only the supported keyword arguments."""

        return await _call_with_supported_kwargs(
            self.ask_func,
            user_input=user_input,
            state=kwargs.get("state"),
            context=kwargs.get("context"),
            parent=kwargs.get("parent"),
        )

    async def async_run(self, user_input: Any = None, **kwargs) -> Any:
        """Invoke the run function when one is configured."""

        if self.run_func is None:
            raise TypeError("FunctionRunnable does not support async_run(...)")
        return await _call_with_supported_kwargs(
            self.run_func,
            user_input=user_input,
            state=kwargs.get("state"),
            context=kwargs.get("context"),
            runnable=self,
            parent=kwargs.get("parent"),
        )


async def _call_with_supported_kwargs(func, **kwargs):
    """Call a sync or async function with only the kwargs it accepts."""

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


async def invoke_workflow_callback(
    func,
    *,
    user_input: Any = None,
    context: Any = None,
    state: Any = None,
    workflow: Any = None,
):
    """Invoke workflow callbacks with compatibility for both modern and legacy signatures."""

    sig = signature(func)
    params = list(sig.parameters.values())
    positional_params = [
        param
        for param in params
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    legacy_state_first = bool(positional_params) and positional_params[0].name == "state" and "user_input" not in sig.parameters

    modern_values = {
        "user_input": user_input,
        "context": context,
        "state": state,
        "workflow": workflow,
    }
    legacy_values = {
        "state": state,
        "context": context,
        "workflow": workflow,
        "user_input": user_input,
    }

    positional_args = []
    accepted_kwargs = {}
    for param in params:
        if param.kind == param.VAR_POSITIONAL:
            continue
        if param.kind == param.VAR_KEYWORD:
            accepted_kwargs = legacy_values if legacy_state_first else modern_values
            break
        value_map = legacy_values if legacy_state_first else modern_values
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            if param.name in value_map:
                positional_args.append(value_map[param.name])
            continue
        if param.kind == param.KEYWORD_ONLY and param.name in value_map:
            accepted_kwargs[param.name] = value_map[param.name]

    result = func(*positional_args, **accepted_kwargs)
    if iscoroutine(result):
        return await result
    return result


async def resolve_runnable(runnable: Any) -> RunnableLike:
    """Resolve a runnable-like object from either a concrete runnable or a callable."""

    current = runnable

    while True:
        if isinstance(current, RunnableLike):
            return current
        if not callable(current):
            raise TypeError(
                "Runnable must be a concrete runnable with async_ask(...) or a callable that returns one"
            )
        # Plain functions are adapted lazily into a runnable shape instead of requiring
        # callers to wrap them manually.
        return FunctionRunnable(
            ask_func=current,
            name=getattr(current, "__name__", None),
        )


async def adapt_runnable(
    runnable: Any,
) -> RunnableAdapter:
    """Resolve a runnable and wrap it in a uniform adapter."""

    resolved_runnable = await resolve_runnable(runnable)
    return RunnableAdapter(runnable=resolved_runnable)


def get_runnable_name(runnable: Any) -> Optional[str]:
    """Return the display name for a runnable when available."""

    return getattr(runnable, "name", None)


def get_run_name(run: RunResultLike) -> Optional[str]:
    """Return the display name for a structured run result."""

    return run.engine_name


def get_run_output(run: Any) -> Any:
    """Extract the output from either a structured run object or a plain value."""

    if isinstance(run, RunResultLike):
        return run.final_output
    return getattr(run, "final_output", run)


def supports_async_run(runnable: Any) -> bool:
    """Return whether a runnable can be safely invoked via `async_run(...)`."""

    if isinstance(runnable, FunctionRunnable):
        return runnable.run_func is not None
    return isinstance(runnable, RunCapableRunnableLike)


async def invoke_runnable(
    runnable: Any,
    user_input: Any = None,
    *,
    state: Any = None,
    context: Any = None,
    usage: Optional[AgentUsage] = None,
    parent: Any = None,
) -> RunnableInvocation:
    """Invoke a runnable, preferring `async_run(...)` when available."""

    adapted_runnable = await adapt_runnable(runnable)

    if supports_async_run(adapted_runnable.runnable):
        # Structured runs preserve nested run metadata for workflow traces.
        runnable_run = await adapted_runnable.async_run(
            user_input,
            state=state,
            context=context,
            usage=usage,
            parent=parent,
        )
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

    # Ask-only runnables still participate uniformly, but do not produce nested run data.
    output = await adapted_runnable.async_ask(
        user_input,
        state=state,
        context=context,
        usage=usage,
        parent=parent,
    )
    return RunnableInvocation(
        output=output,
        runnable_name=adapted_runnable.name,
    )


def function_runnable(
    *,
    ask,
    run=None,
    name: Optional[str] = None,
) -> FunctionRunnable:
    """Create a `FunctionRunnable` from plain callables."""

    return FunctionRunnable(
        ask_func=ask,
        run_func=run,
        name=name,
    )
