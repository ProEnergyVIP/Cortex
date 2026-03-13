from __future__ import annotations

from inspect import signature
from typing import Any, Optional

from cortex.LLMFunc import llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage

from .agent import WorkflowAgent
from .node import NodePolicy, ParallelNode, RouterNode, RunnableNode
from .runtime import FunctionRunnable, function_runnable as build_function_runnable


def workflow(
    *,
    name: str,
    nodes: Optional[list[Any]] = None,
    start_node: Optional[str] = None,
    context: Any = None,
    usage: Any = None,
    max_steps: int = 50,
) -> WorkflowAgent:
    if nodes is None:
        raise ValueError("workflow(...) requires nodes")
    return WorkflowAgent(
        name=name,
        nodes=nodes,
        start_node=start_node,
        context=context,
        usage=usage,
        max_steps=max_steps,
    )


def function_runnable(
    *,
    ask,
    run=None,
    name: Optional[str] = None,
    context: Any = None,
    usage: Any = None,
) -> FunctionRunnable:
    return build_function_runnable(
        ask=ask,
        run=run,
        name=name,
        context=context,
        usage=usage,
    )


def _call_with_supported_args(func, *args, **kwargs):
    sig = signature(func)
    accepted_positional = []
    remaining_args = list(args)
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and remaining_args:
            accepted_positional.append(remaining_args.pop(0))
    accepted_kwargs = {}
    for name, param in sig.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            accepted_kwargs = kwargs
            break
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY) and name in kwargs:
            accepted_kwargs[name] = kwargs[name]
    return func(*accepted_positional, **accepted_kwargs)


async def _normalize_message_input(input_builder, state, context, workflow):
    if input_builder is not None:
        built = _call_with_supported_args(input_builder, state, context, workflow)
        if hasattr(built, "__await__"):
            built = await built
    else:
        built = state.input

    if isinstance(built, list):
        return built
    if isinstance(built, Message):
        return [built]
    return [UserMessage(content=str(built))]


def function_node(
    name: str,
    func,
    *,
    next_node: Optional[str] = None,
    output_key: Optional[str] = None,
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> RunnableNode:
    async def _ask(user_input=None, *, context=None, usage=None, parent=None):
        state = user_input
        result = _call_with_supported_args(func, state, context, parent)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return RunnableNode(
        name=name,
        runnable=function_runnable(name=name, ask=_ask),
        input_builder=lambda state, context, workflow: state,
        output_key=output_key,
        next_node=next_node,
        policy=policy,
        is_final=is_final,
    )


def router_node(
    name: str,
    func,
    *,
    next_node: Optional[str] = None,
    output_key: Optional[str] = None,
    possible_next_nodes: Optional[list[str]] = None,
    policy: Optional[NodePolicy] = None,
) -> RouterNode:
    return RouterNode(
        name=name,
        func=func,
        next_node=next_node,
        output_key=output_key,
        possible_next_nodes=possible_next_nodes,
        policy=policy,
    )


def runnable_node(
    name: str,
    runnable,
    *,
    input_builder=None,
    output_key: Optional[str] = None,
    next_node: Optional[str] = None,
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> RunnableNode:
    return RunnableNode(
        name=name,
        runnable=runnable,
        input_builder=input_builder,
        output_key=output_key,
        next_node=next_node,
        policy=policy,
        is_final=is_final,
    )


def parallel_node(
    name: str,
    nodes: Optional[list[Any]] = None,
    *,
    next_node: Optional[str] = None,
    output_key: Optional[str] = None,
    merge_strategy: str = "error",
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> ParallelNode:
    if nodes is None:
        raise ValueError("parallel_node(...) requires nodes")
    return ParallelNode(
        name=name,
        nodes=nodes,
        next_node=next_node,
        output_key=output_key,
        merge_strategy=merge_strategy,
        policy=policy,
        is_final=is_final,
    )


def llm_node(
    name: str,
    *,
    llm,
    prompt,
    tools=None,
    input_builder=None,
    output_key: Optional[str] = None,
    result_shape: Optional[dict] = None,
    check_func=None,
    max_attempts: int = 3,
    llm_args: Optional[dict[str, Any]] = None,
    next_node: Optional[str] = None,
    response_key: Optional[str] = None,
    is_final: bool = False,
    policy: Optional[NodePolicy] = None,
) -> RunnableNode:
    async def _ask(user_input=None, *, context=None, usage=None, parent=None):
        state = user_input
        resolved_prompt = prompt
        if callable(prompt):
            resolved_prompt = _call_with_supported_args(prompt, state, context, parent)
            if hasattr(resolved_prompt, "__await__"):
                resolved_prompt = await resolved_prompt

        messages = await _normalize_message_input(input_builder, state, context, parent)

        if tools:
            if result_shape or check_func:
                raise ValueError("llm_node with tools does not support result_shape or check_func in this version")
            agent = Agent(
                llm=llm,
                tools=tools,
                sys_prompt=resolved_prompt,
                context=context,
                json_reply=False,
                mode="async",
            )
            result = await agent.async_ask(messages, usage=usage)
        else:
            func_runnable = llmfunc(
                llm,
                prompt=resolved_prompt,
                result_shape=result_shape,
                check_func=check_func,
                max_attempts=max_attempts,
                llm_args=llm_args or {},
                async_mode=True,
            )
            result = await func_runnable(messages, usage=usage)

        if response_key and isinstance(result, dict):
            return result.get(response_key)
        return result

    return RunnableNode(
        name=name,
        runnable=function_runnable(name=name, ask=_ask),
        input_builder=lambda state, context, workflow: state,
        output_key=output_key,
        next_node=next_node,
        policy=policy,
        is_final=is_final,
    )
