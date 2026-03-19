from __future__ import annotations

from typing import Any, Callable, Optional

from cortex.LLMFunc import llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage

from .agent import WorkflowAgent
from .engine import WorkflowStateProtocol
from .node import NodePolicy, ParallelNode, RouterNode, RunnableNode
from .runtime import function_runnable, invoke_workflow_callback


def workflow(
    *,
    name: str,
    nodes: Optional[list[Any]] = None,
    start_node: Optional[str] = None,
    context: Any = None,
    usage: Any = None,
    memory: Any = None,
    max_steps: int = 50,
    state_type: Optional[type[WorkflowStateProtocol]] = None,
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None,
) -> WorkflowAgent:
    if nodes is None:
        raise ValueError("workflow(...) requires nodes")
    return WorkflowAgent(
        name=name,
        nodes=nodes,
        start_node=start_node,
        context=context,
        usage=usage,
        memory=memory,
        max_steps=max_steps,
        state_type=state_type,
        state_factory=state_factory,
    )


async def _normalize_message_input(input_builder, state, context, workflow):
    callback_input = state.data if state.data else state.input
    if input_builder is not None:
        built = await invoke_workflow_callback(
            input_builder,
            user_input=callback_input,
            context=context,
            state=state,
            workflow=workflow,
        )
    else:
        built = callback_input

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
    async def _ask(user_input=None, *, state=None, context=None, usage=None, parent=None):
        return await invoke_workflow_callback(
            func,
            user_input=user_input,
            context=context,
            state=state,
            workflow=parent,
        )

    return RunnableNode(
        name=name,
        runnable=function_runnable(name=name, ask=_ask),
        input_builder=lambda user_input, context, state=None: user_input,
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
    async def _ask(user_input=None, *, state=None, context=None, usage=None, parent=None):
        resolved_prompt = prompt
        if callable(prompt):
            resolved_prompt = await invoke_workflow_callback(
                prompt,
                user_input=user_input,
                context=context,
                state=state,
                workflow=parent,
            )

        messages = await _normalize_message_input(input_builder, state, context, parent)

        if tools:
            if result_shape or check_func:
                raise ValueError("llm_node with tools does not support result_shape or check_func in this version")
            agent = Agent(
                llm=llm,
                tools=tools,
                sys_prompt=resolved_prompt,
                context=getattr(state, "context", None),
                json_reply=False,
                mode="async",
            )
            result = await agent.async_ask(messages, usage=getattr(state, "usage", None))
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
            result = await func_runnable(messages, usage=getattr(state, "usage", None))

        if response_key and isinstance(result, dict):
            return result.get(response_key)
        return result

    return RunnableNode(
        name=name,
        runnable=function_runnable(name=name, ask=_ask),
        input_builder=lambda user_input, context, state=None: user_input,
        output_key=output_key,
        next_node=next_node,
        policy=policy,
        is_final=is_final,
    )
