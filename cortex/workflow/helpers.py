from __future__ import annotations

from typing import Any, Optional

from .agent import WorkflowAgent
from .node import FunctionNode, LLMNode, NodePolicy, ParallelNode, RouterNode, RunnableNode
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


def function_node(
    name: str,
    func,
    *,
    next_node: Optional[str] = None,
    output_key: Optional[str] = None,
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> FunctionNode:
    return FunctionNode(
        name=name,
        func=func,
        next_node=next_node,
        output_key=output_key,
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
) -> LLMNode:
    return LLMNode(
        name=name,
        llm=llm,
        prompt=prompt,
        tools=tools,
        input_builder=input_builder,
        output_key=output_key,
        result_shape=result_shape,
        check_func=check_func,
        max_attempts=max_attempts,
        llm_args=llm_args,
        next_node=next_node,
        response_key=response_key,
        is_final=is_final,
        policy=policy,
    )
