from __future__ import annotations

from typing import Any, Optional

from .agent import WorkflowAgent
from .policy import StepPolicy
from .runtime import FunctionRuntime
from .step import FunctionStep, LLMStep, ParallelStep, RouterStep, RuntimeNode


def workflow(
    *,
    name: str,
    nodes: Optional[list[Any]] = None,
    steps: Optional[list[Any]] = None,
    start_node: Optional[str] = None,
    start_step: Optional[str] = None,
    context: Any = None,
    usage: Any = None,
    max_steps: int = 50,
) -> WorkflowAgent:
    resolved_nodes = steps if steps is not None else nodes
    if resolved_nodes is None:
        raise ValueError("workflow(...) requires steps or nodes")
    resolved_start = start_step if start_step is not None else start_node
    return WorkflowAgent(
        name=name,
        nodes=resolved_nodes,
        start_node=resolved_start,
        context=context,
        usage=usage,
        max_steps=max_steps,
    )


def function_runtime(
    *,
    ask,
    run=None,
    name: Optional[str] = None,
    context: Any = None,
    usage: Any = None,
) -> FunctionRuntime:
    return FunctionRuntime(
        ask_func=ask,
        run_func=run,
        name=name,
        context=context,
        usage=usage,
    )


def function_node(
    name: str,
    func,
    *,
    next_node: Optional[str] = None,
    next_step: Optional[str] = None,
    output_key: Optional[str] = None,
    policy: Optional[StepPolicy] = None,
    is_final: bool = False,
) -> FunctionStep:
    resolved_next = next_step if next_step is not None else next_node
    return FunctionStep(
        name=name,
        func=func,
        next_step=resolved_next,
        output_key=output_key,
        policy=policy,
        is_final=is_final,
    )


def router_node(
    name: str,
    func,
    *,
    next_node: Optional[str] = None,
    next_step: Optional[str] = None,
    output_key: Optional[str] = None,
    possible_next_nodes: Optional[list[str]] = None,
    possible_next_steps: Optional[list[str]] = None,
    policy: Optional[StepPolicy] = None,
) -> RouterStep:
    resolved_next = next_step if next_step is not None else next_node
    resolved_possible = possible_next_steps if possible_next_steps is not None else possible_next_nodes
    return RouterStep(
        name=name,
        func=func,
        next_step=resolved_next,
        output_key=output_key,
        possible_next_steps=resolved_possible,
        policy=policy,
    )


def runtime_node(
    name: str,
    runtime,
    *,
    input_builder=None,
    output_key: Optional[str] = None,
    next_node: Optional[str] = None,
    next_step: Optional[str] = None,
    policy: Optional[StepPolicy] = None,
    is_final: bool = False,
) -> RuntimeNode:
    resolved_next = next_step if next_step is not None else next_node
    return RuntimeNode(
        name=name,
        runtime=runtime,
        input_builder=input_builder,
        output_key=output_key,
        next_step=resolved_next,
        policy=policy,
        is_final=is_final,
    )


def parallel_node(
    name: str,
    nodes: Optional[list[Any]] = None,
    *,
    steps: Optional[list[Any]] = None,
    next_node: Optional[str] = None,
    next_step: Optional[str] = None,
    output_key: Optional[str] = None,
    merge_strategy: str = "error",
    policy: Optional[StepPolicy] = None,
    is_final: bool = False,
) -> ParallelStep:
    resolved_steps = steps if steps is not None else nodes
    if resolved_steps is None:
        raise ValueError("parallel_node(...) requires steps or nodes")
    resolved_next = next_step if next_step is not None else next_node
    return ParallelStep(
        name=name,
        steps=resolved_steps,
        next_step=resolved_next,
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
    next_step: Optional[str] = None,
    response_key: Optional[str] = None,
    is_final: bool = False,
    policy: Optional[StepPolicy] = None,
) -> LLMStep:
    resolved_next = next_step if next_step is not None else next_node
    return LLMStep(
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
        next_step=resolved_next,
        response_key=response_key,
        is_final=is_final,
        policy=policy,
    )
