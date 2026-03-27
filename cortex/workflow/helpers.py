from __future__ import annotations

from typing import Any, Callable, Optional

from cortex.LLMFunc import llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage

from .agent import WorkflowAgent
from .engine import WorkflowEdge, WorkflowStateProtocol
from .workflow_node import NodePolicy, NodeSpec, invoke_workflow_callback


def edge(source: str | Any, target: str | Any) -> WorkflowEdge:
    source_name = source.name if hasattr(source, "name") else str(source)
    target_name = target.name if hasattr(target, "name") else str(target)
    return WorkflowEdge(source=source_name, target=target_name)


def workflow(
    *,
    name: str,
    nodes: Optional[list[Any]] = None,
    edges: Optional[list[WorkflowEdge]] = None,
    start_node: Optional[str] = None,
    context: Any = None,
    memory: Any = None,
    max_steps: int = 50,
    state_type: Optional[type[WorkflowStateProtocol]] = None,
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None,
) -> WorkflowAgent:
    return WorkflowAgent(
        name=name,
        nodes=list(nodes or []),
        edges=list(edges or []),
        start_node=start_node,
        context=context,
        memory=memory,
        max_steps=max_steps,
        state_type=state_type,
        state_factory=state_factory,
    )


async def _normalize_message_input(input_builder, state, context, memory):
    if isinstance(state, dict):
        callback_input = state
    else:
        callback_input = state.data if getattr(state, "data", None) else getattr(state, "input", None)
    if input_builder is not None:
        built = await invoke_workflow_callback(
            input_builder,
            user_input=callback_input,
            context=context,
            memory=memory,
            state=state,
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
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> NodeSpec:
    """Create a function node: ``async def func(data: dict, context) -> dict``."""
    return NodeSpec(
        name=name,
        func=func,
        policy=policy or NodePolicy(),
        is_final=is_final,
    )


def node(
    name: str,
    func,
    *,
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> NodeSpec:
    """Alias of ``function_node`` for the common function-based node case."""

    return function_node(name=name, func=func, policy=policy, is_final=is_final)


def router_node(
    name: str,
    func,
    *,
    output_key: Optional[str] = None,
    possible_next_nodes: Optional[list[str]] = None,
    policy: Optional[NodePolicy] = None,
) -> NodeSpec:
    return NodeSpec(
        name=name,
        func=func,
        kind="router",
        output_key=output_key,
        possible_next_nodes=set(possible_next_nodes or []),
        policy=policy or NodePolicy(),
    )


def parallel_node(
    name: str,
    branches: Optional[dict[str, Callable]] = None,
    *,
    output_key: Optional[str] = None,
    merge_strategy: str = "error",
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> NodeSpec:
    if not branches:
        raise ValueError("parallel_node(...) requires branches")
    return NodeSpec(
        name=name,
        kind="parallel",
        branches=branches,
        output_key=output_key,
        merge_strategy=merge_strategy,
        policy=policy or NodePolicy(),
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
    response_key: Optional[str] = None,
    is_final: bool = False,
    policy: Optional[NodePolicy] = None,
) -> NodeSpec:
    """Create a function node backed by an LLM call.

    The LLM and prompt are captured lazily — nothing is created until the
    engine actually executes this node.
    """

    async def _llm_function(data: dict, context, memory=None) -> dict:
        resolved_prompt = prompt
        if callable(prompt):
            resolved_prompt = await invoke_workflow_callback(
                prompt,
                user_input=data,
                context=context,
                memory=memory,
            )

        messages = await _normalize_message_input(input_builder, data, context, memory)

        if tools:
            if result_shape or check_func:
                raise ValueError("llm_node with tools does not support result_shape or check_func")
            agent = Agent(
                llm=llm,
                tools=tools,
                sys_prompt=resolved_prompt,
                context=context,
                json_reply=False,
                mode="async",
            )
            result = await agent.async_ask(messages, usage=None)
        else:
            func_runnable = llmfunc(
                llm,
                prompt=resolved_prompt,
                result_shape=result_shape,
                check_func=check_func,
                max_attempts=max_attempts,
                llm_args=llm_args or {},
            )
            result = await func_runnable(messages, usage=None)

        updates = {}
        if output_key is not None:
            updates[output_key] = result
        elif response_key is not None:
            updates[response_key] = result
        else:
            updates["_output"] = result
        return updates

    return NodeSpec(
        name=name,
        func=_llm_function,
        policy=policy or NodePolicy(),
        is_final=is_final,
    )
