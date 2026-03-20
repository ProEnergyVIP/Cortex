from __future__ import annotations

from typing import Any, Callable, Optional

from cortex.LLMFunc import llmfunc
from cortex.agent import Agent
from cortex.message import Message, UserMessage

from .agent import WorkflowAgent
from .engine import WorkflowEdge, WorkflowStateProtocol
from .node import FunctionNode, NodePolicy, ParallelNode, RouterNode, invoke_workflow_callback


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
    usage: Any = None,
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
    policy: Optional[NodePolicy] = None,
    is_final: bool = False,
) -> FunctionNode:
    """Create a node that executes a plain function with the simplified contract.
    
    The function should have the signature: async def func(data: dict, context) -> dict
    """
    return FunctionNode(
        name=name,
        func=func,
        policy=policy,
        is_final=is_final,
    )


def router_node(
    name: str,
    func,
    *,
    output_key: Optional[str] = None,
    possible_next_nodes: Optional[list[str]] = None,
    policy: Optional[NodePolicy] = None,
) -> RouterNode:
    return RouterNode(
        name=name,
        func=func,
        output_key=output_key,
        possible_next_nodes=possible_next_nodes,
        policy=policy,
    )


def parallel_node(
    name: str,
    nodes: Optional[list[Any]] = None,
    *,
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
    response_key: Optional[str] = None,
    is_final: bool = False,
    policy: Optional[NodePolicy] = None,
) -> FunctionNode:
    """Create a node that executes an LLM with the simplified contract."""
    
    async def _llm_function(data: dict, context) -> dict:
        # Resolve prompt if it's callable
        resolved_prompt = prompt
        if callable(prompt):
            # Create a mock state for the callback
            class MockState:
                def __init__(self, data_dict):
                    self.data = data_dict
                    self.context = context
                def get(self, key, default=None):
                    return self.data.get(key, default)
            mock_state = MockState(data)
            
            resolved_prompt = await invoke_workflow_callback(
                prompt,
                user_input=data,
                context=context,
                state=mock_state,
                workflow=None,
            )

        # Normalize messages
        messages = await _normalize_message_input(input_builder, mock_state, context, None)

        # Execute LLM
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
            result = await agent.async_ask(messages, usage=getattr(mock_state, "usage", None))
        else:
            func_runnable = llmfunc(
                llm,
                prompt=resolved_prompt,
                result_shape=result_shape,
                check_func=check_func,
                max_attempts=max_attempts,
                llm_args=llm_args or {},
            )
            result = await func_runnable.async_ask(messages, usage=getattr(mock_state, "usage", None))

        # Return updates
        updates = {}
        if output_key is not None:
            updates[output_key] = result
        elif response_key is not None:
            updates[response_key] = result
        else:
            updates["_output"] = result
            
        return updates
    
    return FunctionNode(
        name=name,
        func=_llm_function,
        policy=policy,
        is_final=is_final,
    )
