"""Public workflow API surface.

This module re-exports the workflow engine, node types, helper constructors, runtime
adapters, and shared types that make up the supported `cortex.workflow` interface.
"""

from .agent import WorkflowAgent
from .helpers import function_node, llm_node, parallel_node, router_node, runnable_node, workflow
from .node import FailureStrategy, Node, NodePolicy, ParallelNode, RouterNode, RunnableNode, WorkflowNodeResult
from .runtime import (
    AskCapableRunnableLike,
    FunctionRunnable,
    RunnableAdapter,
    RunnableInvocation,
    RunnableLike,
    RunCapableRunnableLike,
    RunResultLike,
    adapt_runnable,
    function_runnable,
    get_run_name,
    get_runnable_name,
    invoke_runnable,
    resolve_runnable,
)
from .state import NodeTrace, WorkflowRun, WorkflowState, WorkflowStateProtocol
from .types import InputBuilder, NodeFunction, NodeUpdates, NodeValue, PromptBuilder, RouterFunction, WorkflowMessageInput

# Keep exports explicit so the public workflow surface stays easy to inspect and stable
# for users importing from `cortex.workflow`.
__all__ = [
    "WorkflowAgent",
    "workflow",
    "function_runnable",
    "function_node",
    "router_node",
    "parallel_node",
    "runnable_node",
    "llm_node",
    "AskCapableRunnableLike",
    "FunctionRunnable",
    "RunnableAdapter",
    "RunnableInvocation",
    "RunnableLike",
    "RunCapableRunnableLike",
    "RunResultLike",
    "adapt_runnable",
    "get_runnable_name",
    "get_run_name",
    "resolve_runnable",
    "invoke_runnable",
    "NodePolicy",
    "FailureStrategy",
    "WorkflowState",
    "WorkflowStateProtocol",
    "WorkflowRun",
    "NodeTrace",
    "Node",
    "WorkflowNodeResult",
    "ParallelNode",
    "RouterNode",
    "RunnableNode",
    "PromptBuilder",
    "InputBuilder",
    "NodeFunction",
    "RouterFunction",
    "NodeValue",
    "NodeUpdates",
    "WorkflowMessageInput",
]
