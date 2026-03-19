"""Core workflow node types and result objects.

This module defines the abstract node contract plus the concrete node implementations
used by the workflow engine: routing nodes, runnable-backed nodes, and parallel nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from asyncio import iscoroutine
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Literal, Optional, Protocol

from .runtime import invoke_runnable
from .types import InputBuilder, RouterFunction

FailureStrategy = Literal["raise", "fallback"]


@dataclass(frozen=True)
class NodePolicy:
    """Runtime policy controlling retries, fallback behavior, and timeouts."""

    max_retries: int = 0
    failure_strategy: FailureStrategy = "raise"
    fallback_node: Optional[str] = None
    timeout_seconds: Optional[float] = None

    def __post_init__(self):
        """Validate that retry, timeout, and fallback settings are internally consistent."""

        if self.max_retries < 0:
            raise ValueError("NodePolicy.max_retries must be >= 0")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("NodePolicy.timeout_seconds must be > 0 when provided")
        if self.failure_strategy == "fallback" and not self.fallback_node:
            raise ValueError("NodePolicy with failure_strategy='fallback' requires fallback_node")
        if self.failure_strategy != "fallback" and self.fallback_node is not None:
            raise ValueError("fallback_node can only be set when failure_strategy='fallback'")


class EngineNode(Protocol):
    """Structural protocol implemented by node types that the engine can execute."""

    name: str
    next_node: Optional[str]
    policy: NodePolicy

    async def run(self, state, *, context: Any = None, workflow: Any = None):
        ...

    def declared_next_nodes(self) -> set[str]:
        ...

    def is_terminal(self) -> bool:
        ...


async def _resolve_callable(value, *args):
    """Resolve a static value or invoke a sync/async builder with positional args."""

    if callable(value):
        sig = signature(value)
        count = len(sig.parameters)
        result = value(*args[:count]) if count > 0 else value()
        if iscoroutine(result):
            return await result
        return result
    return value


def _call_with_supported_kwargs(func, *args, **kwargs):
    """Invoke a callable with positional args plus only the keyword args it accepts."""

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


class WorkflowNodeError(Exception):
    """Workflow execution error that carries structured trace metadata."""

    def __init__(self, message: str, *, trace_data: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.trace_data = trace_data or {}


@dataclass
class WorkflowNodeResult:
    """Normalized result object returned by workflow nodes."""

    updates: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    next_node: Optional[str] = None
    stop: bool = False
    final_output: Any = None
    trace_data: dict[str, Any] = field(default_factory=dict)

    def apply(self, state) -> None:
        """Apply updates and output fields to the shared workflow state."""

        state.update(self.updates)
        if self.output is not None:
            state.set_output(self.output)
        if self.final_output is not None:
            state.set_final_output(self.final_output)

    @classmethod
    def next(cls, node_name: str, *, output: Any = None, updates: Optional[dict[str, Any]] = None) -> "WorkflowNodeResult":
        """Create a result that routes execution to another node."""

        return cls(updates=updates or {}, output=output, next_node=node_name)

    @classmethod
    def finish(
        cls,
        output: Any = None,
        *,
        updates: Optional[dict[str, Any]] = None,
        final_output: Any = None,
    ) -> "WorkflowNodeResult":
        """Create a terminal result that stops the workflow."""

        return cls(
            updates=updates or {},
            output=output,
            stop=True,
            final_output=output if final_output is None else final_output,
        )

    @classmethod
    def update_state(
        cls,
        updates: dict[str, Any],
        *,
        output: Any = None,
        next_node: Optional[str] = None,
    ) -> "WorkflowNodeResult":
        """Create a result that only updates state and optionally routes onward."""

        return cls(updates=updates, output=output, next_node=next_node)


class Node(ABC):
    """Abstract base class for all workflow nodes."""

    def __init__(self, name: str, next_node: Optional[str] = None, policy: Optional[NodePolicy] = None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Node name must be a non-empty string")
        self.name = name
        self.next_node = next_node
        self.policy = policy or NodePolicy()

    def declared_next_nodes(self) -> set[str]:
        """Return statically declared successor nodes for validation and graph tooling."""

        return {self.next_node} if self.next_node is not None else set()

    def is_terminal(self) -> bool:
        """Return whether this node always terminates the workflow."""

        return False

    @abstractmethod
    async def run(self, state, *, context: Any = None, workflow: Any = None) -> WorkflowNodeResult:
        raise NotImplementedError


class RouterNode(Node):
    """Run a router function that decides the next node."""

    def __init__(
        self,
        name: str,
        func: RouterFunction,
        next_node: Optional[str] = None,
        output_key: Optional[str] = None,
        possible_next_nodes: Optional[list[str]] = None,
        policy: Optional[NodePolicy] = None,
    ):
        super().__init__(name=name, next_node=next_node, policy=policy)
        self.func = func
        self.output_key = output_key
        self.possible_next_nodes = set(possible_next_nodes or [])

    def declared_next_nodes(self) -> set[str]:
        """Return both the default successor and any explicitly declared route targets."""

        return super().declared_next_nodes().union(self.possible_next_nodes)

    async def run(self, state, *, context: Any = None, workflow: Any = None) -> WorkflowNodeResult:
        """Execute the router and normalize its output into a workflow result."""

        result = _call_with_supported_kwargs(
            self.func,
            state,
            workflow,
            context=getattr(state, "context", None),
            usage=getattr(state, "usage", None),
            memory=getattr(state, "memory", None),
        )
        if iscoroutine(result):
            result = await result

        if isinstance(result, WorkflowNodeResult):
            return result

        if result is None:
            return WorkflowNodeResult(next_node=self.next_node)

        output = result
        next_node = str(result)
        updates = {self.output_key: output} if self.output_key is not None else {}
        return WorkflowNodeResult(updates=updates, output=output, next_node=next_node)


class RunnableNode(Node):
    """Execute a nested runnable and map its output back into workflow state."""

    def __init__(
        self,
        name: str,
        runnable,
        *,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        next_node: Optional[str] = None,
        policy: Optional[NodePolicy] = None,
        is_final: bool = False,
    ):
        super().__init__(name=name, next_node=next_node, policy=policy)
        self.runnable = runnable
        self.input_builder = input_builder
        self.output_key = output_key
        self.is_final = is_final

        if self.is_final and self.next_node is not None:
            raise ValueError("Final RunnableNodes cannot declare next_node")

    @classmethod
    def final(
        cls,
        name: str,
        runnable,
        *,
        input_builder: Optional[InputBuilder] = None,
        output_key: Optional[str] = None,
        policy: Optional[NodePolicy] = None,
    ) -> "RunnableNode":
        """Construct a terminal runnable node."""

        return cls(
            name=name,
            runnable=runnable,
            input_builder=input_builder,
            output_key=output_key,
            policy=policy,
            is_final=True,
        )

    def is_terminal(self) -> bool:
        return self.is_final

    async def run(self, state, *, context: Any = None, workflow: Any = None) -> WorkflowNodeResult:
        """Build child input, invoke the runnable, and normalize the child output."""

        child_input = (
            await _resolve_callable(
                lambda *args: _call_with_supported_kwargs(
                    self.input_builder,
                    *args,
                    context=getattr(state, "context", None),
                    usage=getattr(state, "usage", None),
                    memory=getattr(state, "memory", None),
                ),
                state,
                workflow,
            )
            if self.input_builder is not None
            else state.input
        )
        child_invocation = await invoke_runnable(
            self.runnable,
            child_input,
            context=getattr(state, "context", None),
            usage=getattr(state, "usage", None),
            parent=workflow,
        )
        result = child_invocation.output
        if isinstance(result, WorkflowNodeResult):
            # Child workflows may already return a fully normalized node result. In that
            # case, this wrapper only fills in defaults and attaches nested trace metadata.
            if result.next_node is None:
                result.next_node = self.next_node
            if self.is_final and not result.stop:
                result.stop = True
                if result.final_output is None:
                    result.final_output = result.output
            if child_invocation.run is not None:
                result.trace_data = {
                    **result.trace_data,
                    "child_runnable_name": child_invocation.runnable_name,
                    "child_run": child_invocation.run,
                }
            elif child_invocation.runnable_name is not None:
                result.trace_data = {
                    **result.trace_data,
                    "child_runnable_name": child_invocation.runnable_name,
                }
            return result

        # Plain outputs are wrapped into a standard result so the engine only has to reason
        # about one result shape.
        child_trace_payload = {"child_runnable_name": child_invocation.runnable_name}
        if child_invocation.run is not None:
            child_trace_payload["child_run"] = child_invocation.run
        updates = {self.output_key: result} if self.output_key is not None else {}
        final_output = result if self.is_final else None
        return WorkflowNodeResult(
            updates=updates,
            output=result,
            next_node=self.next_node,
            stop=self.is_final,
            final_output=final_output,
            trace_data=child_trace_payload,
        )


class ParallelNode(Node):
    """Execute a set of child nodes concurrently and merge their outputs."""

    def __init__(
        self,
        name: str,
        nodes: list[Node],
        *,
        next_node: Optional[str] = None,
        output_key: Optional[str] = None,
        merge_strategy: str = "error",
        policy: Optional[NodePolicy] = None,
        is_final: bool = False,
    ):
        super().__init__(name=name, next_node=next_node, policy=policy)
        self.nodes = nodes
        self.output_key = output_key
        self.merge_strategy = merge_strategy
        self.is_final = is_final

        if not self.nodes:
            raise ValueError("ParallelNode requires at least one child node")
        child_node_names = [node.name for node in self.nodes]
        if len(set(child_node_names)) != len(child_node_names):
            raise ValueError("ParallelNode child node names must be unique")
        invalid_child_nodes = [node.name for node in self.nodes if node.next_node is not None or node.is_terminal()]
        if invalid_child_nodes:
            formatted = ", ".join(invalid_child_nodes)
            raise ValueError(f"ParallelNode child nodes cannot declare next_node or be terminal: {formatted}")
        if self.merge_strategy not in {"error", "last_write_wins"}:
            raise ValueError("ParallelNode merge_strategy must be 'error' or 'last_write_wins'")
        if self.is_final and self.next_node is not None:
            raise ValueError("Final ParallelNodes cannot declare next_node")

    @classmethod
    def final(
        cls,
        name: str,
        nodes: list[Node],
        *,
        output_key: Optional[str] = None,
        merge_strategy: str = "error",
        policy: Optional[NodePolicy] = None,
    ) -> "ParallelNode":
        """Construct a terminal parallel node."""

        return cls(name=name, nodes=nodes, output_key=output_key, merge_strategy=merge_strategy, policy=policy, is_final=True)

    def is_terminal(self) -> bool:
        return self.is_final

    async def run(self, state, *, context: Any = None, workflow: Any = None) -> WorkflowNodeResult:
        """Run child nodes concurrently and merge their outputs into one result."""

        async def _run_child(node: Node):
            # Each branch gets a copy of state so concurrent child execution does not mutate
            # shared state in place.
            branch_state = state.clone(
                current_node=node.name,
            )
            try:
                result = await node.run(branch_state, context=context, workflow=workflow)
            except Exception as e:
                raise WorkflowNodeError(
                    f"ParallelNode '{self.name}' branch '{node.name}' failed: {e}",
                    trace_data={
                        "parallel_failed_branch": node.name,
                        "parallel_failed_branch_state": branch_state.to_dict(),
                        "parallel_failed_branch_error": str(e),
                    },
                ) from e
            return node.name, result, branch_state

        branch_results = await asyncio.gather(*[_run_child(node) for node in self.nodes])

        merged_updates: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        branch_trace: dict[str, Any] = {}
        update_sources: dict[str, str] = {}

        for node_name, result, branch_state in branch_results:
            for key, value in result.updates.items():
                # Merge conflicts are either rejected or resolved deterministically based on
                # the configured merge strategy.
                if key in merged_updates and self.merge_strategy == "error":
                    previous_node = update_sources[key]
                    raise ValueError(
                        f"ParallelNode '{self.name}' received conflicting updates for key '{key}' "
                        f"from nodes '{previous_node}' and '{node_name}'"
                    )
                merged_updates[key] = value
                update_sources[key] = node_name
            outputs[node_name] = result.output
            branch_trace[node_name] = {
                "output": result.output,
                "updates": dict(result.updates),
                "state_after": branch_state.to_dict(),
            }

        if self.output_key is not None:
            merged_updates[self.output_key] = outputs

        final_output = outputs if self.is_final else None
        return WorkflowNodeResult(
            updates=merged_updates,
            output=outputs,
            next_node=self.next_node,
            stop=self.is_final,
            final_output=final_output,
            trace_data={
                "parallel_branches": branch_trace,
                "parallel_merge_strategy": self.merge_strategy,
                "parallel_update_sources": dict(update_sources),
            },
        )
