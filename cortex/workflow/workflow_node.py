"""Core workflow node descriptors and result objects.

Nodes are lightweight, lazy descriptors — just a named function reference with metadata.
Nothing executes until the engine runs the node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional


async def invoke_workflow_callback(
    func,
    *,
    user_input: Any = None,
    context: Any = None,
    memory: Any = None,
    state: Any = None,
):
    """Invoke a workflow callback with flexible signature support."""
    import inspect

    sig = inspect.signature(func)
    params = sig.parameters

    kwargs = {}
    if "user_input" in params:
        kwargs["user_input"] = user_input
    if "context" in params:
        kwargs["context"] = context
    if "memory" in params:
        kwargs["memory"] = memory
    if "state" in params:
        kwargs["state"] = state

    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    return func(**kwargs)


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
    interrupt: bool = False
    interrupt_reason: Optional[str] = None
    interrupt_payload: dict[str, Any] = field(default_factory=dict)
    resume_node: Optional[str] = None

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

    @classmethod
    def interrupt_run(
        cls,
        *,
        reason: str,
        payload: Optional[dict[str, Any]] = None,
        updates: Optional[dict[str, Any]] = None,
        output: Any = None,
        resume_node: Optional[str] = None,
    ) -> "WorkflowNodeResult":
        """Create a result that pauses execution for human-in-the-loop interaction."""

        return cls(
            updates=updates or {},
            output=output,
            interrupt=True,
            interrupt_reason=reason,
            interrupt_payload=payload or {},
            resume_node=resume_node,
        )


@dataclass
class NodeSpec:
    """Lightweight, lazy node descriptor.

    A node is just a named function reference with metadata. The function is only
    called when the engine actually executes this node, making all nodes inherently lazy.
    """

    name: str
    func: Optional[Callable] = None
    kind: Literal["function", "router", "parallel"] = "function"
    policy: NodePolicy = field(default_factory=NodePolicy)
    is_final: bool = False
    output_key: Optional[str] = None
    possible_next_nodes: set[str] = field(default_factory=set)
    branches: dict[str, Callable] = field(default_factory=dict)
    merge_strategy: str = "error"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Node name must be a non-empty string")
        if self.kind == "parallel":
            if not self.branches:
                raise ValueError(f"Parallel node '{self.name}' requires at least one branch")
            if self.merge_strategy not in {"error", "last_write_wins"}:
                raise ValueError("Parallel merge_strategy must be 'error' or 'last_write_wins'")
        elif self.func is None:
            raise ValueError(f"Node '{self.name}' requires a func")

    def declared_next_nodes(self) -> set[str]:
        """Return statically known successors for graph validation."""
        if self.kind == "router":
            return set(self.possible_next_nodes)
        return set()

    def is_terminal(self) -> bool:
        return self.is_final
