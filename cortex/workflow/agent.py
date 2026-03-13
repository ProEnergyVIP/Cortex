"""Public workflow wrapper built on top of `WorkflowEngine`.

This module keeps the high-level workflow API small and ergonomic while delegating the
actual execution logic to the lower-level engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cortex.message import AgentUsage

from .engine import WorkflowEngine
from .node import Node
from .state import WorkflowRun, WorkflowState


@dataclass
class WorkflowAgent:
    """Execute a named sequence or graph of workflow nodes against shared state."""

    name: str
    nodes: list[Node]
    start_node: Optional[str] = None
    context: Any = None
    usage: Optional[AgentUsage] = None
    max_steps: int = 50
    _engine: WorkflowEngine = field(init=False, repr=False)

    def __post_init__(self):
        """Build the underlying engine and mirror its validated graph metadata."""

        self._engine = WorkflowEngine(
            name=self.name,
            nodes=self.nodes,
            start_node=self.start_node,
            max_steps=self.max_steps,
        )
        self.nodes = self._engine.nodes
        self.start_node = self._engine.start_node
        self.nodes_by_name = self._engine.nodes_by_name
        self.node_order = self._engine.node_order

    def create_state(self, user_input: Any = None, **kwargs) -> WorkflowState:
        """Create a new workflow state initialized with user input and extra values."""
        return self._engine.create_state(user_input, **kwargs)

    def get_node(self, node_name: str):
        """Return a node by name or raise if the node is unknown."""
        return self._engine.get_node(node_name)

    def get_next_node_name(self, current_node_name: str) -> Optional[str]:
        """Return the default ordered successor for a node, if any."""
        return self._engine.get_next_node_name(current_node_name)

    def get_declared_graph(self) -> dict[str, dict[str, Any]]:
        """Return a structured view of the statically declared workflow graph."""
        return self._engine.get_declared_graph()

    def describe_graph(self) -> dict[str, Any]:
        """Return a high-level graph description suitable for inspection or tooling."""
        return self._engine.describe_graph()

    def _validate_declared_node_references(self) -> None:
        """Re-run graph validation on the underlying engine."""

        self._engine._validate_declared_node_references()

    async def async_run(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
    ) -> WorkflowRun:
        """Run the workflow and return the full workflow run record."""

        # Per-call context can override the agent default without mutating the agent.
        active_context = context if context is not None else self.context
        workflow_state = state or self.create_state(user_input)
        return await self._engine.async_run(
            user_input,
            state=workflow_state,
            context=active_context,
            runtime=self,
        )

    async def async_ask(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
    ) -> Any:
        """Run the workflow and return only the final output."""

        run = await self.async_run(user_input, state=state, context=context)
        return run.final_output

