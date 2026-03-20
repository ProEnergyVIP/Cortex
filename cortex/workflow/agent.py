"""Public workflow wrapper built on top of `WorkflowEngine`.

This module keeps the high-level workflow API small and ergonomic while delegating the
actual execution logic to the lower-level engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cortex.message import AgentUsage

from .engine import WorkflowEdge, WorkflowEngine, WorkflowStateProtocol
from .node import Node
from .state import WorkflowRun, WorkflowState


@dataclass
class WorkflowAgent:
    """Execute a named sequence or graph of workflow nodes against shared state."""

    name: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[WorkflowEdge] = field(default_factory=list)
    start_node: Optional[str] = None
    context: Any = None
    usage: Optional[AgentUsage] = None
    memory: Any = None
    max_steps: int = 50
    state_type: Optional[type[WorkflowStateProtocol]] = None
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None
    _engine: Optional[WorkflowEngine] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Build the underlying engine and mirror its validated graph metadata."""

        if not self.nodes:
            self._engine = None
            self.nodes_by_name = {}
            self.node_order = []
            return

        self._engine = WorkflowEngine(
            name=self.name,
            nodes=self.nodes,
            edges=self.edges,
            start_node=self.start_node,
            max_steps=self.max_steps,
            state_type=self.state_type,
            state_factory=self.state_factory,
        )
        self.nodes = self._engine.nodes
        self.edges = self._engine.edges
        self.start_node = self._engine.start_node
        self.nodes_by_name = self._engine.nodes_by_name
        self.node_order = self._engine.node_order

    def _rebuild_engine(self) -> None:
        """Rebuild the underlying engine after graph mutations."""

        self.__post_init__()

    def _require_engine(self) -> WorkflowEngine:
        """Return the built engine or raise when the graph is still empty."""

        if self._engine is None:
            raise ValueError("Workflow graph is empty; add at least one node before using it")
        return self._engine

    def add_node(self, node: Node, *, start: bool = False) -> "WorkflowAgent":
        """Add one node to the workflow graph and optionally set it as the start node."""

        self.nodes.append(node)
        if start or self.start_node is None:
            self.start_node = node.name
        self._rebuild_engine()
        return self

    def add_nodes(self, *nodes: Node) -> "WorkflowAgent":
        """Add multiple nodes to the workflow graph."""

        for node in nodes:
            self.nodes.append(node)
        if self.start_node is None and self.nodes:
            self.start_node = self.nodes[0].name
        self._rebuild_engine()
        return self

    def add_edge(self, source: str | Node, target: str | Node) -> "WorkflowAgent":
        """Add one directed edge to the workflow graph."""

        source_name = source.name if hasattr(source, "name") else str(source)
        target_name = target.name if hasattr(target, "name") else str(target)
        self.edges.append(WorkflowEdge(source=source_name, target=target_name))
        self._rebuild_engine()
        return self

    def connect(self, source: str | Node, target: str | Node) -> "WorkflowAgent":
        """Alias for `add_edge(...)` to keep the API short and readable."""

        return self.add_edge(source, target)

    def set_start(self, node: str | Node) -> "WorkflowAgent":
        """Set the workflow start node explicitly."""

        self.start_node = node.name if hasattr(node, "name") else str(node)
        self._rebuild_engine()
        return self

    def create_state(self, user_input: Any = None, **kwargs) -> WorkflowState:
        """Create a new workflow state initialized with user input and extra values."""
        return self._require_engine().create_state(user_input, memory=self.memory, **kwargs)

    def get_node(self, node_name: str):
        """Return a node by name or raise if the node is unknown."""
        return self._require_engine().get_node(node_name)

    def get_next_node_name(self, current_node_name: str) -> Optional[str]:
        """Return the default graph successor for a node, if any."""
        return self._require_engine().get_next_node_name(current_node_name)

    def get_declared_graph(self) -> dict[str, dict[str, Any]]:
        """Return a structured view of the statically declared workflow graph."""
        return self._require_engine().get_declared_graph()

    def describe_graph(self) -> dict[str, Any]:
        """Return a high-level graph description suitable for inspection or tooling."""
        if self._engine is None:
            return {
                "name": self.name,
                "start_node": self.start_node,
                "node_order": [],
                "edges": [],
                "terminal_nodes": [],
                "graph": {},
            }
        return self._engine.describe_graph()

    def _validate_declared_node_references(self) -> None:
        """Re-run graph validation on the underlying engine."""

        self._require_engine()._validate_declared_node_references()

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
        return await self._require_engine().async_run(
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

