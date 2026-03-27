"""Public workflow wrapper built on top of `WorkflowEngine`.

This module keeps the high-level workflow API small and ergonomic while delegating the
actual execution logic to the lower-level engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from .engine import (
    WorkflowEdge,
    WorkflowEngine,
    WorkflowEvent,
    WorkflowHooks,
    WorkflowObservability,
    WorkflowStateProtocol,
)
from .workflow_node import NodeSpec
from .state import WorkflowRun, WorkflowState


@dataclass
class WorkflowAgent:
    """Execute a named sequence or graph of workflow nodes against shared state."""

    name: str
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[WorkflowEdge] = field(default_factory=list)
    start_node: Optional[str] = None
    context: Any = None
    memory: Any = None
    max_steps: int = 50
    state_type: Optional[type[WorkflowStateProtocol]] = None
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None
    observability: WorkflowObservability = field(default_factory=WorkflowObservability)
    hooks: Optional[WorkflowHooks] = None
    _engine: Optional[WorkflowEngine] = field(init=False, repr=False, default=None)
    _graph_dirty: bool = field(init=False, repr=False, default=False)

    def __post_init__(self):
        """Build the underlying engine and mirror its validated graph metadata."""

        if not self.nodes:
            self._engine = None
            self._graph_dirty = False
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
            observability=self.observability,
            hooks=self.hooks,
        )
        self.nodes = self._engine.nodes
        self.edges = self._engine.edges
        self.start_node = self._engine.start_node
        self.nodes_by_name = self._engine.nodes_by_name
        self.node_order = self._engine.node_order
        self._graph_dirty = False

    def _rebuild_engine(self) -> None:
        """Rebuild the underlying engine after graph mutations."""

        self.__post_init__()

    def _mark_graph_dirty(self) -> None:
        """Mark graph metadata as stale until ``build()`` is called."""

        self._graph_dirty = True
        self._engine = None
        self.nodes_by_name = {node.name: node for node in self.nodes}
        self.node_order = [node.name for node in self.nodes]

    def build(self) -> "WorkflowAgent":
        """Finalize graph construction and build the underlying engine.

        Call this after finishing ``add_node(...)`` / ``add_edge(...)`` mutations.
        """

        self._rebuild_engine()
        return self

    def finalize(self) -> "WorkflowAgent":
        """Alias for ``build()``."""

        return self.build()

    def _require_engine(self) -> WorkflowEngine:
        """Return the built engine or raise when the graph is still empty."""

        if self._engine is None:
            if not self.nodes:
                raise ValueError("Workflow graph is empty; add at least one node before using it")
            raise ValueError("Workflow graph is not built; call workflow.build() after finishing graph mutations")
        if self._graph_dirty:
            raise ValueError("Workflow graph is not built; call workflow.build() after finishing graph mutations")
        return self._engine

    def add_node(self, name_or_spec, func=None, *, start: bool = False, **kwargs) -> "WorkflowAgent":
        """Add a node to the workflow graph.

        Accepts any of:
          - ``add_node(spec)``              — a pre-built ``NodeSpec``
          - ``add_node("name", func)``      — inline name + function
          - ``add_node("name", func, kind="router", ...)``
        """

        if isinstance(name_or_spec, NodeSpec):
            spec = name_or_spec
        elif func is not None:
            spec = NodeSpec(name=str(name_or_spec), func=func, **kwargs)
        elif callable(name_or_spec):
            spec = NodeSpec(name=name_or_spec.__name__, func=name_or_spec, **kwargs)
        else:
            raise TypeError("add_node requires a NodeSpec, (name, func), or a callable")
        self.nodes.append(spec)
        if start or self.start_node is None:
            self.start_node = spec.name
        self._mark_graph_dirty()
        return self

    def add_nodes(self, *specs: NodeSpec) -> "WorkflowAgent":
        """Add multiple pre-built NodeSpec instances."""

        for spec in specs:
            self.nodes.append(spec)
        if self.start_node is None and self.nodes:
            self.start_node = self.nodes[0].name
        self._mark_graph_dirty()
        return self

    def node(self, name: Optional[str] = None, **kwargs):
        """Decorator that registers a function as a workflow node.

        Usage::

            @wf.node
            async def step_a(data, context):
                return {"result": "done"}

            @wf.node("custom_step")
            async def step_b(data, context):
                return {"result": "done"}
        """

        def decorator(inner_func):
            node_name = name or inner_func.__name__
            self.add_node(NodeSpec(name=node_name, func=inner_func, **kwargs))
            return inner_func

        if callable(name):
            func = name
            self.add_node(NodeSpec(name=func.__name__, func=func, **kwargs))
            return func

        if name is not None and not isinstance(name, str):
            raise TypeError("@wf.node name must be a string when provided")

        if name is None:
            return decorator
        return decorator

    def add_edge(self, source: str | NodeSpec, target: str | NodeSpec) -> "WorkflowAgent":
        """Add one directed edge to the workflow graph."""

        source_name = source.name if hasattr(source, "name") else str(source)
        target_name = target.name if hasattr(target, "name") else str(target)
        self.edges.append(WorkflowEdge(source=source_name, target=target_name))
        self._mark_graph_dirty()
        return self

    def connect(self, source: str | NodeSpec, target: str | NodeSpec) -> "WorkflowAgent":
        """Alias for ``add_edge``."""

        return self.add_edge(source, target)

    def set_start(self, node: str | NodeSpec) -> "WorkflowAgent":
        """Set the workflow start node explicitly."""

        self.start_node = node.name if hasattr(node, "name") else str(node)
        self._mark_graph_dirty()
        return self

    def create_state(self, user_input: Any = None, **kwargs) -> WorkflowState:
        """Create a new workflow state initialized with user input and extra values."""
        return self._require_engine().create_state(user_input, **kwargs)

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
            edges = [{"source": edge.source, "target": edge.target} for edge in self.edges]
            graph: dict[str, dict[str, Any]] = {
                node.name: {
                    "kind": node.kind,
                    "is_terminal": node.is_terminal(),
                    "declared_next_nodes": sorted(node.declared_next_nodes()),
                    "edge_next_nodes": [],
                }
                for node in self.nodes
            }
            for edge in self.edges:
                if edge.source in graph:
                    graph[edge.source]["edge_next_nodes"].append(edge.target)
            return {
                "name": self.name,
                "start_node": self.start_node,
                "node_order": [node.name for node in self.nodes],
                "edges": edges,
                "terminal_nodes": [node.name for node in self.nodes if node.is_terminal()],
                "graph": graph,
                "is_built": False,
            }
        description = self._engine.describe_graph()
        description["is_built"] = True
        return description

    def _validate_declared_node_references(self) -> None:
        """Re-run graph validation on the underlying engine."""

        self._require_engine()._validate_declared_node_references()

    async def async_run(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
        hooks: Optional[WorkflowHooks] = None,
        observability: Optional[WorkflowObservability] = None,
        tags: Optional[dict[str, str]] = None,
        parent_run_id: Optional[str] = None,
        start_node: Optional[str] = None,
    ) -> WorkflowRun:
        """Run the workflow and return the full workflow run record."""

        # Per-call context can override the agent default without mutating the agent.
        active_context = context if context is not None else self.context
        workflow_state = state or self.create_state(user_input)
        return await self._require_engine().async_run(
            user_input,
            state=workflow_state,
            context=active_context,
            memory=self.memory,
            hooks=hooks,
            observability=observability,
            tags=tags,
            parent_run_id=parent_run_id,
            start_node=start_node,
        )

    async def async_ask(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
        hooks: Optional[WorkflowHooks] = None,
        observability: Optional[WorkflowObservability] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Any:
        """Run the workflow and return only the final output."""

        run = await self.async_run(
            user_input,
            state=state,
            context=context,
            hooks=hooks,
            observability=observability,
            tags=tags,
        )
        return run.final_output

    async def async_resume(
        self,
        run: WorkflowRun,
        *,
        user_input: Any = None,
        context: Any = None,
        hooks: Optional[WorkflowHooks] = None,
        observability: Optional[WorkflowObservability] = None,
    ) -> WorkflowRun:
        """Resume a paused workflow run."""

        active_context = context if context is not None else self.context
        return await self._require_engine().async_resume(
            run,
            user_input=user_input,
            context=active_context,
            memory=self.memory,
            hooks=hooks,
            observability=observability,
        )

    async def async_run_events(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
        hooks: Optional[WorkflowHooks] = None,
        observability: Optional[WorkflowObservability] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[WorkflowEvent]:
        """Run a workflow and stream structured lifecycle events."""

        active_context = context if context is not None else self.context
        workflow_state = state or self.create_state(user_input)
        async for event in self._require_engine().async_run_events(
            user_input,
            state=workflow_state,
            context=active_context,
            memory=self.memory,
            hooks=hooks,
            observability=observability,
            tags=tags,
        ):
            yield event

