from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cortex.message import AgentUsage

from .engine import WorkflowEngine
from .state import WorkflowRun, WorkflowState
from .step import Step


@dataclass
class WorkflowAgent:
    """Execute a named sequence or graph of workflow steps against shared state."""

    name: str
    nodes: list[Step]
    start_node: Optional[str] = None
    context: Any = None
    usage: Optional[AgentUsage] = None
    max_steps: int = 50
    _engine: WorkflowEngine = field(init=False, repr=False)

    def __post_init__(self):
        """Validate workflow construction and precompute step lookup structures."""
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
        state = WorkflowState(input=user_input)
        if kwargs:
            state.update(kwargs)
        return state

    def get_step(self, step_name: str):
        """Return a step by name or raise if the step is unknown."""
        return self._engine.get_node(step_name)

    def get_next_step_name(self, current_step_name: str) -> Optional[str]:
        """Return the default ordered successor for a step, if any."""
        return self._engine.get_next_node_name(current_step_name)

    def get_declared_graph(self) -> dict[str, dict[str, Any]]:
        """Return a structured view of the statically declared workflow graph."""
        graph = self._engine.get_declared_graph()
        return {
            node_name: {
                "declared_next_nodes": entry["declared_next_nodes"],
                "default_next_node": entry["default_next_node"],
                "is_terminal": entry["is_terminal"],
                "fallback_node": entry["fallback_node"],
            }
            for node_name, entry in graph.items()
        }

    def describe_graph(self) -> dict[str, Any]:
        """Return a high-level graph description suitable for inspection or tooling."""
        return {
            "name": self.name,
            "start_node": self.start_node,
            "node_order": list(self.node_order),
            "terminal_nodes": [node.name for node in self.nodes if node.is_terminal()],
            "graph": self.get_declared_graph(),
        }

    def _validate_declared_step_references(self) -> None:
        """Validate statically declared step references and obvious dead ends."""
        self._engine._validate_declared_node_references()

    async def async_run(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowState] = None,
        context: Any = None,
    ) -> WorkflowRun:
        """Run the workflow and return the full workflow run record."""
        active_context = context if context is not None else self.context
        workflow_state = state or self.create_state(user_input)
        engine_run = await self._engine.async_run(
            user_input,
            state=workflow_state,
            context=active_context,
            runtime=self,
        )
        return WorkflowRun(
            engine_name=engine_run.engine_name,
            traces=engine_run.traces,
            state=engine_run.state,
            status=engine_run.status,
            final_output=engine_run.final_output,
            error=engine_run.error,
            started_at=engine_run.started_at,
            finished_at=engine_run.finished_at,
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

    @property
    def steps(self) -> list[Step]:
        return self.nodes

    @steps.setter
    def steps(self, value: list[Step]) -> None:
        self.nodes = value

    @property
    def start_step(self) -> Optional[str]:
        return self.start_node

    @start_step.setter
    def start_step(self, value: Optional[str]) -> None:
        self.start_node = value

    @property
    def steps_by_name(self) -> dict[str, Step]:
        return self.nodes_by_name

    @steps_by_name.setter
    def steps_by_name(self, value: dict[str, Step]) -> None:
        self.nodes_by_name = value

    @property
    def step_order(self) -> list[str]:
        return self.node_order

    @step_order.setter
    def step_order(self, value: list[str]) -> None:
        self.node_order = value
