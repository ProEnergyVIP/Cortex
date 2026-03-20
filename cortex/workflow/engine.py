from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from inspect import isawaitable
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from cortex.message import AIMessage, Message, UserMessage

from .node import EngineNode


def _serialize_value(value: Any) -> Any:
    """Recursively convert workflow runtime objects into plain Python data."""

    if isinstance(value, EngineRun):
        return value.to_dict()
    if isinstance(value, EngineState):
        return value.to_dict()
    if isinstance(value, EngineTrace):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass
class EngineTrace:
    """Trace record for one node execution, including retries and routing decisions."""

    node_name: str
    status: str = "pending"
    attempt_count: int = 0
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    next_node: Optional[str] = None
    fallback_node: Optional[str] = None
    state_before: Optional[dict[str, Any]] = None
    state_after: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the trace."""

        return {
            "node_name": self.node_name,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "duration_ms": self.duration_ms,
            "next_node": self.next_node,
            "fallback_node": self.fallback_node,
            "state_before": _serialize_value(self.state_before),
            "state_after": _serialize_value(self.state_after),
            "metadata": _serialize_value(self.metadata),
            "output": _serialize_value(self.output),
            "error": self.error,
        }


@dataclass
class EngineState:
    """Minimal workflow state with only essential engine fields."""

    # User-mutable data bag (the only field user code should modify)
    data: dict[str, Any] = field(default_factory=dict)

    # Essential engine fields (read-only for user code)
    context: Any = None
    usage: Any = None
    memory: Any = None
    current_node: Optional[str] = None
    completed_nodes: list[str] = field(default_factory=list)

    # Derived fields that are stored in state.data with reserved keys
    @property
    def input(self) -> Any:
        """Original workflow input from state.data."""
        return self.data.get("input")

    @property
    def last_output(self) -> Any:
        """Last node output from state.data."""
        return self.data.get("_last_output")

    @property
    def final_output(self) -> Any:
        """Terminal workflow output from state.data."""
        return self.data.get("_final_output")

    def set_input(self, value: Any) -> None:
        """Set the original workflow input in state.data."""
        self.data["input"] = value

    def set_last_output(self, value: Any) -> None:
        """Set the last node output in state.data."""
        self.data["_last_output"] = value

    def set_final_output(self, value: Any) -> None:
        """Set the terminal workflow output in state.data."""
        self.data["_final_output"] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from the workflow data bag."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in the workflow data bag."""
        return key in self.data

    def update(self, updates: dict[str, Any]) -> None:
        """Merge updates into the workflow data bag with reserved key handling."""
        for key, value in updates.items():
            if key == "input":
                # Never overwrite original input
                continue
            elif key == "_final_output":
                # Only allow final nodes to set this
                self.data[key] = value
            else:
                # Normal merge for all other keys including "_last_output"
                self.data[key] = value

    def record_output(self, value: Any) -> Any:
        """Record the terminal workflow output and mirror it to last_output."""
        self.set_final_output(value)
        self.set_last_output(value)
        return value

    def set_output(self, value: Any) -> Any:
        """Set the last node output in state.data."""
        self.set_last_output(value)
        return value

    def set(self, key: str, value: Any) -> Any:
        """Store a value in the workflow data bag and return it."""

        self.data[key] = value
        return value


    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the current state."""

        return {
            "data": _serialize_value(dict(self.data)),
            "context": _serialize_value(self.context),
            "usage": _serialize_value(self.usage),
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
        }

    def clone(self, **overrides) -> "EngineState":
        """Return a shallow structural copy of the state with optional field overrides."""

        values = {
            "data": dict(self.data),
            "context": self.context,
            "usage": self.usage,
            "memory": self.memory,
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
        }
        values.update(overrides)
        return type(self)(**values)


@runtime_checkable
class WorkflowStateProtocol(Protocol):
    """Structural contract for workflow state objects used by the engine and nodes."""

    data: dict[str, Any]
    context: Any
    usage: Any
    memory: Any
    current_node: Optional[str]
    completed_nodes: list[str]

    def get(self, key: str, default: Any = None) -> Any: ...
    def has(self, key: str) -> bool: ...
    def update(self, updates: dict[str, Any]) -> None: ...
    def set_output(self, value: Any) -> Any: ...
    def set_final_output(self, value: Any) -> Any: ...

    def to_dict(self) -> dict[str, Any]:
        ...

    def clone(self, **overrides) -> "WorkflowStateProtocol":
        ...


@dataclass
class EngineRun:
    """Top-level record for a workflow execution."""

    engine_name: str
    traces: list[EngineTrace] = field(default_factory=list)
    state: Optional[EngineState] = None
    status: str = "pending"
    final_output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def add_trace(self, trace: EngineTrace) -> None:
        """Append a node trace to the run record."""

        self.traces.append(trace)

    @property
    def duration_ms(self) -> Optional[float]:
        """Return total runtime in milliseconds when timing information is complete."""

        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the run."""

        return {
            "engine_name": self.engine_name,
            "traces": [trace.to_dict() for trace in self.traces],
            "state": self.state.to_dict() if self.state is not None else None,
            "status": self.status,
            "final_output": _serialize_value(self.final_output),
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at is not None else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class WorkflowEngine:
    """Execute a validated graph of workflow nodes against shared state."""

    name: str
    nodes: list[EngineNode]
    start_node: Optional[str] = None
    max_steps: int = 50
    state_type: Optional[type[WorkflowStateProtocol]] = None
    state_factory: Optional[Callable[..., WorkflowStateProtocol]] = None

    def __post_init__(self):
        """Validate workflow structure and precompute fast lookup tables."""

        if not self.nodes:
            raise ValueError("WorkflowEngine requires at least one node")
        if self.state_type is not None and self.state_factory is not None:
            raise ValueError("WorkflowEngine accepts either state_type or state_factory, not both")
        self.nodes_by_name = {node.name: node for node in self.nodes}
        if len(self.nodes_by_name) != len(self.nodes):
            raise ValueError("WorkflowEngine node names must be unique")
        self.node_order = [node.name for node in self.nodes]
        self.start_node = self.start_node or self.nodes[0].name
        if self.start_node not in self.nodes_by_name:
            raise ValueError(f"Unknown start_node: {self.start_node}")
        self._validate_declared_node_references()

    def create_state(self, user_input: Any = None, **kwargs) -> WorkflowStateProtocol:
        """Create a fresh workflow state seeded with user input and extra values."""

        state_attribute_names = {
            "context",
            "usage",
            "memory",
            "current_node",
            "completed_nodes",
            "data",
        }
        state_overrides = {key: value for key, value in kwargs.items() if key in state_attribute_names}
        initial_data = {key: value for key, value in kwargs.items() if key not in state_attribute_names}

        if self.state_factory is not None:
            state = self.state_factory(user_input=user_input, initial_data=initial_data)
        else:
            state_cls = self.state_type or EngineState
            state = state_cls()
        if user_input is not None and not state.get("input"):
            state.set_input(user_input)
        if not isinstance(state, WorkflowStateProtocol):
            raise TypeError("Workflow state must satisfy WorkflowStateProtocol")
        if state_overrides:
            for key, value in state_overrides.items():
                if getattr(state, key) != value:
                    setattr(state, key, value)
        if initial_data:
            data_updates = {}
            for key, value in initial_data.items():
                if state.get(key) != value:
                    data_updates[key] = value
            if user_input is not None and state.get("input") is None:
                data_updates["input"] = user_input
            if data_updates:
                state.update(data_updates)
        elif user_input is not None and state.get("input") is None:
            state.update({"input": user_input})
        return state

    def get_node(self, node_name: str):
        """Return a node by name or raise when the workflow graph is invalid at runtime."""

        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise KeyError(f"Unknown workflow node: {node_name}")
        return node

    def get_next_node_name(self, current_node_name: str) -> Optional[str]:
        """Return the default ordered successor for a node."""

        try:
            node_index = self.node_order.index(current_node_name)
        except ValueError as e:
            raise KeyError(f"Unknown workflow node: {current_node_name}") from e
        if node_index + 1 < len(self.node_order):
            return self.node_order[node_index + 1]
        return None

    def get_declared_graph(self) -> dict[str, dict[str, Any]]:
        """Describe the statically declared graph shape for inspection or tooling."""

        graph: dict[str, dict[str, Any]] = {}
        for index, node in enumerate(self.nodes):
            graph[node.name] = {
                "declared_next_nodes": sorted(node.declared_next_nodes()),
                "default_next_node": self.node_order[index + 1] if index + 1 < len(self.node_order) else None,
                "is_terminal": node.is_terminal(),
                "fallback_node": node.policy.fallback_node,
            }
        return graph

    def describe_graph(self) -> dict[str, Any]:
        """Return a higher-level graph summary for debugging and visualization."""

        return {
            "name": self.name,
            "start_node": self.start_node,
            "node_order": list(self.node_order),
            "terminal_nodes": [node.name for node in self.nodes if node.is_terminal()],
            "graph": self.get_declared_graph(),
        }

    def _validate_declared_node_references(self) -> None:
        """Catch invalid edges and obvious dead ends before execution starts."""

        known_nodes = set(self.node_order)
        invalid_references: list[tuple[str, str]] = []
        dead_end_nodes: list[str] = []

        for index, node in enumerate(self.nodes):
            for next_node in node.declared_next_nodes():
                if next_node not in known_nodes:
                    invalid_references.append((node.name, next_node))
            if node.policy.fallback_node is not None and node.policy.fallback_node not in known_nodes:
                invalid_references.append((f"{node.name} [fallback]", node.policy.fallback_node))
            has_declared_successor = bool(node.declared_next_nodes())
            has_order_successor = index + 1 < len(self.nodes)
            if not node.is_terminal() and not has_declared_successor and not has_order_successor:
                dead_end_nodes.append(node.name)

        if invalid_references:
            formatted = ", ".join(f"{node_name} -> {next_node}" for node_name, next_node in invalid_references)
            raise ValueError(f"Workflow contains invalid next_node references: {formatted}")
        if dead_end_nodes:
            formatted = ", ".join(dead_end_nodes)
            raise ValueError(f"Workflow contains non-terminal dead-end nodes: {formatted}")

    async def _persist_conversation(self, memory: Any, messages: list[Message]) -> None:
        """Persist one conversation round to either sync or async memory objects."""

        if memory is None or not hasattr(memory, "add_messages"):
            return
        added = memory.add_messages(messages)
        if isawaitable(added):
            await added

    def _normalize_conversation_input(self, user_input: Any) -> list[Message]:
        """Convert workflow input into memory-ready conversation messages."""

        if user_input is None:
            return []
        if isinstance(user_input, list) and all(isinstance(item, Message) for item in user_input):
            return list(user_input)
        if isinstance(user_input, Message):
            return [user_input]
        return [UserMessage(content=str(user_input))]

    def _normalize_conversation_output(self, output: Any) -> list[Message]:
        """Convert final workflow output into memory-ready assistant messages."""

        if output is None:
            return []
        if isinstance(output, list) and all(isinstance(item, Message) for item in output):
            return list(output)
        if isinstance(output, Message):
            return [output]
        return [AIMessage(content=str(output))]

    def _attach_runtime_resources(
        self,
        state: WorkflowStateProtocol,
        *,
        context: Any,
        usage: Any,
        memory: Any,
    ) -> None:
        """Expose shared workflow runtime resources on state for node access."""

        state.context = context
        state.usage = usage
        state.memory = memory

    async def async_run(
        self,
        user_input: Any = None,
        *,
        state: Optional[WorkflowStateProtocol] = None,
        context: Any = None,
        runtime: Any = None,
    ) -> EngineRun:
        """Run the workflow from `start_node` until a node stops or the graph ends."""

        run = EngineRun(engine_name=self.name, started_at=datetime.now(), status="running")
        state = state or self.create_state(user_input)
        if user_input is not None and not state.data.get("input"):
            state.set_input(user_input)
        if user_input is not None and not state.get("input"):
            state.update({"input": user_input})
        runtime_context = context if context is not None else getattr(runtime, "context", None)
        runtime_usage = getattr(runtime, "usage", None)
        memory = getattr(runtime, "memory", None)
        self._attach_runtime_resources(
            state,
            context=runtime_context,
            usage=runtime_usage,
            memory=memory,
        )
        run.state = state

        current_node_name = self.start_node
        steps_executed = 0

        try:
            while current_node_name is not None:
                # Guard against accidental cycles or routing bugs.
                steps_executed += 1
                if steps_executed > self.max_steps:
                    raise RuntimeError(f"Workflow exceeded max_steps={self.max_steps}")

                node = self.get_node(current_node_name)
                state.current_node = current_node_name
                trace = EngineTrace(
                    node_name=current_node_name,
                    status="running",
                    started_at=datetime.now(),
                    state_before=state.to_dict(),
                )
                run.add_trace(trace)

                try:
                    attempt_count = 0
                    while True:
                        # Retries are tracked on the same trace so a single node execution
                        # remains easy to inspect.
                        attempt_count += 1
                        trace.attempt_count = attempt_count
                        node_input = state.data if state.data else state.input
                        try:
                            if node.policy.timeout_seconds is not None:
                                result = await asyncio.wait_for(
                                    node.run(node_input, context=state.context, state=state, workflow=runtime),
                                    timeout=node.policy.timeout_seconds,
                                )
                            else:
                                result = await node.run(node_input, context=state.context, state=state, workflow=runtime)
                            break
                        except Exception as e:
                            # Nodes may attach structured trace metadata to exceptions.
                            if hasattr(e, "trace_data") and e.trace_data:
                                trace.metadata.update(e.trace_data)
                            if isinstance(e, TimeoutError):
                                trace.metadata["timeout_seconds"] = node.policy.timeout_seconds
                            if attempt_count <= node.policy.max_retries:
                                continue
                            if node.policy.failure_strategy == "fallback" and node.policy.fallback_node is not None:
                                trace.status = "fallback"
                                trace.error = str(e)
                                trace.fallback_node = node.policy.fallback_node
                                current_node_name = node.policy.fallback_node
                                result = None
                                break
                            trace.status = "failed"
                            trace.error = str(e)
                            run.status = "failed"
                            run.error = str(e)
                            raise

                    if trace.status == "fallback":
                        trace.state_after = state.to_dict()
                        continue

                    # Node results are applied centrally so every node type follows the same
                    # state transition rules.
                    result.apply(state)
                    state.completed_nodes.append(current_node_name)
                    if result.trace_data:
                        trace.metadata.update(result.trace_data)

                    next_node = result.next_node
                    if result.stop:
                        # A terminal result can still expose a next_node for debugging, but
                        # execution stops immediately once `stop=True`.
                        trace.status = "completed"
                        trace.output = result.output
                        trace.next_node = next_node
                        trace.state_after = state.to_dict()
                        run.final_output = state.final_output if state.final_output is not None else result.output
                        run.status = "completed"
                        break

                    if next_node is None:
                        # When a node does not explicitly route, fall back to declared order.
                        next_node = self.get_next_node_name(current_node_name)

                    trace.status = "completed"
                    trace.output = result.output
                    trace.next_node = next_node
                    trace.state_after = state.to_dict()
                    current_node_name = next_node
                finally:
                    trace.finished_at = datetime.now()
                    if trace.started_at is not None:
                        trace.duration_ms = (trace.finished_at - trace.started_at).total_seconds() * 1000

            if run.status == "running":
                # Reaching the end of the graph without an explicit stop still counts as a
                # successful completion.
                run.status = "completed"
                run.final_output = state.final_output if state.final_output is not None else state.last_output

            if run.status == "completed":
                await self._persist_conversation(
                    memory,
                    [
                        *self._normalize_conversation_input(state.input),
                        *self._normalize_conversation_output(run.final_output),
                    ],
                )

            return run
        finally:
            run.finished_at = datetime.now()
