from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .node import EngineNode


def _serialize_value(value: Any) -> Any:
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
class NodeResult:
    updates: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    next_node: Optional[str] = None
    stop: bool = False
    final_output: Any = None
    trace_data: dict[str, Any] = field(default_factory=dict)

    def apply(self, state: "EngineState") -> None:
        state.update(self.updates)
        if self.output is not None:
            state.set_output(self.output)
        if self.final_output is not None:
            state.set_final_output(self.final_output)


@dataclass
class EngineTrace:
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
    input: Any = None
    data: dict[str, Any] = field(default_factory=dict)
    last_output: Any = None
    final_output: Any = None
    current_node: Optional[str] = None
    completed_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data

    def require(self, key: str) -> Any:
        if key not in self.data:
            raise KeyError(f"Workflow state is missing required key: {key}")
        return self.data[key]

    def set(self, key: str, value: Any) -> Any:
        self.data[key] = value
        return value

    def update(self, values: Optional[dict[str, Any]]) -> None:
        if values:
            self.data.update(values)

    def set_output(self, value: Any) -> Any:
        self.last_output = value
        return value

    def set_final_output(self, value: Any) -> Any:
        self.final_output = value
        self.last_output = value
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": _serialize_value(self.input),
            "data": _serialize_value(dict(self.data)),
            "last_output": _serialize_value(self.last_output),
            "final_output": _serialize_value(self.final_output),
            "current_node": self.current_node,
            "completed_nodes": list(self.completed_nodes),
            "metadata": _serialize_value(dict(self.metadata)),
        }


@dataclass
class EngineRun:
    engine_name: str
    traces: list[EngineTrace] = field(default_factory=list)
    state: Optional[EngineState] = None
    status: str = "pending"
    final_output: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def add_trace(self, trace: EngineTrace) -> None:
        self.traces.append(trace)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
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
    name: str
    nodes: list[EngineNode]
    start_node: Optional[str] = None
    max_steps: int = 50

    def __post_init__(self):
        if not self.nodes:
            raise ValueError("WorkflowEngine requires at least one node")
        self.nodes_by_name = {node.name: node for node in self.nodes}
        if len(self.nodes_by_name) != len(self.nodes):
            raise ValueError("WorkflowEngine node names must be unique")
        self.node_order = [node.name for node in self.nodes]
        self.start_node = self.start_node or self.nodes[0].name
        if self.start_node not in self.nodes_by_name:
            raise ValueError(f"Unknown start_node: {self.start_node}")
        self._validate_declared_node_references()

    def create_state(self, user_input: Any = None, **kwargs) -> EngineState:
        state = EngineState(input=user_input)
        if kwargs:
            state.update(kwargs)
        return state

    def get_node(self, node_name: str):
        node = self.nodes_by_name.get(node_name)
        if node is None:
            raise KeyError(f"Unknown workflow node: {node_name}")
        return node

    def get_next_node_name(self, current_node_name: str) -> Optional[str]:
        try:
            node_index = self.node_order.index(current_node_name)
        except ValueError as e:
            raise KeyError(f"Unknown workflow node: {current_node_name}") from e
        if node_index + 1 < len(self.node_order):
            return self.node_order[node_index + 1]
        return None

    def get_declared_graph(self) -> dict[str, dict[str, Any]]:
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
        return {
            "name": self.name,
            "start_node": self.start_node,
            "node_order": list(self.node_order),
            "terminal_nodes": [node.name for node in self.nodes if node.is_terminal()],
            "graph": self.get_declared_graph(),
        }

    def _validate_declared_node_references(self) -> None:
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

    async def async_run(self, user_input: Any = None, *, state: Optional[EngineState] = None, context: Any = None, runtime: Any = None) -> EngineRun:
        run = EngineRun(engine_name=self.name, started_at=datetime.now(), status="running")
        state = state or self.create_state(user_input)
        if user_input is not None and state.input is None:
            state.input = user_input
        run.state = state

        current_node_name = self.start_node
        steps_executed = 0

        try:
            while current_node_name is not None:
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
                        attempt_count += 1
                        trace.attempt_count = attempt_count
                        try:
                            if node.policy.timeout_seconds is not None:
                                result = await asyncio.wait_for(
                                    node.run(state, context=context, workflow=runtime),
                                    timeout=node.policy.timeout_seconds,
                                )
                            else:
                                result = await node.run(state, context=context, workflow=runtime)
                            break
                        except Exception as e:
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

                    result.apply(state)
                    state.completed_nodes.append(current_node_name)
                    if result.trace_data:
                        trace.metadata.update(result.trace_data)

                    next_node = result.next_node
                    if result.stop:
                        trace.status = "completed"
                        trace.output = result.output
                        trace.next_node = next_node
                        trace.state_after = state.to_dict()
                        run.final_output = state.final_output if state.final_output is not None else result.output
                        run.status = "completed"
                        break

                    if next_node is None:
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
                run.status = "completed"
                run.final_output = state.final_output if state.final_output is not None else state.last_output

            return run
        finally:
            run.finished_at = datetime.now()
