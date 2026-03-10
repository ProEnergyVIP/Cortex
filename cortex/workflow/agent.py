from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from cortex.message import AgentUsage

from .state import StepTrace, WorkflowRun, WorkflowState


@dataclass
class WorkflowAgent:
    name: str
    steps: list
    start_step: Optional[str] = None
    context: Any = None
    usage: Optional[AgentUsage] = None
    max_steps: int = 50

    def __post_init__(self):
        if not self.steps:
            raise ValueError("WorkflowAgent requires at least one step")
        self.steps_by_name = {step.name: step for step in self.steps}
        if len(self.steps_by_name) != len(self.steps):
            raise ValueError("WorkflowAgent step names must be unique")
        self.step_order = [step.name for step in self.steps]
        self.start_step = self.start_step or self.steps[0].name
        if self.start_step not in self.steps_by_name:
            raise ValueError(f"Unknown start_step: {self.start_step}")
        self._validate_declared_step_references()

    def create_state(self, user_input: Any = None, **kwargs) -> WorkflowState:
        state = WorkflowState(input=user_input)
        if kwargs:
            state.update(kwargs)
        return state

    def get_step(self, step_name: str):
        step = self.steps_by_name.get(step_name)
        if step is None:
            raise KeyError(f"Unknown workflow step: {step_name}")
        return step

    def get_next_step_name(self, current_step_name: str) -> Optional[str]:
        try:
            step_index = self.step_order.index(current_step_name)
        except ValueError as e:
            raise KeyError(f"Unknown workflow step: {current_step_name}") from e
        if step_index + 1 < len(self.step_order):
            return self.step_order[step_index + 1]
        return None

    def _validate_declared_step_references(self) -> None:
        known_steps = set(self.step_order)
        invalid_references: list[tuple[str, str]] = []

        for step in self.steps:
            for next_step in step.declared_next_steps():
                if next_step not in known_steps:
                    invalid_references.append((step.name, next_step))

        if invalid_references:
            formatted = ", ".join(
                f"{step_name} -> {next_step}" for step_name, next_step in invalid_references
            )
            raise ValueError(f"Workflow contains invalid next_step references: {formatted}")

    async def async_run(self, user_input: Any = None, *, state: Optional[WorkflowState] = None, context: Any = None) -> WorkflowRun:
        run = WorkflowRun(workflow_name=self.name, started_at=datetime.now(), status="running")
        state = state or self.create_state(user_input)
        if user_input is not None and state.input is None:
            state.input = user_input
        run.state = state

        current_step_name = self.start_step
        steps_executed = 0
        active_context = context if context is not None else self.context

        try:
            while current_step_name is not None:
                steps_executed += 1
                if steps_executed > self.max_steps:
                    raise RuntimeError(f"Workflow exceeded max_steps={self.max_steps}")

                step = self.get_step(current_step_name)

                state.current_step = current_step_name
                trace = StepTrace(step_name=current_step_name, status="running", started_at=datetime.now())
                run.add_trace(trace)

                try:
                    result = await step.run(state, context=active_context, workflow=self)
                    result.apply(state)
                    state.completed_steps.append(current_step_name)

                    next_step = result.next_step
                    if result.stop:
                        trace.status = "completed"
                        trace.output = result.output
                        trace.next_step = next_step
                        run.final_output = state.final_output if state.final_output is not None else result.output
                        run.status = "completed"
                        break

                    if next_step is None:
                        next_step = self.get_next_step_name(current_step_name)

                    trace.status = "completed"
                    trace.output = result.output
                    trace.next_step = next_step
                    current_step_name = next_step
                except Exception as e:
                    trace.status = "failed"
                    trace.error = str(e)
                    run.status = "failed"
                    run.error = str(e)
                    raise
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

    async def async_ask(self, user_input: Any = None, *, state: Optional[WorkflowState] = None, context: Any = None) -> Any:
        run = await self.async_run(user_input, state=state, context=context)
        return run.final_output
