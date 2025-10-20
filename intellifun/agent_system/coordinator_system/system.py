from typing import Any, List, Optional

from intellifun import Agent

from intellifun.agent_system.coordinator_system.coordinator_builder import CoordinatorAgentBuilder
from intellifun.agent_system.coordinator_system.worker_builder import WorkerAgentBuilder

from ..core.system import AgentSystem


class CoordinatorSystem(AgentSystem):
    """Coordinator-focused system built on the core AgentSystem.

    - Accepts a coordinator builder
    - Accepts a list of worker builders
    - Builds and caches the coordinator Agent lazily and wires all worker tools
    - Provides a convenience ask API to take end-user input and optional developer context
    """
    def __init__(
        self,
        coordinator_builder: CoordinatorAgentBuilder,
        workers: Optional[List[WorkerAgentBuilder]] = None,
        context: Optional[Any] = None,
    ):
        super().__init__(context)
        self._coordinator_builder = coordinator_builder
        self._workers = workers or []
        self._coordinator_agent: Optional[Agent] = None

    async def get_agent(self) -> Agent:
        if self._coordinator_agent is not None:
            return self._coordinator_agent

        coordinator_name = self._coordinator_builder.name
        tools = [w.install(coordinator_name=coordinator_name) for w in self._workers]
        
        self._coordinator_agent = await self._coordinator_builder.build_agent(context=self._context, tools=tools)
        return self._coordinator_agent
