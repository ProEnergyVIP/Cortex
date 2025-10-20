from typing import Any, List

from cortex import Agent
from cortex.message import Message


class AgentSystem:
    """Minimal base system that forwards messages to an underlying Agent.

    Subclasses should implement get_agent(context) to provide the Agent instance
    appropriate for the given runtime context (e.g., to resolve LLMs, tools, memory).
    """
    def __init__(self, context):
        self._context = context
    
    async def get_agent(self) -> Agent:
        raise NotImplementedError

    async def async_ask(
        self,
        messages: str | Message | List[Message],
    ) -> Any:
        agent = await self.get_agent()
        usage = getattr(self._context, "usage", None)
        return await agent.async_ask(messages, usage=usage)
