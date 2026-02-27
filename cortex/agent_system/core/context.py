from functools import cached_property
from typing import Optional
from cortex import LLM, AsyncAgentMemoryBank, GPTModels
from cortex.message import AgentUsage
from cortex.backend import ReasoningEffort
from pydantic import BaseModel
from .whiteboard import Whiteboard, WhiteboardStorage


class AgentSystemContext(BaseModel):
    """Context object for the agent system.
    
    Holds shared resources like memory bank, usage tracking, and whiteboard.
    Use the `create()` factory method for easy setup.
    """
    
    # Existing fields
    usage: Optional[AgentUsage] = None  # Usage tracking for the agent
    memory_bank: Optional[object] = None  # Memory bank for the agent
    whiteboard: Optional[Whiteboard] = None  # Optional whiteboard for agent communication

    @classmethod
    def create(
        cls,
        *,
        memory_bank: AsyncAgentMemoryBank,
        enable_whiteboard: bool = False,
        whiteboard_storage: Optional[WhiteboardStorage] = None,
        usage: Optional[AgentUsage] = None,
    ) -> "AgentSystemContext":
        """Factory method to create a context with optional whiteboard.
        
        Args:
            memory_bank: The memory bank for the agent system (required)
            enable_whiteboard: If True, create a whiteboard with the given storage
            whiteboard_storage: Storage backend for the whiteboard. If None and 
                enable_whiteboard is True, uses InMemoryStorage.
            usage: Optional usage tracking object
            
        Returns:
            Configured AgentSystemContext instance
            
        Example:
            ```python
            # Without whiteboard
            context = AgentSystemContext.create(
                memory_bank=AsyncAgentMemoryBank()
            )
            
            # With whiteboard (in-memory)
            context = AgentSystemContext.create(
                memory_bank=AsyncAgentMemoryBank(),
                enable_whiteboard=True
            )
            
            # With whiteboard (Redis persistence)
            context = AgentSystemContext.create(
                memory_bank=AsyncAgentMemoryBank(),
                enable_whiteboard=True,
                whiteboard_storage=RedisStorage(redis_client)
            )
            ```
        """
        wb = None
        if enable_whiteboard:
            wb = Whiteboard(storage=whiteboard_storage)
        
        return cls(
            memory_bank=memory_bank,
            whiteboard=wb,
            usage=usage,
        )

    async def get_memory_bank(self) -> AsyncAgentMemoryBank:
        """Get the agent memory bank for this context."""
        if not self.memory_bank:
            raise ValueError("Memory bank not initialized")
        return self.memory_bank

    
    @cached_property
    def llm_primary(self) -> LLM:
        """Primary, general-purpose reasoning LLM for agents.
        
        Centralized here so agents can reuse a consistent default model
        and temperature. Update this property to roll out model changes app-wide.
        """
        return LLM(model=GPTModels.GPT_5_MINI, reasoning_effort=ReasoningEffort.MINIMAL)

    @cached_property
    def llm_creative(self) -> LLM:
        """Creative/high-variance LLM for tasks benefiting from more exploration.
        
        Uses the same base model as `llm_primary` with a higher temperature.
        """
        return LLM(model=GPTModels.GPT_5_MINI, reasoning_effort=ReasoningEffort.MEDIUM)
