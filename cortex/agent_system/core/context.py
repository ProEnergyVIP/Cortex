from functools import cached_property
from typing import Optional
from cortex import LLM, AsyncAgentMemoryBank, GPTModels
from cortex.message import AgentUsage
from cortex.backend import ReasoningEffort
from pydantic import BaseModel
from .whiteboard import Whiteboard


class AgentSystemContext(BaseModel):
    # Existing fields
    usage: Optional[AgentUsage] = None  # Usage tracking for the agent
    memory_bank: Optional[object] = None  # Memory bank for the agent
    whiteboard: Optional[Whiteboard] = None

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
