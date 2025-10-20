from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, ClassVar

from cortex.message import Message

@dataclass
class AgentMemory:
    """Base class for agent memory. Default implementation uses in-memory storage."""
    k: int
    chat_memory: deque = field(default_factory=deque)

    def add_messages(self, msgs: List[Message]) -> None:
        """Add messages to the memory."""
        self.chat_memory.append(msgs)

        if len(self.chat_memory) > self.k:
            self.chat_memory.popleft()

    def load_memory(self) -> List[Message]:
        """Load all messages from memory."""
        return [m for chat in self.chat_memory for m in chat]
    
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.chat_memory) == 0


@dataclass
class AsyncAgentMemory:
    """Asynchronous version of agent memory. Default implementation uses in-memory storage."""
    k: int
    chat_memory: deque = field(default_factory=deque)
    
    async def add_messages(self, msgs: List[Message]) -> None:
        """Add messages to the memory asynchronously."""
        self.chat_memory.append(msgs)

        if len(self.chat_memory) > self.k:
            self.chat_memory.popleft()
            
    async def load_memory(self) -> List[Message]:
        """Load all messages from memory asynchronously."""
        return [m for chat in self.chat_memory for m in chat]
    
    async def is_empty(self) -> bool:
        """Check if memory is empty asynchronously."""
        return len(self.chat_memory) == 0

class AgentMemoryBank:
    """Memory bank for all agents for a user. Default implementation uses in-memory storage."""
    # Static mapping of user IDs to memory banks
    user_memories: ClassVar[Dict[str, 'AgentMemoryBank']] = {}
    
    def __init__(self):
        """Initialize an agent memory bank."""
        self.agent_memories: Dict[str, AgentMemory] = {}

    def get_agent_memory(self, agent_name: str, k: int = 5) -> AgentMemory:
        """Get memory for a named agent."""
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]

        mem = AgentMemory(k=k)
        self.agent_memories[agent_name] = mem
        return mem
    
    def reset_memory(self):
        '''Reset all agent memories in this bank.'''
        self.agent_memories = {}

    @classmethod
    def bank_for(cls, user_id: str, **kwargs) -> 'AgentMemoryBank':
        """Get user memory bank from global store. If it doesn't exist, create one."""
        if user_id in cls.user_memories:
            return cls.user_memories[user_id]

        memory_bank = AgentMemoryBank()
        cls.user_memories[user_id] = memory_bank
        return memory_bank

    @classmethod
    def clear_bank_for(cls, user_id: str) -> None:
        """Clear user memory bank from global store."""
        if user_id in cls.user_memories:
            del cls.user_memories[user_id]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all memory banks for all users."""
        cls.user_memories.clear()

    @classmethod
    def is_active(cls, user_id: str) -> bool:
        """Check if user has memory bank in global store."""
        return user_id in cls.user_memories


class AsyncAgentMemoryBank:
    """Asynchronous memory bank for all agents for a user. Default implementation uses in-memory storage."""
    # Static mapping of user IDs to memory banks
    user_memories: ClassVar[Dict[str, 'AsyncAgentMemoryBank']] = {}
    
    def __init__(self):
        """Initialize an async agent memory bank."""
        self.agent_memories: Dict[str, AsyncAgentMemory] = {}

    async def get_agent_memory(self, agent_name: str, k: int = 5) -> AsyncAgentMemory:
        """Get memory for a named agent asynchronously."""
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]

        mem = AsyncAgentMemory(k=k)
        self.agent_memories[agent_name] = mem
        return mem
    
    async def reset_memory(self):
        '''Reset all agent memories in this bank.'''
        self.agent_memories = {}

    @classmethod
    async def bank_for(cls, user_id: str, **kwargs) -> 'AsyncAgentMemoryBank':
        """Get user memory bank from global store asynchronously. If it doesn't exist, create one."""
        if user_id in cls.user_memories:
            return cls.user_memories[user_id]

        memory_bank = AsyncAgentMemoryBank()
        cls.user_memories[user_id] = memory_bank
        return memory_bank

    @classmethod
    async def clear_bank_for(cls, user_id: str) -> None:
        """Clear user memory bank from global store asynchronously."""
        if user_id in cls.user_memories:
            del cls.user_memories[user_id]

    @classmethod
    async def reset_all(cls) -> None:
        """Reset all memory banks for all users asynchronously."""
        cls.user_memories.clear()

    @classmethod
    async def is_active(cls, user_id: str) -> bool:
        """Check if user has memory bank in global store asynchronously."""
        return user_id in cls.user_memories
