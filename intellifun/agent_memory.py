from collections import deque
from dataclasses import dataclass, field

@dataclass
class AgentMemory:
    k: int
    chat_memory: deque = field(default_factory=deque)

    def add_messages(self, msgs):
        self.chat_memory.append(msgs)

        if len(self.chat_memory) > self.k:
            self.chat_memory.popleft()

    def load_memory(self):
        return [m for chat in self.chat_memory for m in chat]
    
    def is_empty(self):
        return len(self.chat_memory) == 0

class AgentMemoryBank:
    '''Memory bank for all agents for a user'''
    def __init__(self):
        self.agent_memories = {}

    def get_agent_memory(self, agent_name, k=5):
        """get memory for a named agent"""
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]

        mem = AgentMemory(k=k)
        self.agent_memories[agent_name] = mem
        return mem


    # global memory bank for each active user
    user_memories = {}

    @classmethod
    def bank_for(cls, user_id):
        '''get user memory bank from global store. if not exist, create one'''
        if user_id in cls.user_memories:
            return cls.user_memories[user_id]

        memory_bank = AgentMemoryBank()
        cls.user_memories[user_id] = memory_bank
        return memory_bank

    @classmethod
    def clear_bank_for(cls, user_id):
        '''clear user memory bank from global store'''
        if user_id in cls.user_memories:
            del cls.user_memories[user_id]

    @classmethod
    def is_active(cls, user_id):
        '''check if user has memory bank in global store'''
        return user_id in cls.user_memories
