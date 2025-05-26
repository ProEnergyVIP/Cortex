"""
Memcached-based implementation of agent memory system with separate sync and async classes.
"""

import json
from typing import List, Dict, Any, ClassVar, Dict as DictType

from intellifun.agent_memory import AgentMemory, AsyncAgentMemory, AgentMemoryBank, AsyncAgentMemoryBank


class MemcachedAgentMemory(AgentMemory):
    """
    Synchronous memcached-based implementation of agent memory.
    Stores messages in memcached.
    """
    
    def __init__(self, k: int, memcached_client, key_prefix: str):
        """
        Initialize a memcached-based agent memory.
        
        Args:
            k: Maximum number of message groups to store
            memcached_client: A memcached client instance
            key_prefix: Prefix for memcached keys to avoid collisions
        """
        self.k = k
        self.memcached_client = memcached_client
        self.key_prefix = key_prefix
        self._ensure_memory_exists()
    
    def _get_memory_key(self) -> str:
        """Get the memcached key for this memory"""
        return f"{self.key_prefix}:memory"
    
    def _ensure_memory_exists(self) -> None:
        """Ensure the memory exists in memcached"""
        key = self._get_memory_key()
        if not self.memcached_client.get(key):
            self.memcached_client.set(key, json.dumps([]))
    
    def add_messages(self, msgs: List[Dict[str, Any]]) -> None:
        """
        Add messages to the memory.
        
        Args:
            msgs: Messages to add
        """
        key = self._get_memory_key()
        memory_json = self.memcached_client.get(key)
        if not memory_json:
            memory = []
        else:
            memory = json.loads(memory_json)
        
        # Add new messages
        memory.append(msgs)
        
        # Trim if necessary
        if len(memory) > self.k:
            memory = memory[-self.k:]
        
        # Save back to memcached
        self.memcached_client.set(key, json.dumps(memory))
    
    def load_memory(self) -> List[Dict[str, Any]]:
        """
        Load all messages from memory.
        
        Returns:
            List of messages
        """
        key = self._get_memory_key()
        memory_json = self.memcached_client.get(key)
        if not memory_json:
            return []
        
        memory = json.loads(memory_json)
        # Flatten the list of message groups
        return [m for chat in memory for m in chat]
    
    def is_empty(self) -> bool:
        """
        Check if memory is empty.
        
        Returns:
            True if memory is empty, False otherwise
        """
        key = self._get_memory_key()
        memory_json = self.memcached_client.get(key)
        if not memory_json:
            return True
        
        memory = json.loads(memory_json)
        return len(memory) == 0


class AsyncMemcachedAgentMemory(AsyncAgentMemory):
    """
    Asynchronous memcached-based implementation of agent memory.
    Stores messages in memcached.
    """
    
    def __init__(self, k: int, async_client, key_prefix: str):
        """
        Initialize an async memcached-based agent memory.
        
        Args:
            k: Maximum number of message groups to store
            async_client: An async memcached client instance (e.g., aiomcache.Client)
            key_prefix: Prefix for memcached keys to avoid collisions
        """
        self.k = k
        self.async_client = async_client
        self.key_prefix = key_prefix
        # We'll initialize the memory in the first operation
    
    def _get_memory_key(self) -> str:
        """Get the memcached key for this memory"""
        return f"{self.key_prefix}:memory"
    
    async def _ensure_memory_exists(self) -> None:
        """Ensure the memory exists in memcached"""
        key = self._get_memory_key()
        value = await self.async_client.get(key)
        if not value:
            await self.async_client.set(key, json.dumps([]).encode('utf-8'))
    
    async def add_messages(self, msgs: List[Dict[str, Any]]) -> None:
        """
        Add messages to the memory asynchronously.
        
        Args:
            msgs: Messages to add
        """
        await self._ensure_memory_exists()
        
        key = self._get_memory_key()
        memory_bytes = await self.async_client.get(key)
        
        if not memory_bytes:
            memory = []
        else:
            memory = json.loads(memory_bytes.decode('utf-8'))
        
        # Add new messages
        memory.append(msgs)
        
        # Trim if necessary
        if len(memory) > self.k:
            memory = memory[-self.k:]
        
        # Save back to memcached
        await self.async_client.set(key, json.dumps(memory).encode('utf-8'))
    
    async def load_memory(self) -> List[Dict[str, Any]]:
        """
        Load all messages from memory asynchronously.
        
        Returns:
            List of messages
        """
        await self._ensure_memory_exists()
        
        key = self._get_memory_key()
        memory_bytes = await self.async_client.get(key)
        
        if not memory_bytes:
            return []
        
        memory = json.loads(memory_bytes.decode('utf-8'))
        # Flatten the list of message groups
        return [m for chat in memory for m in chat]
    
    async def is_empty(self) -> bool:
        """
        Check if memory is empty asynchronously.
        
        Returns:
            True if memory is empty, False otherwise
        """
        await self._ensure_memory_exists()
        
        key = self._get_memory_key()
        memory_bytes = await self.async_client.get(key)
        
        if not memory_bytes:
            return True
        
        memory = json.loads(memory_bytes.decode('utf-8'))
        return len(memory) == 0


class MemcachedAgentMemoryBank(AgentMemoryBank):
    """
    Synchronous memcached-based implementation of agent memory bank.
    Stores agent memories in memcached.
    """
    # Static mapping of user IDs to memory banks
    _user_memory_banks: ClassVar[DictType[str, 'MemcachedAgentMemoryBank']] = {}
    
    def __init__(self, memcached_client, key_prefix: str = "agent_memory"):
        """
        Initialize a memcached-based agent memory bank.
        
        Args:
            memcached_client: A memcached client instance
            key_prefix: Prefix for memcached keys to avoid collisions
        """
        self.memcached_client = memcached_client
        self.key_prefix = key_prefix
        # We maintain a local cache of agent memories to avoid recreating them
        self.agent_memories: Dict[str, MemcachedAgentMemory] = {}
    
    def _get_agents_key(self) -> str:
        """Get the memcached key for the list of agents"""
        return f"{self.key_prefix}:agents"
    
    def _ensure_agents_exists(self) -> None:
        """Ensure the agents list exists in memcached"""
        key = self._get_agents_key()
        if not self.memcached_client.get(key):
            self.memcached_client.set(key, json.dumps([]))
    
    def get_agent_memory(self, agent_name: str, k: int = 5) -> MemcachedAgentMemory:
        """
        Get memory for a named agent.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            
        Returns:
            Agent memory for the specified agent
        """
        # Check local cache first
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]
        
        # Create a new memory
        memory_key_prefix = f"{self.key_prefix}:{agent_name}"
        mem = MemcachedAgentMemory(
            k=k, 
            memcached_client=self.memcached_client, 
            key_prefix=memory_key_prefix
        )
        
        # Add to local cache
        self.agent_memories[agent_name] = mem
        
        # Add to the list of agents if not already there
        self._ensure_agents_exists()
        agents_key = self._get_agents_key()
        agents_json = self.memcached_client.get(agents_key)
        agents = json.loads(agents_json)
        
        if agent_name not in agents:
            agents.append(agent_name)
            self.memcached_client.set(agents_key, json.dumps(agents))
        
        return mem
    
    @classmethod
    def bank_for(cls, user_id: str, **kwargs) -> 'MemcachedAgentMemoryBank':
        """
        Get user memory bank from global store. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If memcached_client is not provided for a new bank
        """
        if user_id in cls._user_memory_banks:
            return cls._user_memory_banks[user_id]
        
        memcached_client = kwargs.get('memcached_client')
        
        if memcached_client is None:
            raise ValueError("memcached_client must be provided when creating a new memory bank")
        
        memory_bank = MemcachedAgentMemoryBank(
            memcached_client=memcached_client,
            key_prefix=f"user:{user_id}"
        )
        cls._user_memory_banks[user_id] = memory_bank
        return memory_bank
    
    @classmethod
    def clear_bank_for(cls, user_id: str) -> None:
        """
        Clear user memory bank from global store.
        
        Args:
            user_id: User ID
        """
        if user_id in cls._user_memory_banks:
            # Get the memory bank
            memory_bank = cls._user_memory_banks[user_id]
            
            # Get the list of agents
            agents_key = memory_bank._get_agents_key()
            agents_json = memory_bank.memcached_client.get(agents_key)
            
            if agents_json:
                agents = json.loads(agents_json)
                
                # Delete each agent's memory
                for agent_name in agents:
                    memory_key = f"{memory_bank.key_prefix}:{agent_name}:memory"
                    memory_bank.memcached_client.delete(memory_key)
                
                # Delete the agents list
                memory_bank.memcached_client.delete(agents_key)
            
            # Remove from the static mapping
            del cls._user_memory_banks[user_id]
    
    @classmethod
    def is_active(cls, user_id: str) -> bool:
        """
        Check if user has memory bank in global store.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user has memory bank, False otherwise
        """
        return user_id in cls._user_memory_banks


class AsyncMemcachedAgentMemoryBank(AsyncAgentMemoryBank):
    """
    Asynchronous memcached-based implementation of agent memory bank.
    Stores agent memories in memcached.
    """
    # Static mapping of user IDs to memory banks
    _user_memory_banks: ClassVar[DictType[str, 'AsyncMemcachedAgentMemoryBank']] = {}
    
    def __init__(self, async_client, key_prefix: str = "agent_memory"):
        """
        Initialize an async memcached-based agent memory bank.
        
        Args:
            async_client: An async memcached client instance (e.g., aiomcache.Client)
            key_prefix: Prefix for memcached keys to avoid collisions
        """
        self.async_client = async_client
        self.key_prefix = key_prefix
        # We maintain a local cache of agent memories to avoid recreating them
        self.agent_memories: Dict[str, AsyncMemcachedAgentMemory] = {}
    
    def _get_agents_key(self) -> str:
        """Get the memcached key for the list of agents"""
        return f"{self.key_prefix}:agents"
    
    async def _ensure_agents_exists(self) -> None:
        """Ensure the agents list exists in memcached"""
        key = self._get_agents_key()
        value = await self.async_client.get(key)
        if not value:
            await self.async_client.set(key, json.dumps([]).encode('utf-8'))
    
    async def get_agent_memory(self, agent_name: str, k: int = 5) -> AsyncMemcachedAgentMemory:
        """
        Get memory for a named agent asynchronously.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            
        Returns:
            Agent memory for the specified agent
        """
        # Check local cache first
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]
        
        # Create a new memory
        memory_key_prefix = f"{self.key_prefix}:{agent_name}"
        mem = AsyncMemcachedAgentMemory(
            k=k, 
            async_client=self.async_client, 
            key_prefix=memory_key_prefix
        )
        
        # Add to local cache
        self.agent_memories[agent_name] = mem
        
        # Add to the list of agents if not already there
        await self._ensure_agents_exists()
        
        agents_key = self._get_agents_key()
        agents_bytes = await self.async_client.get(agents_key)
        agents = json.loads(agents_bytes.decode('utf-8'))
        
        if agent_name not in agents:
            agents.append(agent_name)
            await self.async_client.set(agents_key, json.dumps(agents).encode('utf-8'))
        
        return mem
    
    @classmethod
    async def bank_for(cls, user_id: str, **kwargs) -> 'AsyncMemcachedAgentMemoryBank':
        """
        Get user memory bank from global store asynchronously. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If async_client is not provided for a new bank
        """
        if user_id in cls._user_memory_banks:
            return cls._user_memory_banks[user_id]
        
        async_client = kwargs.get('async_client')
        
        if async_client is None:
            raise ValueError("async_client must be provided when creating a new memory bank")
        
        memory_bank = AsyncMemcachedAgentMemoryBank(
            async_client=async_client,
            key_prefix=f"user:{user_id}"
        )
        cls._user_memory_banks[user_id] = memory_bank
        return memory_bank
    
    @classmethod
    async def clear_bank_for(cls, user_id: str) -> None:
        """
        Clear user memory bank from global store asynchronously.
        
        Args:
            user_id: User ID
        """
        if user_id in cls._user_memory_banks:
            # Get the memory bank
            memory_bank = cls._user_memory_banks[user_id]
            
            # Get the list of agents
            agents_key = memory_bank._get_agents_key()
            agents_bytes = await memory_bank.async_client.get(agents_key)
            
            if agents_bytes:
                agents = json.loads(agents_bytes.decode('utf-8'))
                
                # Delete each agent's memory
                for agent_name in agents:
                    memory_key = f"{memory_bank.key_prefix}:{agent_name}:memory"
                    await memory_bank.async_client.delete(memory_key)
                
                # Delete the agents list
                await memory_bank.async_client.delete(agents_key)
            
            # Remove from the static mapping
            del cls._user_memory_banks[user_id]
    
    @classmethod
    async def is_active(cls, user_id: str) -> bool:
        """
        Check if user has memory bank in global store asynchronously.
        
        Args:
            user_id: User ID
            
        Returns:
            True if user has memory bank, False otherwise
        """
        return user_id in cls._user_memory_banks
