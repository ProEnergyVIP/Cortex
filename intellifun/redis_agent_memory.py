"""
Redis-based implementation of agent memory system with separate sync and async classes.
Uses native Redis data structures for optimal performance.
"""

import pickle
from typing import List

from intellifun.agent_memory import AgentMemory, AsyncAgentMemory, AgentMemoryBank, AsyncAgentMemoryBank
from intellifun.message import Message


class RedisAgentMemory(AgentMemory):
    """
    Redis-based implementation of agent memory.
    Stores messages in Redis using native data structures.
    """
    
    def __init__(self, k: int, redis_client, key: str):
        """
        Initialize a Redis-based agent memory.
        
        Args:
            k: Maximum number of message groups to store
            redis_client: A Redis client instance
            key: Redis key for this memory
        """
        self.k = k
        self.redis_client = redis_client
        self.key = key
    
    def add_messages(self, msgs: List[Message]) -> None:
        """
        Add messages to the memory.
        
        Args:
            msgs: Messages to add
        """
        
        # Serialize the message list using pickle for better preservation of objects
        serialized_msgs = pickle.dumps(msgs)
        
        # Add new messages to the right of the list
        self.redis_client.rpush(self.key, serialized_msgs)
        
        # Trim if necessary to keep only the last k elements
        if self.redis_client.llen(self.key) > self.k:
            self.redis_client.ltrim(self.key, -self.k, -1)
    
    def load_memory(self) -> List[Message]:
        """
        Load all messages from memory.
        
        Returns:
            List of messages
        """
        
        # Get all elements from the list
        message_groups = self.redis_client.lrange(self.key, 0, -1)
        if not message_groups:
            return []
        
        # Parse each message group and flatten
        result = []
        for serialized_group in message_groups:
            group = pickle.loads(serialized_group)
            result.extend(group)
        
        return result
    
    def is_empty(self) -> bool:
        """
        Check if memory is empty.
        
        Returns:
            True if memory is empty, False otherwise
        """
        
        # Check if the list exists and has elements
        length = self.redis_client.llen(self.key)
        return length == 0


class AsyncRedisAgentMemory(AsyncAgentMemory):
    """
    Asynchronous Redis-based implementation of agent memory.
    Stores messages in Redis using native data structures.
    """
    
    def __init__(self, k: int, async_redis_client, key: str):
        """
        Initialize an async Redis-based agent memory.
        
        Args:
            k: Maximum number of message groups to store
            async_redis_client: An async Redis client instance
            key: Redis key for this memory
        """
        self.k = k
        self.async_redis_client = async_redis_client
        self.key = key
    
    async def add_messages(self, msgs: List[Message]) -> None:
        """
        Add messages to the memory asynchronously.
        
        Args:
            msgs: Messages to add
        """
        
        # Serialize the message list using pickle for better preservation of objects
        serialized_msgs = pickle.dumps(msgs)
        
        # Add new messages to the right of the list
        await self.async_redis_client.rpush(self.key, serialized_msgs)
        
        # Trim if necessary to keep only the last k elements
        if await self.async_redis_client.llen(self.key) > self.k:
            await self.async_redis_client.ltrim(self.key, -self.k, -1)
    
    async def load_memory(self) -> List[Message]:
        """
        Load all messages from memory asynchronously.
        
        Returns:
            List of messages
        """
        
        # Get all elements from the list
        message_groups = await self.async_redis_client.lrange(self.key, 0, -1)
        if not message_groups:
            return []
        
        # Parse each message group and flatten
        result = []
        for serialized_group in message_groups:
            group = pickle.loads(serialized_group)
            result.extend(group)
        
        return result
    
    async def is_empty(self) -> bool:
        """
        Check if memory is empty asynchronously.
        
        Returns:
            True if memory is empty, False otherwise
        """
        
        # Check if the list exists and has elements
        length = await self.async_redis_client.llen(self.key)
        return length == 0


class RedisAgentMemoryBank(AgentMemoryBank):
    """
    Synchronous Redis-based implementation of agent memory bank.
    Stores agent memories in Redis.
    """
    
    def __init__(self, redis_client, key_prefix: str = "agent_memory"):
        """
        Initialize a Redis-based agent memory bank.
        
        Args:
            redis_client: A Redis client instance
            key_prefix: Prefix for Redis keys to avoid collisions
        """
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        # Local cache of agent memories for performance
        self.agent_memories = {}
    
    def _get_agents_key(self) -> str:
        """Get the Redis key for the list of agents"""
        return f"{self.key_prefix}:agents"
    
    def get_agent_memory(self, agent_name: str, k: int = 5) -> RedisAgentMemory:
        """
        Get memory for a named agent.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            
        Returns:
            Agent memory for the specified agent
        """
        # Check local cache first for performance
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]
        
        # Create a new memory
        memory_key = f"{self.key_prefix}:{agent_name}"
        mem = RedisAgentMemory(
            k=k, 
            redis_client=self.redis_client, 
            key=memory_key
        )
        
        # Add to local cache for performance
        self.agent_memories[agent_name] = mem
        
        # Add to the set of agents
        agents_key = self._get_agents_key()
        self.redis_client.sadd(agents_key, agent_name)
        
        return mem
    
    @classmethod
    def bank_for(cls, user_id: str, **kwargs) -> 'RedisAgentMemoryBank':
        """
        Get user memory bank from global store. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If redis_client is not provided for a new bank
        """
        redis_client = kwargs.get('redis_client')
        
        if redis_client is None:
            raise ValueError("redis_client must be provided when creating a new memory bank")
        
        # Create a new memory bank
        memory_bank = RedisAgentMemoryBank(
            redis_client=redis_client,
            key_prefix=f"user:{user_id}"
        )
        
        return memory_bank
    
    @classmethod
    def clear_bank_for(cls, user_id: str, **kwargs) -> None:
        """
        Clear user memory bank from global store.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
            
        Raises:
            ValueError: If redis_client is not provided
        """
        redis_client = kwargs.get('redis_client')
        
        if redis_client is None:
            raise ValueError("redis_client must be provided to clear the memory bank")
        
        # Create a pattern to match all keys for this user
        key_pattern = f"user:{user_id}:*"
        
        # Lua script to delete all keys matching a pattern
        # This is more efficient as it runs atomically on the Redis server
        delete_script = """
        local keys = redis.call('keys', ARGV[1])
        if #keys > 0 then
            return redis.call('del', unpack(keys))
        end
        return 0
        """
        
        # Execute the Lua script
        redis_client.eval(delete_script, 0, key_pattern)
    
    @classmethod
    def is_active(cls, user_id: str, **kwargs) -> bool:
        """
        Check if user has memory bank in global store.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
            
        Returns:
            True if user has memory bank, False otherwise
            
        Raises:
            ValueError: If redis_client is not provided
        """
        redis_client = kwargs.get('redis_client')
        
        if redis_client is None:
            raise ValueError("redis_client must be provided to check if memory bank is active")
        
        # Create a temporary memory bank to access the keys
        memory_bank = RedisAgentMemoryBank(
            redis_client=redis_client,
            key_prefix=f"user:{user_id}"
        )
        
        # Check if the agents key exists
        agents_key = memory_bank._get_agents_key()
        return redis_client.exists(agents_key) > 0


class AsyncRedisAgentMemoryBank(AsyncAgentMemoryBank):
    """
    Asynchronous Redis-based implementation of agent memory bank.
    Stores agent memories in Redis.
    """
    
    def __init__(self, async_redis_client, key_prefix: str = "agent_memory"):
        """
        Initialize an async Redis-based agent memory bank.
        
        Args:
            async_redis_client: An async Redis client instance
            key_prefix: Prefix for Redis keys to avoid collisions
        """
        self.async_redis_client = async_redis_client
        self.key_prefix = key_prefix
        # Local cache of agent memories for performance
        self.agent_memories = {}
    
    def _get_agents_key(self) -> str:
        """Get the Redis key for the list of agents"""
        return f"{self.key_prefix}:agents"
    

    
    async def get_agent_memory(self, agent_name: str, k: int = 5) -> AsyncRedisAgentMemory:
        """
        Get memory for a named agent asynchronously.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            
        Returns:
            Agent memory for the specified agent
        """
        # Check local cache first for performance
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]
        
        # Create a new memory
        memory_key = f"{self.key_prefix}:{agent_name}"
        mem = AsyncRedisAgentMemory(
            k=k, 
            async_redis_client=self.async_redis_client, 
            key=memory_key
        )
        
        # Add to local cache for performance
        self.agent_memories[agent_name] = mem
        
        # Add to the set of agents
        agents_key = self._get_agents_key()
        await self.async_redis_client.sadd(agents_key, agent_name)
        
        return mem
    
    @classmethod
    async def bank_for(cls, user_id: str, **kwargs) -> 'AsyncRedisAgentMemoryBank':
        """
        Get user memory bank from global store asynchronously. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If async_redis_client is not provided for a new bank
        """
        async_redis_client = kwargs.get('async_redis_client')
        
        if async_redis_client is None:
            raise ValueError("async_redis_client must be provided when creating a new memory bank")
        
        # Create a new memory bank
        memory_bank = AsyncRedisAgentMemoryBank(
            async_redis_client=async_redis_client,
            key_prefix=f"user:{user_id}"
        )
        
        return memory_bank
    
    @classmethod
    async def clear_bank_for(cls, user_id: str, **kwargs) -> None:
        """
        Clear user memory bank from global store asynchronously.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
            
        Raises:
            ValueError: If async_redis_client is not provided
        """
        async_redis_client = kwargs.get('async_redis_client')
        
        if async_redis_client is None:
            raise ValueError("async_redis_client must be provided to clear the memory bank")
        
        # Create a pattern to match all keys for this user
        key_pattern = f"user:{user_id}:*"
        
        # Lua script to delete all keys matching a pattern
        # This is more efficient as it runs atomically on the Redis server
        delete_script = """
        local keys = redis.call('keys', ARGV[1])
        if #keys > 0 then
            return redis.call('del', unpack(keys))
        end
        return 0
        """
        
        # Execute the Lua script
        await async_redis_client.eval(delete_script, 0, key_pattern)
    
    @classmethod
    async def is_active(cls, user_id: str, **kwargs) -> bool:
        """
        Check if user has memory bank in global store asynchronously.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
            
        Returns:
            True if user has memory bank, False otherwise
            
        Raises:
            ValueError: If async_redis_client is not provided
        """
        async_redis_client = kwargs.get('async_redis_client')
        
        if async_redis_client is None:
            raise ValueError("async_redis_client must be provided to check if memory bank is active")
        
        # Create a temporary memory bank to access the keys
        memory_bank = AsyncRedisAgentMemoryBank(
            async_redis_client=async_redis_client,
            key_prefix=f"user:{user_id}"
        )
        
        # Check if the agents key exists
        agents_key = memory_bank._get_agents_key()
        exists = await memory_bank.async_redis_client.exists(agents_key)
        return exists > 0
