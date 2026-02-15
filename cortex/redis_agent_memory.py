"""
Redis-based implementation of agent memory system with separate sync and async classes.
Uses native Redis data structures for optimal performance.
"""

import logging
import pickle
from typing import Callable, List, Optional

from cortex.LLM import LLM
from cortex.agent_memory import (
    AgentMemory, AsyncAgentMemory, AgentMemoryBank, AsyncAgentMemoryBank,
    _build_default_summary_fn_sync, _build_default_summary_fn_async,
    _format_messages_for_summary,
)
from cortex.message import Message, SystemMessage

logger = logging.getLogger(__name__)

# Shared Lua script to delete all keys matching a pattern
DELETE_BY_PATTERN_LUA = """
local keys = redis.call('keys', ARGV[1])
if #keys > 0 then
    return redis.call('del', unpack(keys))
end
return 0
"""


class RedisAgentMemory(AgentMemory):
    """
    Redis-based implementation of agent memory.
    Stores messages in Redis using native data structures.
    
    Supports the same conversation summary feature as AgentMemory.
    The summary text is persisted in Redis under ``{key}:summary``.
    
    Args:
        k: Maximum number of message groups to store.
        redis_client: A Redis client instance.
        key: Redis key for this memory.
        enable_summary: If True, enable periodic conversation summarization.
        summary_fn: Custom sync summarization function with signature
            ``(current_summary: str, messages: List[Message]) -> str``.
        summary_llm: LLM instance for the default summarizer.
        summarize_every_n: Run summarization every N evictions. Default 3.
    """
    
    def __init__(self, k: int, redis_client, key: str,
                 enable_summary: bool = False,
                 summary_fn: Optional[Callable] = None,
                 summary_llm: Optional[object] = None,
                 summarize_every_n: int = 3):
        self.k = k
        self.redis_client = redis_client
        self.key = key
        self.enable_summary = enable_summary
        self.summary_fn = summary_fn
        self.summary_llm = summary_llm
        self.summarize_every_n = summarize_every_n
        # Transient in-memory state
        self._eviction_counter: int = 0
        self._eviction_buffer: List[Message] = []
        self._summary_fn_resolved: Optional[Callable] = None

    @property
    def _summary_key(self) -> str:
        return f"{self.key}:summary"

    def _get_summary_fn(self) -> Callable:
        """Lazily resolve the summarization function."""
        if self._summary_fn_resolved is not None:
            return self._summary_fn_resolved
        if self.summary_fn is not None:
            self._summary_fn_resolved = self.summary_fn
        else:
            llm = self.summary_llm
            if llm is None:
                llm = LLM(model='gpt-5-nano')
                self.summary_llm = llm
            self._summary_fn_resolved = _build_default_summary_fn_sync(llm)
        return self._summary_fn_resolved

    def _run_summarization(self, buffered_msgs: List[Message]) -> None:
        """Run the summarization function on buffered evicted messages."""
        try:
            fn = self._get_summary_fn()
            current = self.get_summary()
            new_summary = fn(current, buffered_msgs)
            self.set_summary(new_summary)
            logger.debug("Redis conversation summary updated (%d chars)", len(new_summary))
        except Exception as e:
            logger.warning("Redis conversation summarization failed: %s", e)

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
            # Grab the evicted round before trimming
            evicted_data = self.redis_client.lindex(self.key, 0)
            self.redis_client.ltrim(self.key, -self.k, -1)

            if self.enable_summary and evicted_data is not None:
                self._eviction_buffer.extend(pickle.loads(evicted_data))
                self._eviction_counter += 1
                if self._eviction_counter >= self.summarize_every_n:
                    self._run_summarization(self._eviction_buffer)
                    self._eviction_buffer = []
                    self._eviction_counter = 0
    
    def load_memory(self) -> List[Message]:
        """
        Load all messages from memory.
        
        If a conversation summary exists in Redis, it is prepended as a
        SystemMessage so the agent has access to important earlier context.
        
        If there are buffered evicted messages not yet summarized, they are
        also prepended so the LLM never loses visibility of recent evictions.
        
        Returns:
            List of messages
        """
        
        # Get all elements from the list
        message_groups = self.redis_client.lrange(self.key, 0, -1)
        
        prefix = []
        summary = self.get_summary()
        if summary:
            prefix.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
        if self._eviction_buffer:
            buffer_text = _format_messages_for_summary(self._eviction_buffer)
            prefix.append(SystemMessage(content=f"Recent conversation not yet in summary:\n{buffer_text}"))
        
        result = list(prefix)
        if message_groups:
            for serialized_group in message_groups:
                group = pickle.loads(serialized_group)
                result.extend(group)
        
        return result
    
    def get_summary(self) -> str:
        """Return the current conversation summary from Redis."""
        val = self.redis_client.get(self._summary_key)
        if val is None:
            return ""
        return val if isinstance(val, str) else val.decode('utf-8')

    def set_summary(self, summary: str) -> None:
        """Persist the conversation summary to Redis."""
        if summary:
            self.redis_client.set(self._summary_key, summary)
        else:
            self.redis_client.delete(self._summary_key)

    def is_empty(self) -> bool:
        """
        Check if memory is empty.
        
        Returns:
            True if memory is empty, False otherwise
        """
        
        length = self.redis_client.llen(self.key)
        if length > 0:
            return False
        return not self.get_summary()


class AsyncRedisAgentMemory(AsyncAgentMemory):
    """
    Asynchronous Redis-based implementation of agent memory.
    Stores messages in Redis using native data structures.
    
    Supports the same conversation summary feature as AsyncAgentMemory.
    The summary text is persisted in Redis under ``{key}:summary``.
    
    Args:
        k: Maximum number of message groups to store.
        async_redis_client: An async Redis client instance.
        key: Redis key for this memory.
        enable_summary: If True, enable periodic conversation summarization.
        summary_fn: Custom async summarization function with signature
            ``async (current_summary: str, messages: List[Message]) -> str``.
        summary_llm: LLM instance for the default summarizer.
        summarize_every_n: Run summarization every N evictions. Default 3.
    """
    
    def __init__(self, k: int, async_redis_client, key: str,
                 enable_summary: bool = False,
                 summary_fn: Optional[Callable] = None,
                 summary_llm: Optional[object] = None,
                 summarize_every_n: int = 3):
        self.k = k
        self.async_redis_client = async_redis_client
        self.key = key
        self.enable_summary = enable_summary
        self.summary_fn = summary_fn
        self.summary_llm = summary_llm
        self.summarize_every_n = summarize_every_n
        # Transient in-memory state
        self._eviction_counter: int = 0
        self._eviction_buffer: List[Message] = []
        self._summary_fn_resolved: Optional[Callable] = None

    @property
    def _summary_key(self) -> str:
        return f"{self.key}:summary"

    def _get_summary_fn(self) -> Callable:
        """Lazily resolve the async summarization function."""
        if self._summary_fn_resolved is not None:
            return self._summary_fn_resolved
        if self.summary_fn is not None:
            self._summary_fn_resolved = self.summary_fn
        else:
            llm = self.summary_llm
            if llm is None:
                llm = LLM(model='gpt-5-nano')
                self.summary_llm = llm
            self._summary_fn_resolved = _build_default_summary_fn_async(llm)
        return self._summary_fn_resolved

    async def _run_summarization(self, buffered_msgs: List[Message]) -> None:
        """Run the async summarization function on buffered evicted messages."""
        try:
            fn = self._get_summary_fn()
            current = await self.get_summary()
            new_summary = await fn(current, buffered_msgs)
            await self.set_summary(new_summary)
            logger.debug("Redis async conversation summary updated (%d chars)", len(new_summary))
        except Exception as e:
            logger.warning("Redis async conversation summarization failed: %s", e)

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
            # Grab the evicted round before trimming
            evicted_data = await self.async_redis_client.lindex(self.key, 0)
            await self.async_redis_client.ltrim(self.key, -self.k, -1)

            if self.enable_summary and evicted_data is not None:
                self._eviction_buffer.extend(pickle.loads(evicted_data))
                self._eviction_counter += 1
                if self._eviction_counter >= self.summarize_every_n:
                    await self._run_summarization(self._eviction_buffer)
                    self._eviction_buffer = []
                    self._eviction_counter = 0
    
    async def load_memory(self) -> List[Message]:
        """
        Load all messages from memory asynchronously.
        
        If a conversation summary exists in Redis, it is prepended as a
        SystemMessage so the agent has access to important earlier context.
        
        If there are buffered evicted messages not yet summarized, they are
        also prepended so the LLM never loses visibility of recent evictions.
        
        Returns:
            List of messages
        """
        
        message_groups = await self.async_redis_client.lrange(self.key, 0, -1)
        
        prefix = []
        summary = await self.get_summary()
        if summary:
            prefix.append(SystemMessage(content=f"Summary of earlier conversation:\n{summary}"))
        if self._eviction_buffer:
            buffer_text = _format_messages_for_summary(self._eviction_buffer)
            prefix.append(SystemMessage(content=f"Recent conversation not yet in summary:\n{buffer_text}"))
        
        result = list(prefix)
        if message_groups:
            for serialized_group in message_groups:
                group = pickle.loads(serialized_group)
                result.extend(group)
        
        return result
    
    async def get_summary(self) -> str:
        """Return the current conversation summary from Redis."""
        val = await self.async_redis_client.get(self._summary_key)
        if val is None:
            return ""
        return val if isinstance(val, str) else val.decode('utf-8')

    async def set_summary(self, summary: str) -> None:
        """Persist the conversation summary to Redis."""
        if summary:
            await self.async_redis_client.set(self._summary_key, summary)
        else:
            await self.async_redis_client.delete(self._summary_key)

    async def is_empty(self) -> bool:
        """
        Check if memory is empty asynchronously.
        
        Returns:
            True if memory is empty, False otherwise
        """
        
        length = await self.async_redis_client.llen(self.key)
        if length > 0:
            return False
        return not await self.get_summary()


class RedisAgentMemoryBank(AgentMemoryBank):
    """
    Synchronous Redis-based implementation of agent memory bank.
    Stores agent memories in Redis.
    """
    # Static mapping of Redis clients
    _redis_clients = {}
    
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
    
    @classmethod
    def register_client(cls, client_name: str, redis_client):
        """
        Register a Redis client for later use.
        
        Args:
            client_name: Name to identify this client
            redis_client: Redis client instance
        """
        cls._redis_clients[client_name] = redis_client
    
    @classmethod
    def _resolve_client(cls, kwargs):
        """Resolve a Redis client from kwargs or registry."""
        redis_client = kwargs.get('redis_client')
        client_name = kwargs.get('client_name', 'default')
        if redis_client is None:
            redis_client = cls._redis_clients.get(client_name)
        if redis_client is None:
            raise ValueError("No Redis client available. Either provide 'redis_client' or register a client with 'client_name'")
        return redis_client
    
    @staticmethod
    def _delete_by_pattern(redis_client, key_pattern: str) -> None:
        """Delete all keys matching pattern using Lua script."""
        redis_client.eval(DELETE_BY_PATTERN_LUA, 0, key_pattern)
    
    def _get_agents_key(self) -> str:
        """Get the Redis key for the list of agents"""
        return f"{self.key_prefix}:agents"
    
    def get_agent_memory(self, agent_name: str, k: int = 5, **kwargs) -> RedisAgentMemory:
        """
        Get memory for a named agent.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            **kwargs: Additional keyword arguments forwarded to RedisAgentMemory
                (e.g. enable_summary, summary_fn, summary_llm, summarize_every_n).
            
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
            key=memory_key,
            **kwargs
        )
        
        # Add to local cache for performance
        self.agent_memories[agent_name] = mem
        
        # Add to the set of agents
        agents_key = self._get_agents_key()
        self.redis_client.sadd(agents_key, agent_name)
        
        return mem
    
    def reset_memory(self):
        # Delete all keys for this bank's prefix
        self._delete_by_pattern(self.redis_client, self.key_prefix + ':*')
        
        # Clear the local cache
        self.agent_memories.clear()

    @classmethod
    def bank_for(cls, user_id: str, **kwargs) -> 'RedisAgentMemoryBank':
        """
        Get user memory bank from global store. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
                client_name: Name of a registered Redis client to use
                redis_client: A Redis client instance (alternative to client_name)
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If no Redis client is available
        """
        # Try to get client from kwargs
        redis_client = kwargs.get('redis_client')
        client_name = kwargs.get('client_name', 'default')
        
        # If no direct client provided, try to get from registered clients
        if redis_client is None:
            redis_client = cls._redis_clients.get(client_name)
        
        if redis_client is None:
            raise ValueError("No Redis client available. Either provide 'redis_client' or register a client with 'client_name'")
        
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
                client_name: Name of a registered Redis client to use
                redis_client: A Redis client instance (alternative to client_name)
            
        Raises:
            ValueError: If no Redis client is available
        """
        # Resolve client
        redis_client = cls._resolve_client(kwargs)
        
        # Create a pattern to match all keys for this user
        key_pattern = f"user:{user_id}:*"
        # Delete by pattern
        cls._delete_by_pattern(redis_client, key_pattern)
    
    @classmethod
    def reset_all(cls, **kwargs) -> None:
        """
        Reset all memory banks for all users by deleting all matching Redis keys.
        
        Args:
            **kwargs: Additional arguments for accessing Redis
                client_name: Name of a registered Redis client to use
                redis_client: A Redis client instance (alternative to client_name)
        
        Raises:
            ValueError: If no Redis client is available
        """
        # Resolve client
        redis_client = cls._resolve_client(kwargs)
        
        # Pattern for all users
        key_pattern = "user:*"
        # Delete by pattern
        cls._delete_by_pattern(redis_client, key_pattern)
    
    @classmethod
    def is_active(cls, user_id: str, **kwargs) -> bool:
        """
        Check if user has memory bank in global store.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
                client_name: Name of a registered Redis client to use
                redis_client: A Redis client instance (alternative to client_name)
            
        Returns:
            True if user has memory bank, False otherwise
            
        Raises:
            ValueError: If no Redis client is available
        """
        # Resolve client
        redis_client = cls._resolve_client(kwargs)
        # Check if the agents key exists for this user
        agents_key = f"user:{user_id}:agents"
        return redis_client.exists(agents_key) > 0


class AsyncRedisAgentMemoryBank(AsyncAgentMemoryBank):
    """
    Asynchronous Redis-based implementation of agent memory bank.
    Stores agent memories in Redis.
    """
    # Static mapping of async Redis clients
    _async_redis_clients = {}
    
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
    
    @classmethod
    def register_client(cls, client_name: str, async_redis_client):
        """
        Register an async Redis client for later use.
        
        Args:
            client_name: Name to identify this client
            async_redis_client: Async Redis client instance
        """
        cls._async_redis_clients[client_name] = async_redis_client
    
    @classmethod
    def _resolve_client(cls, kwargs):
        """Resolve an async Redis client from kwargs or registry."""
        async_redis_client = kwargs.get('async_redis_client')
        client_name = kwargs.get('client_name', 'default')
        if async_redis_client is None:
            async_redis_client = cls._async_redis_clients.get(client_name)
        if async_redis_client is None:
            raise ValueError("No async Redis client available. Either provide 'async_redis_client' or register a client with 'client_name'")
        return async_redis_client
    
    @staticmethod
    async def _delete_by_pattern(async_redis_client, key_pattern: str) -> None:
        """Delete all keys matching pattern using Lua script (async)."""
        await async_redis_client.eval(DELETE_BY_PATTERN_LUA, 0, key_pattern)
    
    def _get_agents_key(self) -> str:
        """Get the Redis key for the list of agents"""
        return f"{self.key_prefix}:agents"
    
    async def get_agent_memory(self, agent_name: str, k: int = 5, **kwargs) -> AsyncRedisAgentMemory:
        """
        Get memory for a named agent asynchronously.
        
        Args:
            agent_name: Name of the agent
            k: Maximum number of message groups to store
            **kwargs: Additional keyword arguments forwarded to AsyncRedisAgentMemory
                (e.g. enable_summary, summary_fn, summary_llm, summarize_every_n).
            
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
            key=memory_key,
            **kwargs
        )
        
        # Add to local cache for performance
        self.agent_memories[agent_name] = mem
        
        # Add to the set of agents
        agents_key = self._get_agents_key()
        await self.async_redis_client.sadd(agents_key, agent_name)
        
        return mem
    
    async def reset_memory(self):
        # Delete all keys for this bank's prefix
        await self._delete_by_pattern(self.async_redis_client, self.key_prefix + ':*')
        
        # Clear the local cache
        self.agent_memories.clear()
    
    @classmethod
    async def bank_for(cls, user_id: str, **kwargs) -> 'AsyncRedisAgentMemoryBank':
        """
        Get user memory bank from global store asynchronously. If it doesn't exist, create one.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for creating a new bank
                client_name: Name of a registered async Redis client to use
                async_redis_client: An async Redis client instance (alternative to client_name)
            
        Returns:
            Memory bank for the specified user
            
        Raises:
            ValueError: If no async Redis client is available
        """
        # Try to get client from kwargs
        async_redis_client = kwargs.get('async_redis_client')
        client_name = kwargs.get('client_name', 'default')
        
        # If no direct client provided, try to get from registered clients
        if async_redis_client is None:
            async_redis_client = cls._async_redis_clients.get(client_name)
        
        if async_redis_client is None:
            raise ValueError("No async Redis client available. Either provide 'async_redis_client' or register a client with 'client_name'")
        
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
                client_name: Name of a registered async Redis client to use
                async_redis_client: An async Redis client instance (alternative to client_name)
            
        Raises:
            ValueError: If no async Redis client is available
        """
        # Resolve client
        async_redis_client = cls._resolve_client(kwargs)
        
        # Create a pattern to match all keys for this user
        key_pattern = f"user:{user_id}:*"
        # Delete by pattern
        await cls._delete_by_pattern(async_redis_client, key_pattern)
    
    @classmethod
    async def reset_all(cls, **kwargs) -> None:
        """
        Reset all memory banks for all users asynchronously by deleting all matching Redis keys.
        
        Args:
            **kwargs: Additional arguments for accessing Redis
                client_name: Name of a registered async Redis client to use
                async_redis_client: An async Redis client instance (alternative to client_name)
        
        Raises:
            ValueError: If no async Redis client is available
        """
        # Resolve client
        async_redis_client = cls._resolve_client(kwargs)
        
        # Pattern for all users
        key_pattern = "user:*"
        # Delete by pattern
        await cls._delete_by_pattern(async_redis_client, key_pattern)
    
    @classmethod
    async def is_active(cls, user_id: str, **kwargs) -> bool:
        """
        Check if user has memory bank in global store asynchronously.
        
        Args:
            user_id: User ID
            **kwargs: Additional arguments for accessing the memory bank
                client_name: Name of a registered async Redis client to use
                async_redis_client: An async Redis client instance (alternative to client_name)
            
        Returns:
            True if user has memory bank, False otherwise
            
        Raises:
            ValueError: If no async Redis client is available
        """
        # Resolve client
        async_redis_client = cls._resolve_client(kwargs)
        # Check if the agents key exists for this user
        agents_key = f"user:{user_id}:agents"
        exists = await async_redis_client.exists(agents_key)
        return exists > 0
