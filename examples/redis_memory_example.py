"""
Example of using Redis-based agent memory system.

This example demonstrates both synchronous and asynchronous usage of the Redis-based agent memory system.
"""

import asyncio
import redis
import redis.asyncio as aioredis

from intellifun.redis_agent_memory import (
    RedisAgentMemory,
    AsyncRedisAgentMemory,
    RedisAgentMemoryBank,
    AsyncRedisAgentMemoryBank
)


def demo_redis_memory():
    """Demonstrate using Redis-based agent memory"""
    print("\n=== Redis Memory Demo ===")
    
    try:
        # Create a Redis client
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Create a memory instance
        memory = RedisAgentMemory(
            k=5,  # Keep last 5 message groups
            redis_client=redis_client,
            key_prefix="redis_memory_demo"
        )
        
        # Add some messages
        messages = [
            {"role": "user", "content": "Hello, can you remember this in Redis?"},
            {"role": "assistant", "content": "Yes, I can store this in Redis."}
        ]
        memory.add_messages(messages)
        
        # Check if memory is empty
        is_empty = memory.is_empty()
        print(f"Is memory empty? {is_empty}")
        
        # Load memory
        loaded_messages = memory.load_memory()
        print(f"Memory contains {len(loaded_messages)} messages:")
        for msg in loaded_messages:
            print(f"- {msg['role']}: {msg['content']}")
        
        # Clean up
        redis_client.delete(memory._get_memory_key())
        print("Memory cleared")
        
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis server: {e}")
        print("Make sure Redis is running on localhost:6379")


def demo_redis_memory_bank():
    """Demonstrate using Redis-based agent memory bank"""
    print("\n=== Redis Memory Bank Demo ===")
    
    try:
        # Create a Redis client
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Get a memory bank for a user
        memory_bank = RedisAgentMemoryBank.bank_for(
            'redis_user',
            redis_client=redis_client
        )
        
        # Get memory for an agent
        memory = memory_bank.get_agent_memory('redis_agent', k=5)
        
        # Add some messages
        messages = [
            {"role": "user", "content": "Hello, can you remember this in Redis?"},
            {"role": "assistant", "content": "Yes, I can store this in Redis."}
        ]
        memory.add_messages(messages)
        
        # Check if memory is empty
        is_empty = memory.is_empty()
        print(f"Is memory empty? {is_empty}")
        
        # Load memory
        loaded_messages = memory.load_memory()
        print(f"Memory contains {len(loaded_messages)} messages:")
        for msg in loaded_messages:
            print(f"- {msg['role']}: {msg['content']}")
        
        # Check if memory bank is active
        is_active = RedisAgentMemoryBank.is_active('redis_user', redis_client=redis_client)
        print(f"Is memory bank active? {is_active}")
        
        # Clean up
        RedisAgentMemoryBank.clear_bank_for('redis_user', redis_client=redis_client)
        print("Memory bank cleared")
        
    except redis.exceptions.ConnectionError as e:
        print(f"Could not connect to Redis server: {e}")
        print("Make sure Redis is running on localhost:6379")


async def demo_async_redis_memory():
    """Demonstrate using async Redis-based agent memory"""
    print("\n=== Async Redis Memory Demo ===")
    
    try:
        # Create an async Redis client
        async_redis_client = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Create a memory instance
        memory = AsyncRedisAgentMemory(
            k=5,  # Keep last 5 message groups
            async_redis_client=async_redis_client,
            key_prefix="async_redis_memory_demo"
        )
        
        # Add some messages
        messages = [
            {"role": "user", "content": "Hello, can you remember this in Redis asynchronously?"},
            {"role": "assistant", "content": "Yes, I can store this in Redis asynchronously."}
        ]
        await memory.add_messages(messages)
        
        # Check if memory is empty
        is_empty = await memory.is_empty()
        print(f"Is memory empty? {is_empty}")
        
        # Load memory
        loaded_messages = await memory.load_memory()
        print(f"Memory contains {len(loaded_messages)} messages:")
        for msg in loaded_messages:
            print(f"- {msg['role']}: {msg['content']}")
        
        # Clean up
        await async_redis_client.delete(memory._get_memory_key())
        print("Memory cleared")
        
        # Close the async client
        await async_redis_client.close()
        
    except (ConnectionRefusedError, OSError) as e:
        print(f"Could not connect to Redis server: {e}")
        print("Make sure Redis is running on localhost:6379")


async def demo_async_redis_memory_bank():
    """Demonstrate using async Redis-based agent memory bank"""
    print("\n=== Async Redis Memory Bank Demo ===")
    
    try:
        # Create an async Redis client
        async_redis_client = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Get a memory bank for a user
        memory_bank = await AsyncRedisAgentMemoryBank.bank_for(
            'async_redis_user',
            async_redis_client=async_redis_client
        )
        
        # Get memory for an agent
        memory = await memory_bank.get_agent_memory('async_redis_agent', k=5)
        
        # Add some messages
        messages = [
            {"role": "user", "content": "Hello, can you remember this in Redis asynchronously?"},
            {"role": "assistant", "content": "Yes, I can store this in Redis asynchronously."}
        ]
        await memory.add_messages(messages)
        
        # Check if memory is empty
        is_empty = await memory.is_empty()
        print(f"Is memory empty? {is_empty}")
        
        # Load memory
        loaded_messages = await memory.load_memory()
        print(f"Memory contains {len(loaded_messages)} messages:")
        for msg in loaded_messages:
            print(f"- {msg['role']}: {msg['content']}")
        
        # Check if memory bank is active
        is_active = await AsyncRedisAgentMemoryBank.is_active(
            'async_redis_user', 
            async_redis_client=async_redis_client
        )
        print(f"Is memory bank active? {is_active}")
        
        # Clean up
        await AsyncRedisAgentMemoryBank.clear_bank_for(
            'async_redis_user',
            async_redis_client=async_redis_client
        )
        print("Memory bank cleared")
        
        # Close the async client
        await async_redis_client.close()
        
    except (ConnectionRefusedError, OSError) as e:
        print(f"Could not connect to Redis server: {e}")
        print("Make sure Redis is running on localhost:6379")


async def demo_agent_with_async_redis_memory():
    """Demonstrate using an Agent with async Redis memory operations"""
    print("\n=== Agent with Async Redis Memory Demo ===")
    
    # This would be implemented with an actual agent
    # For this example, we're just showing the memory part
    
    try:
        # Create an async Redis client
        async_redis_client = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Get a memory bank for a user
        memory_bank = await AsyncRedisAgentMemoryBank.bank_for(
            'agent_user',
            async_redis_client=async_redis_client
        )
        
        # Get memory for an agent
        memory = await memory_bank.get_agent_memory('agent', k=10)
        
        # The agent would use this memory for conversations
        # For example:
        await memory.add_messages([
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ])
        
        # Later, when the agent needs to recall the conversation:
        conversation_history = await memory.load_memory()
        print("Agent memory contains:")
        for msg in conversation_history:
            print(f"- {msg['role']}: {msg['content']}")
        
        # Clean up
        await AsyncRedisAgentMemoryBank.clear_bank_for(
            'agent_user',
            async_redis_client=async_redis_client
        )
        
        # Close the async client
        await async_redis_client.close()
        
    except (ConnectionRefusedError, OSError) as e:
        print(f"Could not connect to Redis server: {e}")
        print("Make sure Redis is running on localhost:6379")


async def run_async_demos():
    """Run all async demos"""
    await demo_async_redis_memory()
    await demo_async_redis_memory_bank()
    await demo_agent_with_async_redis_memory()


if __name__ == "__main__":
    # Run synchronous demos
    demo_redis_memory()
    demo_redis_memory_bank()
    
    # Run asynchronous demos
    asyncio.run(run_async_demos())
