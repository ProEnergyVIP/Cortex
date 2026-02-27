"""
Example: Whiteboard Usage in Agent System - Channel-Based Messaging

This example demonstrates how to use the new channel-based whiteboard feature
for multi-agent collaboration. The whiteboard enables asynchronous communication
between the coordinator and worker agents through channels.

Key Concepts:
- Channels: String identifiers for organizing messages (e.g., "project:demo")
- Messages: JSON content with metadata (sender, timestamp, thread, reply_to)
- Storage: InMemory (default) or Redis (persistent)
- Tools: whiteboard_post, whiteboard_read, whiteboard_subscribe
"""

import asyncio
from cortex import LLM, GPTModels, AgentSystemContext, AsyncAgentMemoryBank
from cortex.agent_system import (
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    CoordinatorSystem,
)
from cortex.agent_system.core.whiteboard import Whiteboard, RedisStorage


async def basic_whiteboard_example():
    """Demonstrate basic whiteboard operations."""
    
    print("\n" + "=" * 60)
    print("Example 1: Basic Whiteboard Operations")
    print("=" * 60)
    
    # Create a whiteboard with default in-memory storage
    wb = Whiteboard()
    
    # Post messages to different channels
    await wb.post(
        sender="Coordinator",
        channel="project:acme-merger",
        content={"type": "goal", "description": "Analyze merger risks"}
    )
    
    await wb.post(
        sender="RiskAnalyst",
        channel="project:acme-merger",
        content={"type": "finding", "risk": "High debt-to-equity ratio"},
        thread="financial-analysis"
    )
    
    await wb.post(
        sender="LegalAnalyst",
        channel="project:acme-merger",
        content={"type": "finding", "risk": "Antitrust review required"},
        thread="regulatory-analysis"
    )
    
    # Read all messages from the channel
    messages = await wb.read(channel="project:acme-merger")
    print(f"Posted and read {len(messages)} messages")
    
    # Read only messages from a specific thread
    financial_msgs = await wb.read(
        channel="project:acme-merger",
        thread="financial-analysis"
    )
    print(f"Financial analysis thread: {len(financial_msgs)} messages")
    
    # List all channels
    channels = wb.list_channels()
    print(f"Active channels: {channels}")
    
    await wb.close()


async def redis_whiteboard_example():
    """Demonstrate Redis-backed whiteboard (skipped if Redis unavailable)."""
    
    print("\n" + "=" * 60)
    print("Example 2: Redis Persistence")
    print("=" * 60)
    
    try:
        from redis.asyncio import Redis
        
        redis_client = Redis(host='localhost', port=6379)
        await redis_client.ping()  # Test connection
        
        # Create whiteboard with Redis storage
        wb = Whiteboard(storage=RedisStorage(
            redis_client=redis_client,
            key_prefix="demo:whiteboard"
        ))
        
        await wb.post(
            sender="System",
            channel="demo:redis",
            content={"message": "This will persist in Redis"}
        )
        
        messages = await wb.read(channel="demo:redis")
        print(f"✓ Redis-backed whiteboard working: {len(messages)} messages")
        
        await wb.close()
        
    except Exception as e:
        print(f"(Skipped - Redis not available: {e})")


async def agent_system_with_whiteboard():
    """Demonstrate agent system with whiteboard integration."""
    
    print("\n" + "=" * 60)
    print("Example 3: Agent System with Whiteboard")
    print("=" * 60)
    
    # Create context with whiteboard enabled
    context = AgentSystemContext.create(
        memory_bank=AsyncAgentMemoryBank(),
        enable_whiteboard=True,
    )
    
    # Create worker builders
    math_worker = WorkerAgentBuilder(
        name="Math Expert",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: "You are a math expert. Show your work.",
        introduction="Expert in mathematical calculations.",
    )
    
    # Create coordinator
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: "You coordinate workers using the whiteboard.",
    )
    
    # Build the system
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[math_worker],
        context=context,
    )
    
    print("✓ Agent system created with whiteboard integration")
    print("  - Agents can use whiteboard_post to share information")
    print("  - Agents can use whiteboard_read to get context")
    print("  - Messages persist across the conversation")
    
    # Demonstrate that whiteboard tools are available
    has_whiteboard = context.whiteboard is not None
    print(f"  - Whiteboard enabled: {has_whiteboard}")


async def custom_whiteboard_extension():
    """Demonstrate extending the whiteboard for custom needs."""
    
    print("\n" + "=" * 60)
    print("Example 4: Custom Whiteboard Extension")
    print("=" * 60)
    
    class LoggingWhiteboard(Whiteboard):
        """Whiteboard that logs all operations."""
        
        def __init__(self, storage=None):
            super().__init__(storage)
            self.operations = []
        
        async def _after_post(self, message):
            self.operations.append({
                "action": "post",
                "sender": message.sender,
                "channel": message.channel,
                "time": message.timestamp.isoformat(),
            })
        
        async def _after_read(self, channel, messages):
            self.operations.append({
                "action": "read",
                "channel": channel,
                "count": len(messages),
            })
    
    # Create custom whiteboard
    wb = LoggingWhiteboard()
    
    # Perform operations
    await wb.post(sender="A", channel="test", content={"msg": 1})
    await wb.post(sender="B", channel="test", content={"msg": 2})
    await wb.read(channel="test")
    
    print("✓ Custom whiteboard with logging created")
    print(f"  - Logged operations: {len(wb.operations)}")
    print(f"  - Posts: {sum(1 for op in wb.operations if op['action'] == 'post')}")
    print(f"  - Reads: {sum(1 for op in wb.operations if op['action'] == 'read')}")
    
    await wb.close()


async def workflow_example():
    """Demonstrate a typical multi-agent workflow."""
    
    print("\n" + "=" * 60)
    print("Example 5: Multi-Agent Workflow")
    print("=" * 60)
    
    wb = Whiteboard()
    
    # Phase 1: Coordinator posts goal
    await wb.post(
        sender="Coordinator",
        channel="project:research",
        content={
            "type": "goal",
            "title": "Research renewable energy trends",
            "deadline": "2024-03-01"
        }
    )
    
    # Phase 2: Workers read goal and post updates
    goal_msg = (await wb.read(channel="project:research"))[0]
    
    await wb.post(
        sender="Researcher",
        channel="project:research",
        content={
            "type": "progress",
            "task": "Gather market data",
            "status": "in_progress"
        },
        thread="data-collection"
    )
    
    await wb.post(
        sender="Analyst",
        channel="project:research",
        content={
            "type": "progress",
            "task": "Analyze trends",
            "status": "pending"
        },
        thread="analysis"
    )
    
    # Phase 3: Workers complete tasks and post results
    await wb.post(
        sender="Researcher",
        channel="project:research",
        content={
            "type": "result",
            "task": "Gather market data",
            "findings": ["Solar up 25%", "Wind up 18%"],
            "status": "completed"
        },
        thread="data-collection"
    )
    
    # Phase 4: Coordinator aggregates results
    all_messages = await wb.read(channel="project:research")
    results = [m for m in all_messages if m.content.get("type") == "result"]
    
    await wb.post(
        sender="Coordinator",
        channel="project:research",
        content={
            "type": "summary",
            "completed_tasks": len(results),
            "overall_status": "on_track"
        }
    )
    
    # Show final state
    final_messages = await wb.read(channel="project:research")
    print(f"✓ Workflow completed with {len(final_messages)} messages")
    
    message_types = {}
    for m in final_messages:
        t = m.content.get("type", "unknown")
        message_types[t] = message_types.get(t, 0) + 1
    
    print(f"  - Message types: {message_types}")
    
    await wb.close()


async def main():
    """Run all examples."""
    
    print("=" * 60)
    print("Whiteboard System Examples")
    print("=" * 60)
    print("""
The new whiteboard system provides channel-based messaging for
multi-agent coordination. Unlike the old topic-based system with
predefined fields (mission, progress, blockers), the new system
uses flexible channels and JSON messages.

Benefits:
- Simple, flexible message format
- No imposed business logic
- Storage agnostic (in-memory or Redis)
- Extensible via subclassing
- Optional - works without modifying agent code
    """)
    
    await basic_whiteboard_example()
    await redis_whiteboard_example()
    await agent_system_with_whiteboard()
    await custom_whiteboard_extension()
    await workflow_example()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key APIs:

1. Creating a whiteboard:
   wb = Whiteboard()  # In-memory
   wb = Whiteboard(storage=RedisStorage(redis_client))  # Persistent

2. Posting messages:
   await wb.post(
       sender="Agent",
       channel="project:abc",
       content={"any": "json", "data": True},
       thread="optional-thread",
       reply_to="optional-parent-msg-id"
   )

3. Reading messages:
   messages = await wb.read(
       channel="project:abc",
       since=datetime_obj,
       limit=100,
       thread="optional-filter"
   )

4. Integration with agent system:
   context = AgentSystemContext.create(
       memory_bank=AsyncAgentMemoryBank(),
       enable_whiteboard=True,
       whiteboard_storage=optional_storage
   )

5. Custom extensions:
   class MyWhiteboard(Whiteboard):
       async def _before_post(self, sender, channel, content):
           # Add validation or enrichment
           pass
       
       async def _after_read(self, channel, messages):
           # Add filtering or logging
           pass
    """)


if __name__ == "__main__":
    asyncio.run(main())
