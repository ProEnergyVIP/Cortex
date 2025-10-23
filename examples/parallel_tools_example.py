#!/usr/bin/env python3
"""
Example: Parallel Tool Execution

This example demonstrates how the Agent automatically executes multiple tool calls
in parallel for improved performance.

Key Features:
- Async tools run concurrently using asyncio.gather()
- Sync tools run in parallel using ThreadPoolExecutor
- Configurable parallelism (enable/disable, max concurrent)
- Automatic for multiple independent tool calls
"""

import asyncio
import time
from cortex import Agent
from cortex.tool import FunctionTool
from cortex.backends.anthropic import Claude

# ============================================================================
# Example 1: Async Tools with Concurrent Execution
# ============================================================================

async def fetch_user_data(user_id: str) -> dict:
    """Fetch user data from database (simulated with delay)"""
    await asyncio.sleep(1)  # Simulate database query
    return {
        "user_id": user_id,
        "name": "John Doe",
        "email": "john@example.com"
    }

async def fetch_user_orders(user_id: str) -> list:
    """Fetch user orders from database (simulated with delay)"""
    await asyncio.sleep(1)  # Simulate database query
    return [
        {"order_id": "001", "total": 99.99},
        {"order_id": "002", "total": 149.99}
    ]

async def fetch_user_preferences(user_id: str) -> dict:
    """Fetch user preferences from cache (simulated with delay)"""
    await asyncio.sleep(1)  # Simulate cache lookup
    return {
        "theme": "dark",
        "language": "en",
        "notifications": True
    }

async def example_async_parallel():
    """Example: Async agent with concurrent tool execution"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Async Agent - Concurrent Tool Execution")
    print("="*70)
    
    # Create async tools
    tools = [
        FunctionTool(
            name="fetch_user_data",
            func=fetch_user_data,
            description="Fetch user profile data",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        ),
        FunctionTool(
            name="fetch_user_orders",
            func=fetch_user_orders,
            description="Fetch user order history",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        ),
        FunctionTool(
            name="fetch_user_preferences",
            func=fetch_user_preferences,
            description="Fetch user preferences",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User ID"}
                },
                "required": ["user_id"]
            }
        ),
    ]
    
    # Create agent with parallel execution enabled (default)
    llm = Claude(model="claude-3-5-sonnet-20241022")
    agent = Agent(
        llm=llm,
        tools=tools,
        mode='async',
        enable_parallel_tools=True,  # Default: True
        sys_prompt="You are a helpful assistant. When asked for user information, "
                   "call all relevant tools to gather complete data efficiently."
    )
    
    print("\n📝 Task: Get complete user profile for user_123")
    print("💡 Agent will call all 3 tools concurrently (not sequentially)")
    
    start = time.time()
    
    # Note: In a real scenario, the LLM would decide to call multiple tools
    # For this example, we're demonstrating the capability
    print("\n⏱️  Expected time:")
    print("   - Sequential: ~3 seconds (1s + 1s + 1s)")
    print("   - Concurrent: ~1 second (all run at same time)")
    
    # Simulate the agent calling tools
    # In practice, this happens automatically when LLM returns multiple function_calls
    print("\n🚀 Running tools concurrently...")
    
    # This is what happens internally when LLM calls multiple tools:
    # The agent's async_process_func_call will use asyncio.gather()
    tasks = [
        fetch_user_data("user_123"),
        fetch_user_orders("user_123"),
        fetch_user_preferences("user_123")
    ]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\n✅ Completed in {elapsed:.2f}s (expected ~1s)")
    print(f"📊 Results: {len(results)} tools executed concurrently")
    print(f"⚡ Speedup: ~{3/elapsed:.1f}x faster than sequential")


# ============================================================================
# Example 2: Limiting Concurrent Execution
# ============================================================================

async def example_limited_concurrency():
    """Example: Limit concurrent tool execution (useful for rate limiting)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Limited Concurrent Execution")
    print("="*70)
    
    # Create agent with limited concurrency
    llm = Claude(model="claude-3-5-sonnet-20241022")
    agent = Agent(
        llm=llm,
        tools=[],  # Tools would be added here
        mode='async',
        enable_parallel_tools=True,
        max_parallel_tools=2,  # Limit to 2 concurrent tools
        sys_prompt="You are a helpful assistant."
    )
    
    print("\n⚙️  Configuration:")
    print(f"   - enable_parallel_tools: {agent.enable_parallel_tools}")
    print(f"   - max_parallel_tools: {agent.max_parallel_tools}")
    
    print("\n💡 Use Case: API rate limiting")
    print("   If you have 5 tools to call but API allows max 2 concurrent requests,")
    print("   set max_parallel_tools=2 to respect the limit.")
    
    print("\n⏱️  With 5 tools and max_parallel_tools=2:")
    print("   - First 2 tools run concurrently")
    print("   - Then next 2 tools run concurrently")
    print("   - Finally last tool runs")
    print("   - Total time: ~3 seconds (instead of 5 sequential or 1 unlimited)")


# ============================================================================
# Example 3: Sync Tools with Parallel Execution
# ============================================================================

def sync_database_query(query: str) -> dict:
    """Sync database query (simulated with delay)"""
    time.sleep(1)
    return {"result": f"Data for: {query}"}

def sync_api_call(endpoint: str) -> dict:
    """Sync API call (simulated with delay)"""
    time.sleep(1)
    return {"data": f"Response from: {endpoint}"}

def example_sync_parallel():
    """Example: Sync agent with parallel tool execution"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Sync Agent - Parallel Tool Execution")
    print("="*70)
    
    # Create sync tools
    tools = [
        FunctionTool(
            name="database_query",
            func=sync_database_query,
            description="Query the database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        ),
        FunctionTool(
            name="api_call",
            func=sync_api_call,
            description="Make API call",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"}
                },
                "required": ["endpoint"]
            }
        ),
    ]
    
    # Create agent with parallel execution
    llm = Claude(model="claude-3-5-sonnet-20241022")
    agent = Agent(
        llm=llm,
        tools=tools,
        mode='sync',
        enable_parallel_tools=True,
        sys_prompt="You are a helpful assistant."
    )
    
    print("\n💡 Sync mode uses ThreadPoolExecutor for parallel execution")
    print("   - Each tool runs in a separate thread")
    print("   - Good for I/O-bound operations (DB, API, files)")
    
    print("\n⏱️  With 2 tools:")
    print("   - Sequential: ~2 seconds")
    print("   - Parallel: ~1 second")


# ============================================================================
# Example 4: Disabling Parallel Execution
# ============================================================================

async def example_sequential_execution():
    """Example: Disable parallel execution (for debugging or dependencies)"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Sequential Execution (Parallel Disabled)")
    print("="*70)
    
    llm = Claude(model="claude-3-5-sonnet-20241022")
    agent = Agent(
        llm=llm,
        tools=[],
        mode='async',
        enable_parallel_tools=False,  # Disable parallel execution
        sys_prompt="You are a helpful assistant."
    )
    
    print("\n⚙️  Configuration:")
    print(f"   - enable_parallel_tools: {agent.enable_parallel_tools}")
    
    print("\n💡 Use Cases for Sequential Execution:")
    print("   1. Debugging - easier to trace execution")
    print("   2. Tool dependencies - Tool B needs result from Tool A")
    print("   3. Resource constraints - avoid overwhelming external services")
    print("   4. Deterministic testing - consistent execution order")


# ============================================================================
# Example 5: Real-World Scenario
# ============================================================================

async def check_inventory(product_id: str) -> dict:
    """Check product inventory"""
    await asyncio.sleep(0.5)
    return {"product_id": product_id, "in_stock": True, "quantity": 42}

async def check_pricing(product_id: str) -> dict:
    """Check product pricing"""
    await asyncio.sleep(0.5)
    return {"product_id": product_id, "price": 99.99, "currency": "USD"}

async def check_reviews(product_id: str) -> dict:
    """Check product reviews"""
    await asyncio.sleep(0.5)
    return {"product_id": product_id, "rating": 4.5, "count": 128}

async def check_shipping(product_id: str, zip_code: str) -> dict:
    """Check shipping options"""
    await asyncio.sleep(0.5)
    return {"available": True, "delivery_days": 3, "cost": 5.99}

async def example_real_world():
    """Example: E-commerce product page - fetch all data in parallel"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Real-World Scenario - E-commerce Product Page")
    print("="*70)
    
    tools = [
        FunctionTool(
            name="check_inventory",
            func=check_inventory,
            description="Check if product is in stock",
            parameters={"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}
        ),
        FunctionTool(
            name="check_pricing",
            func=check_pricing,
            description="Get product pricing",
            parameters={"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}
        ),
        FunctionTool(
            name="check_reviews",
            func=check_reviews,
            description="Get product reviews",
            parameters={"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]}
        ),
        FunctionTool(
            name="check_shipping",
            func=check_shipping,
            description="Check shipping options",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "zip_code": {"type": "string"}
                },
                "required": ["product_id", "zip_code"]
            }
        ),
    ]
    
    llm = Claude(model="claude-3-5-sonnet-20241022")
    agent = Agent(
        llm=llm,
        tools=tools,
        mode='async',
        enable_parallel_tools=True,
        sys_prompt="You are an e-commerce assistant. When showing product details, "
                   "fetch all relevant information (inventory, pricing, reviews, shipping) "
                   "in parallel for the best user experience."
    )
    
    print("\n📦 Scenario: User asks 'Show me details for product ABC123'")
    print("\n💡 Agent Strategy:")
    print("   - Identifies 4 independent data sources needed")
    print("   - Calls all 4 tools concurrently")
    print("   - Aggregates results and presents to user")
    
    print("\n⏱️  Performance:")
    print("   - Sequential: 4 × 0.5s = 2.0 seconds")
    print("   - Concurrent: max(0.5s) = 0.5 seconds")
    print("   - Speedup: 4x faster! ⚡")
    
    start = time.time()
    
    # Simulate concurrent execution
    tasks = [
        check_inventory("ABC123"),
        check_pricing("ABC123"),
        check_reviews("ABC123"),
        check_shipping("ABC123", "12345")
    ]
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    print(f"\n✅ Fetched all data in {elapsed:.2f}s")
    print(f"📊 Results: {len(results)} data sources")
    print("\n💬 Agent can now respond with complete product information instantly!")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "🚀 "*30)
    print("PARALLEL TOOL EXECUTION EXAMPLES")
    print("🚀 "*30)
    
    # Run examples
    await example_async_parallel()
    await example_limited_concurrency()
    example_sync_parallel()
    await example_sequential_execution()
    await example_real_world()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. ✅ Parallel execution is ENABLED BY DEFAULT for better performance
2. ✅ Works automatically when LLM calls multiple tools
3. ✅ Async mode uses asyncio.gather() for concurrency
4. ✅ Sync mode uses ThreadPoolExecutor for parallelism
5. ✅ Configure with enable_parallel_tools and max_parallel_tools
6. ✅ Typical speedup: 2-10x for I/O-bound operations
7. ✅ Order of results is always preserved
8. ✅ Each tool handles its own errors independently

💡 Best Practice: Design tools to be independent when possible,
   so the agent can execute them in parallel for maximum efficiency!
    """)

if __name__ == "__main__":
    asyncio.run(main())
