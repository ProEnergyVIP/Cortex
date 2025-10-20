"""
Example demonstrating the Agent System - a higher-level API for building multi-agent systems.

The Agent System provides:
1. AgentBuilder - Base class for building agents with prompts, tools, and memory
2. AgentSystem - Base system for managing agent lifecycle
3. CoordinatorSystem - Concrete implementation with coordinator-worker pattern
4. CoordinatorAgentBuilder - Builder for coordinator agents that orchestrate workers
5. WorkerAgentBuilder - Builder for specialized worker agents
6. AgentSystemContext - Runtime context for agents (memory, usage tracking, etc.)
"""

import asyncio
from intellifun import (
    LLM,
    GPTModels,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    CoordinatorSystem,
    AgentSystemContext,
    AsyncAgentMemoryBank,
)


# Example 1: Simple Coordinator-Worker System
async def simple_coordinator_worker_example():
    """
    Demonstrates a basic coordinator-worker setup where:
    - A coordinator agent orchestrates the workflow
    - Worker agents handle specialized tasks
    """
    
    # Create context with memory bank
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)
    
    # Define a worker agent for math operations
    def math_prompt_builder(ctx):
        return """You are a math expert. You help users solve mathematical problems.
        Provide clear, step-by-step solutions."""
    
    math_worker = WorkerAgentBuilder(
        name="Math Expert",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=math_prompt_builder,
        introduction="Expert at solving mathematical problems and equations",
        thinking=True,
    )
    
    # Define a worker agent for writing tasks
    def writing_prompt_builder(ctx):
        return """You are a writing expert. You help users with writing tasks,
        including editing, proofreading, and creative writing."""
    
    writing_worker = WorkerAgentBuilder(
        name="Writing Expert",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=writing_prompt_builder,
        introduction="Expert at writing, editing, and proofreading",
        thinking=True,
    )
    
    # Define the coordinator agent
    def coordinator_prompt_builder(ctx):
        return """You coordinate between specialized worker agents.
        Analyze the user's request and delegate to the appropriate expert."""
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt_builder,
        thinking=True,
    )
    
    # Create the coordinator system
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[math_worker, writing_worker],
        context=context,
    )
    
    # Use the system
    response = await system.async_ask("What is the derivative of x^2 + 3x + 5?")
    print("Response:", response)
    
    # Follow-up question (uses memory)
    response = await system.async_ask("Can you explain that in simpler terms?")
    print("Follow-up Response:", response)


# Example 2: Worker with Custom Tools
async def worker_with_tools_example():
    """
    Demonstrates how to add custom tools to a worker agent.
    """
    from intellifun import Tool
    
    # Define a custom tool
    def calculate_area(args, context):
        """Calculate the area of a rectangle."""
        width = args.get("width", 0)
        height = args.get("height", 0)
        return {"area": width * height}
    
    area_tool = Tool(
        name="calculate_area",
        func=calculate_area,
        description="Calculate the area of a rectangle",
        parameters={
            "type": "object",
            "properties": {
                "width": {"type": "number", "description": "Width of rectangle"},
                "height": {"type": "number", "description": "Height of rectangle"},
            },
            "required": ["width", "height"],
        },
    )
    
    # Create a worker with tools
    def geometry_prompt_builder(ctx):
        return """You are a geometry expert with access to calculation tools.
        Use the tools when needed to provide accurate results."""
    
    def geometry_tools_builder(ctx):
        return [area_tool]
    
    geometry_worker = WorkerAgentBuilder(
        name="Geometry Expert",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=geometry_prompt_builder,
        tools_builder=geometry_tools_builder,
        introduction="Expert at geometry calculations",
        thinking=True,
    )
    
    # Create context and system
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate specialized workers."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[geometry_worker],
        context=context,
    )
    
    response = await system.async_ask("Calculate the area of a rectangle with width 5 and height 10")
    print("Response:", response)


# Example 3: Using AgentSystemContext with Usage Tracking
async def usage_tracking_example():
    """
    Demonstrates how to track usage across agent calls.
    """
    from intellifun.message import AgentUsage
    
    # Create context with usage tracking
    usage = AgentUsage()
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(usage=usage, memory_bank=memory_bank)
    
    # Create a simple worker
    def assistant_prompt_builder(ctx):
        return "You are a helpful assistant."
    
    assistant_worker = WorkerAgentBuilder(
        name="Assistant",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=assistant_prompt_builder,
    )
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate tasks."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[assistant_worker],
        context=context,
    )
    
    # Make multiple calls
    await system.async_ask("Hello, how are you?")
    await system.async_ask("What's the weather like?")
    
    # Check usage
    print(f"Total calls: {usage.call_count}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")


# Example 4: Custom Context with LLM Properties
async def custom_context_example():
    """
    Demonstrates using AgentSystemContext's built-in LLM properties.
    """
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)
    
    # Access pre-configured LLMs from context
    primary_llm = context.llm_primary  # GPT-5-MINI with minimal reasoning
    creative_llm = context.llm_creative  # GPT-5-MINI with medium reasoning
    
    # Use these LLMs in your builders
    def creative_prompt_builder(ctx):
        return "You are a creative storyteller."
    
    creative_worker = WorkerAgentBuilder(
        name="Storyteller",
        llm=creative_llm,  # Use the creative LLM
        prompt_builder=creative_prompt_builder,
    )
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate creative tasks."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=primary_llm,  # Use the primary LLM
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[creative_worker],
        context=context,
    )
    
    response = await system.async_ask("Tell me a short story about a robot")
    print("Response:", response)


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Simple Coordinator-Worker System")
    print("=" * 80)
    # asyncio.run(simple_coordinator_worker_example())
    
    print("\n" + "=" * 80)
    print("Example 2: Worker with Custom Tools")
    print("=" * 80)
    # asyncio.run(worker_with_tools_example())
    
    print("\n" + "=" * 80)
    print("Example 3: Usage Tracking")
    print("=" * 80)
    # asyncio.run(usage_tracking_example())
    
    print("\n" + "=" * 80)
    print("Example 4: Custom Context with LLM Properties")
    print("=" * 80)
    # asyncio.run(custom_context_example())
    
    print("\nUncomment the examples you want to run!")
