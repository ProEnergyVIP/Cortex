"""
Example demonstrating how to create a custom AgentSystemContext by subclassing.

This shows how users can extend the context with their own properties and methods
to pass custom data and configuration to their agents.
"""

import asyncio
from functools import cached_property
from typing import Optional
from intellifun import (
    LLM,
    GPTModels,
    AgentSystemContext,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    CoordinatorSystem,
    AsyncAgentMemoryBank,
)


# Example 1: Custom Context with Application-Specific Data
class MyAppContext(AgentSystemContext):
    """Custom context that includes application-specific data."""
    
    # Add custom fields
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    app_config: Optional[dict] = None
    
    @cached_property
    def llm_fast(self) -> LLM:
        """Fast LLM for simple tasks."""
        return LLM(model=GPTModels.GPT_4O_MINI)
    
    @cached_property
    def llm_powerful(self) -> LLM:
        """Powerful LLM for complex reasoning."""
        return LLM(model=GPTModels.GPT_4O)
    
    def get_user_preferences(self) -> dict:
        """Get user preferences from app config."""
        if self.app_config:
            return self.app_config.get("user_preferences", {})
        return {}


async def custom_context_example():
    """Demonstrate using a custom context."""
    
    # Create custom context with app-specific data
    memory_bank = AsyncAgentMemoryBank()
    context = MyAppContext(
        memory_bank=memory_bank,
        user_id="user_123",
        session_id="session_456",
        app_config={
            "user_preferences": {
                "language": "en",
                "timezone": "UTC",
            }
        }
    )
    
    # Workers can access custom context properties
    def assistant_prompt_builder(ctx: MyAppContext):
        prefs = ctx.get_user_preferences()
        return f"""You are a helpful assistant.
        User preferences: {prefs}
        Session ID: {ctx.session_id}
        """
    
    assistant_worker = WorkerAgentBuilder(
        name="Assistant",
        llm=context.llm_fast,  # Use custom LLM property
        prompt_builder=assistant_prompt_builder,
        introduction="General purpose assistant",
    )
    
    def coordinator_prompt_builder(ctx: MyAppContext):
        return f"You coordinate tasks for user {ctx.user_id}"
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=context.llm_fast,
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[assistant_worker],
        context=context,
    )
    
    response = await system.async_ask("Hello!")
    print("Response:", response)


# Example 2: Context with Database Access
class DatabaseContext(AgentSystemContext):
    """Context that provides database access to agents."""
    
    db_connection: Optional[object] = None
    
    async def query_database(self, query: str):
        """Execute a database query."""
        # In a real app, this would use self.db_connection
        print(f"Executing query: {query}")
        return {"result": "mock data"}
    
    async def save_to_database(self, data: dict):
        """Save data to database."""
        print(f"Saving data: {data}")
        return {"status": "success"}


async def database_context_example():
    """Demonstrate context with database access."""
    
    memory_bank = AsyncAgentMemoryBank()
    context = DatabaseContext(
        memory_bank=memory_bank,
        db_connection=None,  # Would be a real DB connection
    )
    
    # Define a tool that uses the context's database methods
    from intellifun import Tool
    
    async def search_database_func(args, ctx: DatabaseContext):
        """Search the database."""
        query = args.get("query", "")
        result = await ctx.query_database(query)
        return result
    
    search_tool = Tool(
        name="search_database",
        func=search_database_func,
        description="Search the database",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    )
    
    def data_prompt_builder(ctx):
        return "You are a data analyst with database access."
    
    def data_tools_builder(ctx):
        return [search_tool]
    
    data_worker = WorkerAgentBuilder(
        name="Data Analyst",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=data_prompt_builder,
        tools_builder=data_tools_builder,
        introduction="Analyzes data from the database",
    )
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate data analysis tasks."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[data_worker],
        context=context,
    )
    
    response = await system.async_ask("Search for users created in the last week")
    print("Response:", response)


# Example 3: Context with Dynamic Configuration
class DynamicContext(AgentSystemContext):
    """Context with dynamic configuration that can change at runtime."""
    
    environment: str = "production"
    feature_flags: dict = {}
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.feature_flags.get(feature_name, False)
    
    @cached_property
    def llm_for_environment(self) -> LLM:
        """Return appropriate LLM based on environment."""
        if self.environment == "development":
            return LLM(model=GPTModels.GPT_4O_MINI)
        else:
            return LLM(model=GPTModels.GPT_4O)


async def dynamic_context_example():
    """Demonstrate context with dynamic configuration."""
    
    memory_bank = AsyncAgentMemoryBank()
    context = DynamicContext(
        memory_bank=memory_bank,
        environment="production",
        feature_flags={
            "advanced_reasoning": True,
            "web_search": False,
        }
    )
    
    def smart_prompt_builder(ctx: DynamicContext):
        prompt = "You are an intelligent assistant."
        if ctx.is_feature_enabled("advanced_reasoning"):
            prompt += "\nUse advanced reasoning for complex problems."
        return prompt
    
    smart_worker = WorkerAgentBuilder(
        name="Smart Assistant",
        llm=context.llm_for_environment,  # Automatically selects based on env
        prompt_builder=smart_prompt_builder,
        introduction="Intelligent assistant with configurable features",
    )
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate intelligent tasks."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=context.llm_for_environment,
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[smart_worker],
        context=context,
    )
    
    response = await system.async_ask("Solve this complex problem: ...")
    print("Response:", response)


# Example 4: Multiple Import Paths for AgentSystemContext
def show_import_paths():
    """Demonstrate different ways to import AgentSystemContext."""
    
    # Method 1: Import from main package
    from intellifun import AgentSystemContext as Context1
    
    # Method 2: Import from agent_system subpackage
    from intellifun.agent_system import AgentSystemContext as Context2
    
    # Method 3: Import from core module
    from intellifun.agent_system.core import AgentSystemContext as Context3
    
    # Method 4: Import from context module directly
    from intellifun.agent_system.core.context import AgentSystemContext as Context4
    
    # All are the same class
    assert Context1 is Context2 is Context3 is Context4
    print("✓ All import paths work and reference the same class!")
    
    # Users can subclass from any import path
    class CustomContext1(Context1):
        pass
    
    class CustomContext2(Context2):
        pass
    
    print("✓ Can subclass from any import path!")


if __name__ == "__main__":
    print("=" * 80)
    print("Demonstrating Import Paths")
    print("=" * 80)
    show_import_paths()
    
    print("\n" + "=" * 80)
    print("Example 1: Custom Context with App-Specific Data")
    print("=" * 80)
    # asyncio.run(custom_context_example())
    
    print("\n" + "=" * 80)
    print("Example 2: Context with Database Access")
    print("=" * 80)
    # asyncio.run(database_context_example())
    
    print("\n" + "=" * 80)
    print("Example 3: Context with Dynamic Configuration")
    print("=" * 80)
    # asyncio.run(dynamic_context_example())
    
    print("\nUncomment the examples you want to run!")
