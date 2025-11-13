"""
Example demonstrating the shared context capabilities of AgentSystemContext.

This shows how agents can coordinate through shared context fields:
- mission, current_focus, progress
- team_roles, protocols, updates
- artifacts, active_blockers
"""

import asyncio
from datetime import datetime, timedelta
from cortex import (
    LLM,
    GPTModels,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    CoordinatorSystem,
    AgentSystemContext,
    ContextUpdate,
    AsyncAgentMemoryBank,
    Tool,
)


# Example 1: Basic Shared Context Usage
async def basic_shared_context_example():
    """Demonstrate basic shared context fields."""
    
    # Create context with shared mission and roles
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(
        memory_bank=memory_bank,
        mission="Analyze customer feedback and generate insights",
        current_focus="Processing Q4 2024 feedback",
        team_roles={
            "data_analyst": "Data Analysis Expert",
            "report_writer": "Report Generation Specialist"
        },
        protocols=[
            "Always cite data sources",
            "Flag any anomalies immediately",
            "Update progress after each major step"
        ]
    )
    
    # Workers can access shared context
    def analyst_prompt_builder(ctx: AgentSystemContext):
        return f"""You are a {ctx.team_roles.get('data_analyst', 'Data Analyst')}.
        
Mission: {ctx.mission}
Current Focus: {ctx.current_focus}

Protocols:
{chr(10).join(f"- {p}" for p in ctx.protocols)}

Analyze data and update the shared context with your findings."""
    
    # Tool to update shared context
    async def log_finding_func(args, ctx: AgentSystemContext):
        """Log a finding to shared context."""
        finding = args.get("finding", "")
        ctx.add_update(
            agent_name="data_analyst",
            update_type="finding",
            content=finding,
            tags=["analysis", "q4"]
        )
        return f"Finding logged: {finding}"
    
    log_finding_tool = Tool(
        name="log_finding",
        func=log_finding_func,
        description="Log an important finding to the shared context",
        parameters={
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "The finding to log"},
            },
            "required": ["finding"],
        },
    )
    
    def analyst_tools_builder(ctx):
        return [log_finding_tool]
    
    analyst_worker = WorkerAgentBuilder(
        name="Data Analyst",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=analyst_prompt_builder,
        tools_builder=analyst_tools_builder,
        introduction="Analyzes data and logs findings",
    )
    
    def coordinator_prompt_builder(ctx):
        return "You coordinate the analysis team."
    
    coordinator = CoordinatorAgentBuilder(
        name="Coordinator",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=coordinator_prompt_builder,
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator,
        workers=[analyst_worker],
        context=context,
    )
    
    # Make a request
    response = await system.async_ask("Analyze the customer satisfaction trends")
    print("Response:", response)
    
    # Check shared context updates
    print("\n=== Shared Context Updates ===")
    print(f"Mission: {context.mission}")
    print(f"Progress: {context.progress}")
    print(f"Updates: {len(context.updates)}")
    for update in context.updates:
        print(f"  - [{update.update_type}] {update.agent_name}: {update.content}")


# Example 2: Agent View and Recent Updates
async def agent_view_example():
    """Demonstrate get_agent_view and get_recent_updates methods."""
    
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(
        memory_bank=memory_bank,
        mission="Build a recommendation system",
        current_focus="Feature engineering phase",
        progress="Completed data collection, starting feature extraction",
        team_roles={
            "ml_engineer": "ML Engineer",
            "data_engineer": "Data Engineer"
        }
    )
    
    # Simulate some updates
    context.add_update(
        agent_name="data_engineer",
        update_type="status",
        content="Data pipeline deployed successfully",
        tags=["infrastructure", "completed"]
    )
    
    context.add_update(
        agent_name="ml_engineer",
        update_type="decision",
        content="Choosing gradient boosting for initial model",
        tags=["modeling", "decision"]
    )
    
    context.add_update(
        agent_name="data_engineer",
        update_type="blocker",
        content="API rate limit reached, need to implement backoff",
        tags=["infrastructure", "blocker"]
    )
    
    # Get agent-specific view
    print("=== ML Engineer's View ===")
    ml_view = context.get_agent_view("ml_engineer")
    print(f"My Role: {ml_view['my_role']}")
    print(f"Mission: {ml_view['mission']}")
    print(f"Progress: {ml_view['progress']}")
    print(f"\nRecent Updates:")
    for update in ml_view['recent_updates']:
        print(f"  - [{update['update_type']}] {update['agent_name']}: {update['content']}")
    
    # Get filtered updates
    print("\n=== Recent Status Updates ===")
    status_updates = context.get_recent_updates(update_type="status")
    for update in status_updates:
        print(f"  - {update.agent_name}: {update.content}")
    
    # Get updates since a specific time
    print("\n=== Updates in Last Hour ===")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent = context.get_recent_updates(since=one_hour_ago)
    print(f"Found {len(recent)} updates in the last hour")


# Example 3: Artifacts and Blockers
async def artifacts_and_blockers_example():
    """Demonstrate artifacts and active_blockers fields."""
    
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(
        memory_bank=memory_bank,
        mission="Deploy new API version",
        artifacts={
            "code": [
                {"name": "api_v2.py", "status": "completed", "url": "/repo/api_v2.py"},
                {"name": "tests.py", "status": "in_progress", "url": "/repo/tests.py"}
            ],
            "docs": [
                {"name": "API_SPEC.md", "status": "completed", "url": "/docs/API_SPEC.md"}
            ]
        },
        active_blockers=[
            "Waiting for security review approval",
            "Database migration script needs testing"
        ]
    )
    
    print("=== Project Artifacts ===")
    for artifact_type, items in context.artifacts.items():
        print(f"\n{artifact_type.upper()}:")
        for item in items:
            print(f"  - {item['name']} ({item['status']})")
    
    print("\n=== Active Blockers ===")
    for blocker in context.active_blockers:
        print(f"  - {blocker}")
    
    # Update blockers
    context.active_blockers.remove("Waiting for security review approval")
    context.add_update(
        agent_name="devops",
        update_type="status",
        content="Security review approved, blocker removed",
        tags=["security", "unblocked"]
    )
    
    print("\n=== Updated Blockers ===")
    for blocker in context.active_blockers:
        print(f"  - {blocker}")


# Example 4: Backward Compatibility Test
async def backward_compatibility_example():
    """Verify that existing code still works without using shared context."""
    
    # Create context the old way (without shared context fields)
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)
    
    # Old-style worker definition
    def simple_prompt_builder(ctx):
        return "You are a helpful assistant."
    
    simple_worker = WorkerAgentBuilder(
        name="Assistant",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=simple_prompt_builder,
        introduction="General purpose assistant",
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
        workers=[simple_worker],
        context=context,
    )
    
    # This should work exactly as before
    response = await system.async_ask("Hello, how are you?")
    print("Response:", response)
    
    # Verify shared context fields have default values
    print("\n=== Default Shared Context Values ===")
    print(f"Mission: '{context.mission}' (empty string)")
    print(f"Updates: {len(context.updates)} (empty list)")
    print(f"Team Roles: {context.team_roles} (empty dict)")
    print(f"Protocols: {context.protocols} (empty list)")
    print("✓ Backward compatibility maintained!")


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Basic Shared Context Usage")
    print("=" * 80)
    # asyncio.run(basic_shared_context_example())
    
    print("\n" + "=" * 80)
    print("Example 2: Agent View and Recent Updates")
    print("=" * 80)
    asyncio.run(agent_view_example())
    
    print("\n" + "=" * 80)
    print("Example 3: Artifacts and Blockers")
    print("=" * 80)
    asyncio.run(artifacts_and_blockers_example())
    
    print("\n" + "=" * 80)
    print("Example 4: Backward Compatibility Test")
    print("=" * 80)
    # asyncio.run(backward_compatibility_example())
    
    print("\nUncomment the examples you want to run!")
