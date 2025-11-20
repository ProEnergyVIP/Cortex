"""
Example demonstrating the Whiteboard capabilities of AgentSystemContext.

This shows how agents can coordinate through Whiteboard fields:
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
    AsyncAgentMemoryBank,
    Tool,
    Whiteboard,
    WhiteboardUpdateType,
)


# Example 1: Basic Whiteboard Usage
async def basic_whiteboard_example():
    """Demonstrate basic Whiteboard fields."""
    
    # Create context with a Whiteboard, shared roles, and protocols
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard(
        team_roles={
            "data_analyst": "Data Analysis Expert",
            "report_writer": "Report Generation Specialist",
        },
        protocols=[
            "Always cite data sources",
            "Flag any anomalies immediately",
            "Update progress after each major step",
        ],
    )
    # Set mission/focus on the current topic
    whiteboard.set_mission_focus(
        mission="Analyze customer feedback and generate insights",
        focus="Processing Q4 2024 feedback",
    )
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    # Workers can access the Whiteboard
    def analyst_prompt_builder(ctx: AgentSystemContext):
        wb = ctx.whiteboard
        view = wb.get_agent_view("data_analyst") if wb else {}
        return f"""You are a {view.get('my_role', 'Data Analyst')}.

Mission: {view.get('mission', '')}
Current Focus: {view.get('current_focus', '')}

Protocols:
{chr(10).join(f"- {p}" for p in (wb.protocols if wb else []))}

Analyze data and update the Whiteboard with your findings."""
    
    # Tool to update the Whiteboard
    async def log_finding_func(args, ctx: AgentSystemContext):
        """Log a finding to the Whiteboard."""
        finding = args.get("finding", "")
        if ctx.whiteboard:
            ctx.whiteboard.add_update(
                agent_name="data_analyst",
                update_type=WhiteboardUpdateType.FINDING,
                content={"finding": finding},
                tags=["analysis", "q4"],
            )
        return f"Finding logged: {finding}"
    
    log_finding_tool = Tool(
        name="log_finding",
        func=log_finding_func,
        description="Log an important finding to the Whiteboard",
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
    
    # Check Whiteboard updates
    print("\n=== Whiteboard Updates ===")
    wb = context.whiteboard
    state = wb._state() if wb else None
    if state:
        print(f"Mission: {state.mission}")
        print(f"Progress: {state.progress}")
        print(f"Updates: {len(state.updates)}")
        for update in state.updates:
            print(f"  - [{update.type}] {update.agent_name}: {update.content}")


# Example 2: Agent View and Recent Updates
async def agent_view_example():
    """Demonstrate get_agent_view and get_recent_updates methods."""
    
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard(
        team_roles={
            "ml_engineer": "ML Engineer",
            "data_engineer": "Data Engineer",
        }
    )
    whiteboard.set_mission_focus(
        mission="Build a recommendation system",
        focus="Feature engineering phase",
    )
    whiteboard.update_progress(progress="Completed data collection, starting feature extraction")
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    # Simulate some updates
    context.whiteboard.add_update(
        agent_name="data_engineer",
        update_type=WhiteboardUpdateType.PROGRESS,
        content={"status": "Data pipeline deployed successfully"},
        tags=["infrastructure", "completed"],
    )
    
    context.whiteboard.add_update(
        agent_name="ml_engineer",
        update_type=WhiteboardUpdateType.DECISION,
        content={"decision": "Choosing gradient boosting for initial model"},
        tags=["modeling", "decision"],
    )
    
    context.whiteboard.add_update(
        agent_name="data_engineer",
        update_type=WhiteboardUpdateType.BLOCKER,
        content={"blocker": "API rate limit reached, need to implement backoff"},
        tags=["infrastructure", "blocker"],
    )
    
    # Get agent-specific view
    print("=== ML Engineer's View ===")
    ml_view = context.whiteboard.get_agent_view("ml_engineer")
    print(f"My Role: {ml_view['my_role']}")
    print(f"Mission: {ml_view['mission']}")
    print(f"Progress: {ml_view['progress']}")
    print("\nRecent Updates:")
    for update in ml_view['recent_updates']:
        print(f"  - [{update['update_type']}] {update['agent_name']}: {update['content']}")
    
    # Get filtered updates
    print("\n=== Recent Status Updates ===")
    status_updates = context.whiteboard.get_recent_updates(update_type=WhiteboardUpdateType.PROGRESS)
    for update in status_updates:
        print(f"  - {update.agent_name}: {update.content}")
    
    # Get updates since a specific time
    print("\n=== Updates in Last Hour ===")
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent = context.whiteboard.get_recent_updates(since=one_hour_ago)
    print(f"Found {len(recent)} updates in the last hour")


# Example 3: Artifacts and Blockers
async def artifacts_and_blockers_example():
    """Demonstrate artifacts and active_blockers fields."""
    
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard()
    whiteboard.set_mission_focus(mission="Deploy new API version", focus=None)
    # Seed artifacts and blockers on the current topic
    state = whiteboard._state()
    state.artifacts = {
        "code": [
            {"name": "api_v2.py", "status": "completed", "url": "/repo/api_v2.py"},
            {"name": "tests.py", "status": "in_progress", "url": "/repo/tests.py"},
        ],
        "docs": [
            {"name": "API_SPEC.md", "status": "completed", "url": "/docs/API_SPEC.md"},
        ],
    }
    whiteboard.add_blocker(blocker="Waiting for security review approval")
    whiteboard.add_blocker(blocker="Database migration script needs testing")
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    print("=== Project Artifacts ===")
    for artifact_type, items in state.artifacts.items():
        print(f"\n{artifact_type.upper()}:")
        for item in items:
            print(f"  - {item['name']} ({item['status']})")
    
    print("\n=== Active Blockers ===")
    for blocker in state.active_blockers:
        print(f"  - {blocker}")
    
    # Update blockers
    context.whiteboard.remove_blocker(blocker="Waiting for security review approval")
    context.whiteboard.add_update(
        agent_name="devops",
        update_type=WhiteboardUpdateType.PROGRESS,
        content={"status": "Security review approved, blocker removed"},
        tags=["security", "unblocked"],
    )
    
    print("\n=== Updated Blockers ===")
    for blocker in state.active_blockers:
        print(f"  - {blocker}")




if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Basic Whiteboard Usage")
    print("=" * 80)
    # asyncio.run(basic_whiteboard_example())
    
    print("\n" + "=" * 80)
    print("Example 2: Agent View and Recent Updates")
    print("=" * 80)
    asyncio.run(agent_view_example())
    
    print("\n" + "=" * 80)
    print("Example 3: Artifacts and Blockers")
    print("=" * 80)
    asyncio.run(artifacts_and_blockers_example())
    
    print("\nUncomment the examples you want to run!")
