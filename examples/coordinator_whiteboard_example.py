"""
Example demonstrating coordinator-managed Whiteboard in a multi-agent system.

This example shows:
1. Coordinator setting mission and focus at task start
2. Workers automatically receiving the Whiteboard
3. Workers logging their findings to the Whiteboard
4. Coordinator tracking progress and managing blockers
5. Full team coordination through the Whiteboard
"""

import asyncio
from cortex import LLM, GPTModels
from cortex.agent_memory import AsyncAgentMemoryBank
from cortex.agent_system import (
    AgentSystemContext,
    CoordinatorSystem,
    CoordinatorAgentBuilder,
    WorkerAgentBuilder,
    Whiteboard,
    WhiteboardUpdateType,
)


# Example: Data Analysis Team
async def data_analysis_team_example():
    """
    Demonstrate a data analysis team coordinating through the Whiteboard.
    
    Team structure:
    - Coordinator: Manages overall workflow and context
    - Data Engineer: Handles data infrastructure
    - Data Analyst: Performs analysis
    - ML Engineer: Builds models
    """
    
    # Initialize the Whiteboard with team roles
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard(
        team_roles={
            "Data Engineer": "Infrastructure & Data Pipeline",
            "Data Analyst": "Analysis & Insights",
            "ML Engineer": "Model Development",
        }
    )
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    # Define worker agents
    data_engineer_builder = WorkerAgentBuilder(
        name="Data Engineer",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: """
        You are a data engineer responsible for:
        - Setting up data pipelines
        - Managing data infrastructure
        - Ensuring data quality and availability
        
        When you complete tasks, be specific about what you've done.
        """,
        thinking=True,
        introduction="Data Engineer agent - handles data infrastructure and pipelines"
    )
    
    data_analyst_builder = WorkerAgentBuilder(
        name="Data Analyst",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: """
        You are a data analyst responsible for:
        - Analyzing data patterns and trends
        - Creating insights from data
        - Identifying key metrics and KPIs
        
        Provide clear, actionable insights based on data.
        """,
        thinking=True,
        introduction="Data Analyst agent - performs data analysis and generates insights"
    )
    
    ml_engineer_builder = WorkerAgentBuilder(
        name="ML Engineer",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: """
        You are an ML engineer responsible for:
        - Building and training machine learning models
        - Model evaluation and optimization
        - Deploying models to production
        
        Make data-driven decisions about model architecture and approach.
        """,
        thinking=True,
        introduction="ML Engineer agent - builds and deploys machine learning models"
    )
    
    # Define coordinator
    coordinator_builder = CoordinatorAgentBuilder(
        name="Team Lead",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: """
        You are the team lead coordinating a data science team.
        
        Your responsibilities:
        - Understand user requirements
        - Break down tasks and delegate to appropriate team members
        - Track team progress and manage blockers
        - Ensure team coordination through the Whiteboard
        
        Use your context management tools to:
        - Set mission at the start of new projects
        - Track progress as work completes
        - Manage blockers when issues arise
        - Log important decisions
        """
    )
    
    # Build the system
    system = CoordinatorSystem(
        coordinator_builder=coordinator_builder,
        workers=[data_engineer_builder, data_analyst_builder, ml_engineer_builder],
        context=context
    )
    
    print("=" * 80)
    print("EXAMPLE: Data Analysis Team with Whiteboard")
    print("=" * 80)
    
    # Task 1: Start a new project
    print("\n--- TASK 1: Build a customer churn prediction system ---\n")
    response = await system.async_ask(
        "We need to build a customer churn prediction system. "
        "Start by setting up the data pipeline and analyzing customer behavior patterns."
    )
    print(f"Coordinator Response:\n{response}\n")
    
    # Check Whiteboard after Task 1
    print("\n--- WHITEBOARD AFTER TASK 1 ---")
    wb = context.whiteboard
    state = wb._state() if wb else None
    if state:
        print(f"Mission: {state.mission}")
        print(f"Current Focus: {state.current_focus}")
        print(f"Progress: {state.progress}")
        print(f"Active Blockers: {state.active_blockers}")
        print(f"\nTotal Updates: {len(state.updates)}")
        print("\nRecent Updates:")
        for update in state.updates[-5:]:
            print(f"  - [{update.type}] {update.agent_name}: {str(update.content)[:100]}")
    
    # Task 2: Continue with model building
    print("\n\n--- TASK 2: Build the prediction model ---\n")
    response = await system.async_ask(
        "Based on the analysis, build a churn prediction model. "
        "Use the insights from the data analyst."
    )
    print(f"Coordinator Response:\n{response}\n")
    
    # Check Whiteboard after Task 2
    print("\n--- WHITEBOARD AFTER TASK 2 ---")
    if state:
        print(f"Progress: {state.progress}")
        print(f"Active Blockers: {state.active_blockers}")
        print(f"\nTotal Updates: {len(state.updates)}")
        print("\nRecent Updates:")
        for update in state.updates[-5:]:
            print(f"  - [{update.type}] {update.agent_name}: {str(update.content)[:100]}")
    
    # Task 3: Handle a blocker
    print("\n\n--- TASK 3: Report an issue ---\n")
    response = await system.async_ask(
        "The API for customer data is rate-limited. We need to implement backoff logic."
    )
    print(f"Coordinator Response:\n{response}\n")
    
    # Check blockers
    print("\n--- ACTIVE BLOCKERS ---")
    if state:
        for blocker in state.active_blockers:
            print(f"  - {blocker}")
    
    # Task 4: Get team status
    print("\n\n--- TASK 4: Check team status ---\n")
    response = await system.async_ask("What's the current status of the project?")
    print(f"Coordinator Response:\n{response}\n")
    
    # Final context summary
    print("\n" + "=" * 80)
    print("FINAL WHITEBOARD SUMMARY")
    print("=" * 80)
    if state:
        print(f"Mission: {state.mission}")
        print(f"Current Focus: {state.current_focus}")
        print(f"Progress: {state.progress}")
        print(f"Active Blockers: {len(state.active_blockers)}")
        print(f"Total Updates: {len(state.updates)}")
    print(f"Team Roles: {len(context.whiteboard.team_roles) if context.whiteboard else 0}")
    
    print("\n--- ALL UPDATES ---")
    if state:
        for i, update in enumerate(state.updates, 1):
            print(f"{i}. [{update.type}] {update.agent_name} @ {update.timestamp.strftime('%H:%M:%S')}")
            print(f"   Content: {str(update.content)[:150]}")
            print(f"   Tags: {', '.join(update.tags)}")
            print()


# Example: How workers receive context automatically
async def worker_context_awareness_example():
    """
    Demonstrate how workers automatically receive the Whiteboard.
    """
    
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard(
        team_roles={
            "Sales Analyst": "Revenue & Sales Metrics",
            "Marketing Analyst": "Campaign Performance",
        }
    )
    whiteboard.set_mission_focus(mission="Analyze Q4 sales performance", focus="Revenue analysis")
    whiteboard.update_progress(progress="Initial data collection complete")
    whiteboard.add_blocker(blocker="Waiting for Q4 data from finance team")
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    # Add some existing updates to show context sharing
    context.whiteboard.add_update(
        agent_name="Sales Analyst",
        update_type=WhiteboardUpdateType.FINDING,
        content={"finding": "Revenue increased 15% in Q4"},
        tags=["sales", "q4"],
    )
    
    context.whiteboard.add_update(
        agent_name="Marketing Analyst",
        update_type=WhiteboardUpdateType.FINDING,
        content={"finding": "Email campaign conversion rate: 8.5%"},
        tags=["marketing", "campaigns"],
    )
    
    # Create a worker
    analyst_builder = WorkerAgentBuilder(
        name="Product Analyst",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: """
        You are a product analyst responsible for analyzing product performance.
        
        When you receive a task, you'll automatically get:
        - Team mission and current focus
        - Your role in the team
        - Active blockers
        - Recent updates from other team members
        
        Use this context to align your work with the team's goals.
        """,
        thinking=True,
        introduction="Product Analyst - analyzes product metrics and user behavior"
    )
    
    coordinator_builder = CoordinatorAgentBuilder(
        name="Analytics Lead",
        llm=LLM(model=GPTModels.GPT_4O_MINI),
        prompt_builder=lambda ctx: "You coordinate the analytics team."
    )
    
    system = CoordinatorSystem(
        coordinator_builder=coordinator_builder,
        workers=[analyst_builder],
        context=context
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLE: Worker Context Awareness")
    print("=" * 80)
    
    print("\n--- INITIAL WHITEBOARD ---")
    wb2 = context.whiteboard
    st2 = wb2._state() if wb2 else None
    if st2:
        print(f"Mission: {st2.mission}")
        print(f"Current Focus: {st2.current_focus}")
        print(f"Progress: {st2.progress}")
        print(f"Active Blockers: {', '.join(st2.active_blockers)}")
        print("\nExisting Updates:")
        for update in st2.updates:
            print(f"  - [{update.type}] {update.agent_name}: {update.content}")
    
    print("\n--- TASK: Analyze product usage ---\n")
    response = await system.async_ask(
        "Analyze product usage patterns in Q4 and identify top features."
    )
    print(f"Response:\n{response}\n")
    
    print("\n--- WHITEBOARD AFTER WORKER EXECUTION ---")
    st3 = context.whiteboard._state()
    print(f"Total Updates: {len(st3.updates)}")
    print("\nLatest Update (from Product Analyst):")
    latest = st3.updates[-1]
    print(f"  Agent: {latest.agent_name}")
    print(f"  Type: {latest.type}")
    print(f"  Content: {latest.content}")
    print(f"  Tags: {latest.tags}")
    
    print("\n--- WHAT THE NEXT WORKER WOULD SEE ---")
    next_worker_view = context.whiteboard.get_agent_view("Marketing Analyst")
    print(f"Mission: {next_worker_view['mission']}")
    print(f"My Role: {next_worker_view['my_role']}")
    print(f"Active Blockers: {next_worker_view['active_blockers']}")
    print(f"Recent Updates: {len(next_worker_view['recent_updates'])}")
    for update in next_worker_view['recent_updates'][-3:]:
        print(f"  - [{update['type']}] {update['agent_name']}: {str(update['content'])[:80]}")


async def main():
    """Run all examples."""
    
    # Example 1: Full team coordination
    await data_analysis_team_example()
    
    # Example 2: Worker context awareness
    await worker_context_awareness_example()


if __name__ == "__main__":
    asyncio.run(main())
