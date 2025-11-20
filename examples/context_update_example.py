"""
Example demonstrating the WhiteboardUpdate model with WhiteboardUpdateType enum.

Shows how to create updates with flexible content dictionaries and proper typing,
and how to use Whiteboard from AgentSystemContext.
"""

from cortex.agent_system import (
    AgentSystemContext,
    Whiteboard,
    WhiteboardUpdate,
    WhiteboardUpdateType,
)
from cortex import AsyncAgentMemoryBank


def example_whiteboard_update_creation():
    """Demonstrate creating WhiteboardUpdate objects with different types."""
    print("=" * 60)
    print("Creating WhiteboardUpdate Objects")
    print("=" * 60)
    
    # Progress update
    progress_update = WhiteboardUpdate(
        agent_name="data_processor",
        type=WhiteboardUpdateType.PROGRESS,
        content={
            "stage": "data_cleaning",
            "percentage": 75,
            "records_processed": 7500,
            "records_total": 10000
        },
        tags=["data", "processing"]
    )
    
    print("\n1. Progress Update:")
    print(f"   ID: {progress_update.id}")
    print(f"   Agent: {progress_update.agent_name}")
    print(f"   Type: {progress_update.type.value}")
    print(f"   Content: {progress_update.content}")
    print(f"   Tags: {progress_update.tags}")
    
    # Finding update
    finding_update = WhiteboardUpdate(
        agent_name="analyst",
        type=WhiteboardUpdateType.FINDING,
        content={
            "title": "Customer Satisfaction Trend",
            "summary": "15% increase in satisfaction scores",
            "confidence": 0.95,
            "data_source": "Q4 2024 surveys"
        },
        tags=["analysis", "positive", "customer"]
    )
    
    print("\n2. Finding Update:")
    print(f"   ID: {finding_update.id}")
    print(f"   Type: {finding_update.type.value}")
    print(f"   Finding: {finding_update.content['title']}")
    print(f"   Summary: {finding_update.content['summary']}")
    
    # Decision update
    decision_update = WhiteboardUpdate(
        agent_name="coordinator",
        type=WhiteboardUpdateType.DECISION,
        content={
            "decision": "Adopt gradient boosting model",
            "rationale": "Best performance on validation set",
            "alternatives_considered": ["random_forest", "neural_network"],
            "approval_required": False
        },
        tags=["modeling", "decision", "ml"]
    )
    
    print("\n3. Decision Update:")
    print(f"   ID: {decision_update.id}")
    print(f"   Decision: {decision_update.content['decision']}")
    print(f"   Rationale: {decision_update.content['rationale']}")
    
    # Artifact update
    artifact_update = WhiteboardUpdate(
        agent_name="developer",
        type=WhiteboardUpdateType.ARTIFACT,
        content={
            "artifact_type": "code",
            "name": "data_pipeline.py",
            "location": "/src/pipelines/data_pipeline.py",
            "status": "completed",
            "version": "1.0.0"
        },
        tags=["code", "pipeline", "completed"]
    )
    
    print("\n4. Artifact Update:")
    print(f"   ID: {artifact_update.id}")
    print(f"   Artifact: {artifact_update.content['name']}")
    print(f"   Status: {artifact_update.content['status']}")
    
    # Blocker update
    blocker_update = WhiteboardUpdate(
        agent_name="devops",
        type=WhiteboardUpdateType.BLOCKER,
        content={
            "blocker": "API rate limit exceeded",
            "severity": "high",
            "impact": "Data ingestion paused",
            "estimated_resolution": "2 hours",
            "workaround": "Implement exponential backoff"
        },
        tags=["blocker", "infrastructure", "urgent"]
    )
    
    print("\n5. Blocker Update:")
    print(f"   ID: {blocker_update.id}")
    print(f"   Blocker: {blocker_update.content['blocker']}")
    print(f"   Severity: {blocker_update.content['severity']}")
    print(f"   Impact: {blocker_update.content['impact']}")


def example_context_with_updates():
    """Demonstrate using updates via Whiteboard in AgentSystemContext."""
    print("\n" + "=" * 60)
    print("Using Updates in AgentSystemContext (Whiteboard)")
    print("=" * 60)
    
    memory_bank = AsyncAgentMemoryBank()
    whiteboard = Whiteboard(
        team_roles={
            "ml_engineer": "ML Engineer",
            "data_engineer": "Data Engineer",
        }
    )
    whiteboard.set_mission_focus(mission="Build recommendation system", focus=None)
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=whiteboard)
    
    # Add various updates
    context.whiteboard.add_update(
        agent_name="data_engineer",
        update_type=WhiteboardUpdateType.PROGRESS,
        content={
            "task": "data_collection",
            "status": "completed",
            "records": 50000
        },
        tags=["data", "completed"]
    )
    
    context.whiteboard.add_update(
        agent_name="ml_engineer",
        update_type=WhiteboardUpdateType.FINDING,
        content={
            "finding": "Feature importance analysis complete",
            "top_features": ["user_age", "purchase_history", "session_duration"]
        },
        tags=["ml", "features"]
    )
    
    context.whiteboard.add_update(
        agent_name="ml_engineer",
        update_type=WhiteboardUpdateType.DECISION,
        content={
            "decision": "Use collaborative filtering approach",
            "reason": "Better cold-start performance"
        },
        tags=["ml", "architecture"]
    )
    
    context.whiteboard.add_update(
        agent_name="data_engineer",
        update_type=WhiteboardUpdateType.BLOCKER,
        content={
            "blocker": "Database connection timeout",
            "severity": "medium"
        },
        tags=["infrastructure", "blocker"]
    )
    
    st = context.whiteboard._state()
    print(f"\nTotal updates: {len(st.updates)}")
    
    # Filter by type
    print("\n--- Progress Updates ---")
    progress_updates = context.whiteboard.get_recent_updates(update_type=WhiteboardUpdateType.PROGRESS)
    for update in progress_updates:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    print("\n--- Findings ---")
    findings = context.whiteboard.get_recent_updates(update_type=WhiteboardUpdateType.FINDING)
    for update in findings:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    print("\n--- Blockers ---")
    blockers = context.whiteboard.get_recent_updates(update_type=WhiteboardUpdateType.BLOCKER)
    for update in blockers:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    # Filter by agent
    print("\n--- ML Engineer Updates ---")
    ml_updates = context.whiteboard.get_recent_updates(agent_name="ml_engineer")
    for update in ml_updates:
        print(f"  [{update.type.value}] {update.content}")
    
    # Get agent view
    print("\n--- ML Engineer's View ---")
    view = context.whiteboard.get_agent_view("ml_engineer")
    print(f"My Role: {view['my_role']}")
    print(f"Mission: {view['mission']}")
    print(f"Recent Updates ({len(view['recent_updates'])}):")
    for update in view['recent_updates']:
        print(f"  [{update['type']}] {update['agent_name']}: {list(update['content'].keys())}")


def example_serialization():
    """Demonstrate that WhiteboardUpdate is serializable."""
    print("\n" + "=" * 60)
    print("Serialization Example")
    print("=" * 60)
    
    update = WhiteboardUpdate(
        agent_name="test_agent",
        type=WhiteboardUpdateType.FINDING,
        content={
            "key": "value",
            "number": 42,
            "nested": {"data": "here"}
        },
        tags=["test"]
    )
    
    # Pydantic model can be converted to dict
    update_dict = update.model_dump()
    print("\nSerialized to dict:")
    print(f"  {update_dict}")
    
    # Note: WhiteboardUpdateType enum is serialized as string value due to Config.use_enum_values
    print(f"\nType field serialized as: {update_dict['type']} (string)")
    
    # Can be converted to JSON
    update_json = update.model_dump_json()
    print("\nSerialized to JSON:")
    print(f"  {update_json}")
    
    # Can be reconstructed from dict
    reconstructed = WhiteboardUpdate(**update_dict)
    print("\nReconstructed from dict:")
    print(f"  ID: {reconstructed.id}")
    print(f"  Type: {reconstructed.type}")
    print(f"  Content: {reconstructed.content}")


def example_enum_usage():
    """Demonstrate WhiteboardUpdateType enum usage."""
    print("\n" + "=" * 60)
    print("WhiteboardUpdateType Enum Usage")
    print("=" * 60)
    
    print("\nAvailable update types:")
    for update_type in WhiteboardUpdateType:
        print(f"  - WhiteboardUpdateType.{update_type.name} = '{update_type.value}'")
    
    print("\nUsing enum in code:")
    print(f"  WhiteboardUpdateType.PROGRESS = {WhiteboardUpdateType.PROGRESS}")
    print(f"  WhiteboardUpdateType.FINDING = {WhiteboardUpdateType.FINDING}")
    print(f"  WhiteboardUpdateType.DECISION = {WhiteboardUpdateType.DECISION}")
    print(f"  WhiteboardUpdateType.ARTIFACT = {WhiteboardUpdateType.ARTIFACT}")
    print(f"  WhiteboardUpdateType.BLOCKER = {WhiteboardUpdateType.BLOCKER}")
    
    print("\nEnum comparison:")
    update = WhiteboardUpdate(
        agent_name="test",
        type=WhiteboardUpdateType.PROGRESS,
        content={"test": "data"}
    )
    
    if update.type == WhiteboardUpdateType.PROGRESS:
        print("  ✓ Update is a PROGRESS type")
    
    print(f"\nEnum value: {update.type.value}")
    print(f"Enum name: {update.type.name}")


if __name__ == "__main__":
    example_whiteboard_update_creation()
    example_context_with_updates()
    example_serialization()
    example_enum_usage()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
