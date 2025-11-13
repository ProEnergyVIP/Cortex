"""
Example demonstrating the ContextUpdate model with UpdateType enum.

Shows how to create updates with flexible content dictionaries and proper typing.
"""

from datetime import datetime, timedelta
from cortex.agent_system import AgentSystemContext, ContextUpdate, UpdateType, AsyncAgentMemoryBank


def example_context_update_creation():
    """Demonstrate creating ContextUpdate objects with different types."""
    print("=" * 60)
    print("Creating ContextUpdate Objects")
    print("=" * 60)
    
    # Progress update
    progress_update = ContextUpdate(
        agent_name="data_processor",
        type=UpdateType.PROGRESS,
        content={
            "stage": "data_cleaning",
            "percentage": 75,
            "records_processed": 7500,
            "records_total": 10000
        },
        tags=["data", "processing"]
    )
    
    print(f"\n1. Progress Update:")
    print(f"   ID: {progress_update.id}")
    print(f"   Agent: {progress_update.agent_name}")
    print(f"   Type: {progress_update.type.value}")
    print(f"   Content: {progress_update.content}")
    print(f"   Tags: {progress_update.tags}")
    
    # Finding update
    finding_update = ContextUpdate(
        agent_name="analyst",
        type=UpdateType.FINDING,
        content={
            "title": "Customer Satisfaction Trend",
            "summary": "15% increase in satisfaction scores",
            "confidence": 0.95,
            "data_source": "Q4 2024 surveys"
        },
        tags=["analysis", "positive", "customer"]
    )
    
    print(f"\n2. Finding Update:")
    print(f"   ID: {finding_update.id}")
    print(f"   Type: {finding_update.type.value}")
    print(f"   Finding: {finding_update.content['title']}")
    print(f"   Summary: {finding_update.content['summary']}")
    
    # Decision update
    decision_update = ContextUpdate(
        agent_name="coordinator",
        type=UpdateType.DECISION,
        content={
            "decision": "Adopt gradient boosting model",
            "rationale": "Best performance on validation set",
            "alternatives_considered": ["random_forest", "neural_network"],
            "approval_required": False
        },
        tags=["modeling", "decision", "ml"]
    )
    
    print(f"\n3. Decision Update:")
    print(f"   ID: {decision_update.id}")
    print(f"   Decision: {decision_update.content['decision']}")
    print(f"   Rationale: {decision_update.content['rationale']}")
    
    # Artifact update
    artifact_update = ContextUpdate(
        agent_name="developer",
        type=UpdateType.ARTIFACT,
        content={
            "artifact_type": "code",
            "name": "data_pipeline.py",
            "location": "/src/pipelines/data_pipeline.py",
            "status": "completed",
            "version": "1.0.0"
        },
        tags=["code", "pipeline", "completed"]
    )
    
    print(f"\n4. Artifact Update:")
    print(f"   ID: {artifact_update.id}")
    print(f"   Artifact: {artifact_update.content['name']}")
    print(f"   Status: {artifact_update.content['status']}")
    
    # Blocker update
    blocker_update = ContextUpdate(
        agent_name="devops",
        type=UpdateType.BLOCKER,
        content={
            "blocker": "API rate limit exceeded",
            "severity": "high",
            "impact": "Data ingestion paused",
            "estimated_resolution": "2 hours",
            "workaround": "Implement exponential backoff"
        },
        tags=["blocker", "infrastructure", "urgent"]
    )
    
    print(f"\n5. Blocker Update:")
    print(f"   ID: {blocker_update.id}")
    print(f"   Blocker: {blocker_update.content['blocker']}")
    print(f"   Severity: {blocker_update.content['severity']}")
    print(f"   Impact: {blocker_update.content['impact']}")


def example_context_with_updates():
    """Demonstrate using updates in AgentSystemContext."""
    print("\n" + "=" * 60)
    print("Using Updates in AgentSystemContext")
    print("=" * 60)
    
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(
        memory_bank=memory_bank,
        mission="Build recommendation system",
        team_roles={
            "ml_engineer": "ML Engineer",
            "data_engineer": "Data Engineer"
        }
    )
    
    # Add various updates
    context.add_update(
        agent_name="data_engineer",
        update_type=UpdateType.PROGRESS,
        content={
            "task": "data_collection",
            "status": "completed",
            "records": 50000
        },
        tags=["data", "completed"]
    )
    
    context.add_update(
        agent_name="ml_engineer",
        update_type=UpdateType.FINDING,
        content={
            "finding": "Feature importance analysis complete",
            "top_features": ["user_age", "purchase_history", "session_duration"]
        },
        tags=["ml", "features"]
    )
    
    context.add_update(
        agent_name="ml_engineer",
        update_type=UpdateType.DECISION,
        content={
            "decision": "Use collaborative filtering approach",
            "reason": "Better cold-start performance"
        },
        tags=["ml", "architecture"]
    )
    
    context.add_update(
        agent_name="data_engineer",
        update_type=UpdateType.BLOCKER,
        content={
            "blocker": "Database connection timeout",
            "severity": "medium"
        },
        tags=["infrastructure", "blocker"]
    )
    
    print(f"\nTotal updates: {len(context.updates)}")
    
    # Filter by type
    print("\n--- Progress Updates ---")
    progress_updates = context.get_recent_updates(update_type=UpdateType.PROGRESS)
    for update in progress_updates:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    print("\n--- Findings ---")
    findings = context.get_recent_updates(update_type=UpdateType.FINDING)
    for update in findings:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    print("\n--- Blockers ---")
    blockers = context.get_recent_updates(update_type=UpdateType.BLOCKER)
    for update in blockers:
        print(f"  [{update.id}] {update.agent_name}: {update.content}")
    
    # Filter by agent
    print("\n--- ML Engineer Updates ---")
    ml_updates = context.get_recent_updates(agent_name="ml_engineer")
    for update in ml_updates:
        print(f"  [{update.type.value}] {update.content}")
    
    # Get agent view
    print("\n--- ML Engineer's View ---")
    view = context.get_agent_view("ml_engineer")
    print(f"My Role: {view['my_role']}")
    print(f"Mission: {view['mission']}")
    print(f"Recent Updates ({len(view['recent_updates'])}):")
    for update in view['recent_updates']:
        print(f"  [{update['type']}] {update['agent_name']}: {list(update['content'].keys())}")


def example_serialization():
    """Demonstrate that ContextUpdate is serializable."""
    print("\n" + "=" * 60)
    print("Serialization Example")
    print("=" * 60)
    
    update = ContextUpdate(
        agent_name="test_agent",
        type=UpdateType.FINDING,
        content={
            "key": "value",
            "number": 42,
            "nested": {"data": "here"}
        },
        tags=["test"]
    )
    
    # Pydantic model can be converted to dict
    update_dict = update.dict()
    print("\nSerialized to dict:")
    print(f"  {update_dict}")
    
    # Note: UpdateType enum is serialized as string value due to Config.use_enum_values
    print(f"\nType field serialized as: {update_dict['type']} (string)")
    
    # Can be converted to JSON
    import json
    update_json = update.json()
    print(f"\nSerialized to JSON:")
    print(f"  {update_json}")
    
    # Can be reconstructed from dict
    reconstructed = ContextUpdate(**update_dict)
    print(f"\nReconstructed from dict:")
    print(f"  ID: {reconstructed.id}")
    print(f"  Type: {reconstructed.type}")
    print(f"  Content: {reconstructed.content}")


def example_enum_usage():
    """Demonstrate UpdateType enum usage."""
    print("\n" + "=" * 60)
    print("UpdateType Enum Usage")
    print("=" * 60)
    
    print("\nAvailable update types:")
    for update_type in UpdateType:
        print(f"  - UpdateType.{update_type.name} = '{update_type.value}'")
    
    print("\nUsing enum in code:")
    print(f"  UpdateType.PROGRESS = {UpdateType.PROGRESS}")
    print(f"  UpdateType.FINDING = {UpdateType.FINDING}")
    print(f"  UpdateType.DECISION = {UpdateType.DECISION}")
    print(f"  UpdateType.ARTIFACT = {UpdateType.ARTIFACT}")
    print(f"  UpdateType.BLOCKER = {UpdateType.BLOCKER}")
    
    print("\nEnum comparison:")
    update = ContextUpdate(
        agent_name="test",
        type=UpdateType.PROGRESS,
        content={"test": "data"}
    )
    
    if update.type == UpdateType.PROGRESS:
        print("  ✓ Update is a PROGRESS type")
    
    print(f"\nEnum value: {update.type.value}")
    print(f"Enum name: {update.type.name}")


if __name__ == "__main__":
    example_context_update_creation()
    example_context_with_updates()
    example_serialization()
    example_enum_usage()
    
    print("\n" + "=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)
