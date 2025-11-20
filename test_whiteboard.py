"""
Quick tests to verify Whiteboard functionality with the new design.
"""

from datetime import datetime, timedelta
from cortex.agent_system import (
    AgentSystemContext,
    Whiteboard,
    WhiteboardUpdate,
    WhiteboardUpdateType,
)
from cortex import AsyncAgentMemoryBank


def test_optional_whiteboard_absent_by_default():
    """AgentSystemContext has no whiteboard unless provided."""
    memory_bank = AsyncAgentMemoryBank()
    context = AgentSystemContext(memory_bank=memory_bank)
    assert context.whiteboard is None


def test_whiteboard_fields():
    """Test Whiteboard topic fields via Whiteboard."""
    memory_bank = AsyncAgentMemoryBank()
    wb = Whiteboard(
        team_roles={"agent1": "Tester", "agent2": "Developer"},
        protocols=["Write tests", "Document changes"],
    )
    wb.set_mission_focus(mission="Test mission", focus="Testing phase")
    wb.update_progress(progress="50% complete")
    wb.add_blocker(blocker="Waiting for review")
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=wb)

    st = context.whiteboard._state()
    assert st.mission == "Test mission"
    assert st.current_focus == "Testing phase"
    assert st.progress == "50% complete"
    assert len(context.whiteboard.team_roles) == 2
    assert len(context.whiteboard.protocols) == 2
    assert len(st.active_blockers) == 1


def test_add_update():
    """Test Whiteboard.add_update method."""
    memory_bank = AsyncAgentMemoryBank()
    wb = Whiteboard()
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=wb)

    # Add an update
    context.whiteboard.add_update(
        agent_name="test_agent",
        update_type=WhiteboardUpdateType.PROGRESS,
        content={"text": "Test update"},
        tags=["test", "demo"],
    )

    st = context.whiteboard._state()
    assert len(st.updates) == 1
    assert st.updates[0].agent_name == "test_agent"
    assert st.updates[0].type == WhiteboardUpdateType.PROGRESS
    assert st.updates[0].content == {"text": "Test update"}
    assert "test" in st.updates[0].tags
    assert isinstance(st.updates[0].timestamp, datetime)

    # Add another update
    context.whiteboard.add_update(
        agent_name="another_agent",
        update_type=WhiteboardUpdateType.FINDING,
        content={"finding": "Important finding"},
    )

    assert len(st.updates) == 2


def test_get_agent_view():
    """Test Whiteboard.get_agent_view method."""
    memory_bank = AsyncAgentMemoryBank()
    wb = Whiteboard(
        team_roles={"agent1": "Role1", "agent2": "Role2"},
        protocols=["Protocol 1"],
    )
    wb.set_mission_focus(mission="Test mission", focus="Testing")
    wb.update_progress(progress="In progress")
    st = wb._state()
    st.artifacts = {"code": [{"name": "test.py"}]}
    st.active_blockers = ["Blocker 1"]
    wb.add_update(agent_name="agent1", update_type=WhiteboardUpdateType.PROGRESS, content={"t": 1})
    wb.add_update(agent_name="agent2", update_type=WhiteboardUpdateType.FINDING, content={"t": 2})
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=wb)

    # Get agent view
    view = context.whiteboard.get_agent_view("agent1")
    assert view["mission"] == "Test mission"
    assert view["current_focus"] == "Testing"
    assert view["progress"] == "In progress"
    assert view["my_role"] == "Role1"
    assert "agent1" in view["team_roles"]
    assert len(view["protocols"]) == 1
    assert len(view["recent_updates"]) == 2
    assert "code" in view["artifacts"]
    assert len(view["active_blockers"]) == 1
    assert "last_activity" in view


def test_get_recent_updates():
    """Test Whiteboard.get_recent_updates method."""
    memory_bank = AsyncAgentMemoryBank()
    wb = Whiteboard()
    wb.add_update(agent_name="agent1", update_type=WhiteboardUpdateType.PROGRESS, content={"text": "Status 1"}, tags=["tag1"])
    wb.add_update(agent_name="agent2", update_type=WhiteboardUpdateType.FINDING, content={"text": "Finding 1"}, tags=["tag2"])
    wb.add_update(agent_name="agent1", update_type=WhiteboardUpdateType.PROGRESS, content={"text": "Status 2"}, tags=["tag1"])
    wb.add_update(agent_name="agent3", update_type=WhiteboardUpdateType.DECISION, content={"text": "Decision 1"}, tags=["tag3"])
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=wb)

    # Test: get all updates
    all_updates = context.whiteboard.get_recent_updates()
    assert len(all_updates) == 4

    # Test: filter by agent_name
    agent1_updates = context.whiteboard.get_recent_updates(agent_name="agent1")
    assert len(agent1_updates) == 2
    assert all(u.agent_name == "agent1" for u in agent1_updates)

    # Test: filter by update_type
    status_updates = context.whiteboard.get_recent_updates(update_type=WhiteboardUpdateType.PROGRESS)
    assert len(status_updates) == 2
    assert all(u.type == WhiteboardUpdateType.PROGRESS for u in status_updates)

    # Test: filter by time (all updates should be recent)
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_updates = context.whiteboard.get_recent_updates(since=one_hour_ago)
    assert len(recent_updates) == 4

    # Test: combined filters
    agent1_status = context.whiteboard.get_recent_updates(
        agent_name="agent1", update_type=WhiteboardUpdateType.PROGRESS
    )
    assert len(agent1_status) == 2


def test_whiteboard_update_model():
    """Test WhiteboardUpdate model."""
    update = WhiteboardUpdate(
        agent_name="test_agent",
        type=WhiteboardUpdateType.FINDING,
        content={"text": "Test content"},
        tags=["tag1", "tag2"],
    )
    assert update.agent_name == "test_agent"
    assert update.type == WhiteboardUpdateType.FINDING
    assert update.content == {"text": "Test content"}
    assert len(update.tags) == 2
    assert isinstance(update.timestamp, datetime)


def test_pydantic_validation():
    """Test that Pydantic validation works correctly on Whiteboard."""
    memory_bank = AsyncAgentMemoryBank()
    wb = Whiteboard(team_roles={"agent1": "Role1"})
    wb.set_mission_focus(mission="Valid mission", focus=None)
    context = AgentSystemContext(memory_bank=memory_bank, whiteboard=wb)
    st = context.whiteboard._state()
    assert st.mission == "Valid mission"

    # Modify fields
    st.mission = "Updated mission"
    assert st.mission == "Updated mission"

    context.whiteboard.team_roles["agent2"] = "Role2"
    assert len(context.whiteboard.team_roles) == 2


if __name__ == "__main__":
    print("=" * 60)
    print("Running Whiteboard Tests")
    print("=" * 60)
    
    try:
        test_optional_whiteboard_absent_by_default()
        test_whiteboard_fields()
        test_add_update()
        test_get_agent_view()
        test_get_recent_updates()
        test_whiteboard_update_model()
        test_pydantic_validation()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nWhiteboard functionality is working correctly.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
