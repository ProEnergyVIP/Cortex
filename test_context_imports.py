#!/usr/bin/env python3
"""Test that AgentSystemContext can be imported from multiple paths and subclassed."""

import sys
sys.path.insert(0, '.')

# Test all import paths
from cortex.agent_system.core.context import AgentSystemContext as C1
from cortex.agent_system.core import AgentSystemContext as C2
from cortex.agent_system import AgentSystemContext as C3

# Verify they're all the same class
assert C1 is C2 is C3, "Import paths should reference the same class"
print("✓ All agent_system import paths work correctly!")

# Test subclassing
class MyCustomContext(C1):
    """Custom context for testing."""
    custom_field: str = "test"

# Verify subclass works
context = MyCustomContext(memory_bank=None)
assert hasattr(context, 'custom_field')
assert context.custom_field == "test"
print("✓ Can subclass AgentSystemContext successfully!")

print("\nAll tests passed! AgentSystemContext is properly exported.")
