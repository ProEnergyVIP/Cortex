#!/usr/bin/env python3
"""
Example script demonstrating how to use the LoggingConfig feature
to customize message printing behavior.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import cortex
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex.message import (
    SystemMessage, UserMessage, AIMessage, ToolMessage
)
from cortex.logging_config import (
    LoggingConfig, set_default_logging_config
)
from cortex.agent import Agent


logger = logging.getLogger(__name__)


# Example 1: Basic message printing (no logging config involved)
logger.debug("\n=== Example 1: Basic Message Printing ===\n")

# Create some example messages
system_msg = SystemMessage(content="I am a system message")
user_msg = UserMessage(content="hello, I'm a user message", user_name="Alice")
ai_msg = AIMessage(content="I'm an AI response message")
tool_msg = ToolMessage(content="I'm a tool message", tool_call_id="tool1")

# Print messages directly (no logging config)
logger.debug(system_msg.decorate())
logger.debug(user_msg.decorate())
logger.debug(ai_msg.decorate())
logger.debug(tool_msg.decorate())

# Example 2: Agent with default logging configuration
print("\n=== Example 2: Agent with Default Logging Configuration ===\n")

# Create mock LLM for demonstration (doesn't actually do anything)
class MockLLM:
    def call(self, *args, **kwargs):
        return AIMessage(content="This is a mock response")

# Create a mock memory for the agent
class MockMemory:
    def load_memory(self):
        return []
    
    def add_messages(self, messages):
        pass

# Create an agent with default logging config
default_agent = Agent(
    MockLLM(), 
    name="DefaultAgent",
    memory=MockMemory()
)

# The agent will use its default logging configuration when printing messages
# By default, system prompt is not printed
print("Running default_agent.ask():")
default_agent.ask("Hello DefaultAgent")

# Example 3: Agent with custom logging configuration
print("\n=== Example 3: Agent with Custom Logging Configuration ===\n")

# Create a custom logging configuration that shows system prompt
custom_config = LoggingConfig(
    print_system_prompt=True,  # Show system prompt
    print_messages=True,       # Show all messages
    print_usage_report=True    # Show usage report
)

# Create an agent with the custom configuration
verbose_agent = Agent(
    MockLLM(), 
    name="VerboseAgent",
    logging_config=custom_config,
    sys_prompt="I am a system prompt that will be shown",
    memory=MockMemory()
)

# This agent will print system prompt
print("Running verbose_agent.ask():")
verbose_agent.ask("Hello VerboseAgent")

# Example 4: Agent with minimal logging configuration
print("\n=== Example 4: Agent with Minimal Logging Configuration ===\n")

# Create a minimal logging configuration
minimal_config = LoggingConfig(
    print_system_prompt=False, # Hide system prompt
    print_messages=False,      # Hide all messages
    print_usage_report=False   # Hide usage report
)

# Create an agent with the minimal configuration
quiet_agent = Agent(
    MockLLM(), 
    name="QuietAgent",
    logging_config=minimal_config,
    memory=MockMemory()
)

# This agent will only print its name but no messages
print("Running quiet_agent.ask():")
quiet_agent.ask("Hello QuietAgent")

# Example 5: Agent that only shows AI responses
print("\n=== Example 5: Agent with Custom Message Handling ===\n")

# For this example, we need to use the old configuration style
# by modifying the agent's behavior directly
ai_only_agent = Agent(
    MockLLM(), 
    name="AIOnlyAgent",
    memory=MockMemory()
)

# Override the ask method to only print AI messages
original_ask = ai_only_agent.ask

def custom_ask(message, user_name=None, usage=None):
    # Save original config
    original_config = ai_only_agent.logging_config
    
    try:
        # Create a custom config just for this call
        from cortex.logging_config import LoggingConfig
        custom_config = LoggingConfig(
            print_system_prompt=False,
            print_messages=True,
            print_usage_report=False
        )
        
        # Set the custom config
        ai_only_agent.logging_config = custom_config
        
        # Call the original ask method
        result = original_ask(message, user_name, usage)
        
        return result
    finally:
        # Restore original config and print_message function
        ai_only_agent.logging_config = original_config

# Replace the ask method with our custom version
ai_only_agent.ask = custom_ask.__get__(ai_only_agent)

print("Running ai_only_agent.ask():")
ai_only_agent.ask("Hello AIOnlyAgent")

# Example 6: Setting global default logging configuration
print("\n=== Example 6: Setting Global Default Logging Configuration ===\n")

# Create a global logging configuration
global_config = LoggingConfig(
    print_system_prompt=True,  # Show system prompt globally
    print_messages=True,       # Show all messages
    print_usage_report=True    # Show usage report
)

# Set as the default global configuration
set_default_logging_config(global_config)

# Create an agent that will use the global config
global_agent = Agent(
    MockLLM(), 
    name="GlobalConfigAgent",
    memory=MockMemory(),
    sys_prompt="I am a system prompt that will be shown because of global config"
)

# This agent will use the global configuration
print("Running global_agent.ask():")
global_agent.ask("Hello GlobalConfigAgent")
