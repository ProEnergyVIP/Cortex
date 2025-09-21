# Logging Configuration

The IntellifFun library provides a flexible logging configuration system that allows you to control which messages are printed during agent interactions.

## Overview

The logging system allows you to:

1. Control which types of messages are printed (system prompt, conversation messages, usage reports)
2. Configure logging globally or per-agent
3. Display the agent name before message interactions

## LoggingConfig Class

The `LoggingConfig` class is defined in `intellifun.logging_config` and has the following fields:

```python
@dataclass
class LoggingConfig:
    # Enable/disable different message types
    print_system_prompt: bool = False  # System prompts are hidden by default
    print_messages: bool = True        # Controls user, AI, and tool messages
    print_usage_report: bool = True
```

## Usage Examples

### Basic Agent with Default Logging Configuration

```python
from intellifun.agent import Agent

# Create an agent with default logging configuration
agent = Agent(llm, name="MyAgent")

# The agent will:
# - Print its name "MyAgent" at the start of the conversation
# - Hide system prompt
# - Show all conversation messages (user, AI, tool)
# - Show usage reports
agent.ask("Hello agent")
```

### Agent with Custom Logging Configuration

```python
from intellifun.agent import Agent
from intellifun.logging_config import LoggingConfig

# Create a custom logging configuration
custom_config = LoggingConfig(
    print_system_prompt=True,   # Show system prompt
    print_messages=True,        # Show all conversation messages
    print_usage_report=False    # Hide usage reports
)

# Create an agent with the custom configuration
agent = Agent(llm, name="CustomAgent", logging_config=custom_config)

# The agent will:
# - Print its name "CustomAgent" at the start of the conversation
# - Show system prompt and all conversation messages
# - Hide usage reports
agent.ask("Hello agent")
```

### Setting Global Default Logging Configuration

```python
from intellifun.logging_config import LoggingConfig, set_default_logging_config
from intellifun.agent import Agent

# Create a global logging configuration
global_config = LoggingConfig(
    print_system_prompt=True,  # Show system prompt globally
    print_messages=True,       # Show all conversation messages
    print_usage_report=True    # Show usage reports
)

# Set as the default global configuration
set_default_logging_config(global_config)

# Create an agent without specifying a logging_config
# It will use the global configuration
agent = Agent(llm, name="GlobalConfigAgent")

# This agent will use the global configuration
agent.ask("Hello agent")
```

## Advanced Usage: Custom Message Filtering

If you need more fine-grained control over which messages are printed, you can create a custom solution:

```python
from intellifun.agent import Agent

# Create an agent
agent = Agent(llm, name="CustomFilterAgent")

# Now the agent will only print AI messages
agent.ask("Hello agent")
```

## How It Works

1. The `Agent` class checks the logging configuration before printing each type of message.
2. If an agent has a name, it will be printed once at the beginning of the conversation.
3. Messages are only printed if their corresponding configuration flag is enabled.

## Best Practices

1. **Library Defaults**: By default, system prompts are hidden as they are typically long and not relevant to end users.

2. **Global Configuration**: Use `set_default_logging_config()` to set a global configuration that will be used by all agents that don't specify their own configuration.

3. **Agent-specific Configuration**: Each agent can have its own configuration, allowing different agents to have different logging behaviors.

4. **Debugging**: When debugging, enable all message types including system prompts to see the full context of the conversation.

5. **Performance**: For production use with many agents, consider disabling unnecessary message types to reduce console output.
