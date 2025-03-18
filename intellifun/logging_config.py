"""
Logging configuration module for the intellifun library.

This module provides a flexible logging configuration system that allows
controlling which messages are printed during agent interactions.
"""

from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration for controlling message logging behavior"""
    # Enable/disable different message types
    print_system_prompt: bool = False
    print_messages: bool = True      # Controls user, AI, and tool messages
    print_usage_report: bool = True
    
    @classmethod
    def create_default(cls) -> 'LoggingConfig':
        """Create the default logging configuration"""
        return cls(
            print_system_prompt=False,
            print_messages=True,
            print_usage_report=True
        )


# Global default logging configuration
DEFAULT_LOGGING_CONFIG = LoggingConfig.create_default()


def set_default_logging_config(config: LoggingConfig):
    """Set the default logging configuration globally
    
    Args:
        config: The logging configuration to set as default
    """
    global DEFAULT_LOGGING_CONFIG
    DEFAULT_LOGGING_CONFIG = config


def get_default_logging_config() -> LoggingConfig:
    """Get the current default logging configuration
    
    Returns:
        The current default logging configuration
    """
    return DEFAULT_LOGGING_CONFIG
