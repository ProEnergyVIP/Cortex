"""
Logging configuration module for the intellifun library.

This module provides a flexible logging configuration system that allows
controlling which messages are printed during agent interactions.

Example usage:
    from intellifun.logging_config import LoggingConfig, set_default_logging_config

    # Create a custom configuration
    config = LoggingConfig(
        print_system_prompt=True,
        print_messages=True,
        print_usage_report=False
    )

    # Set as global default
    set_default_logging_config(config)
"""

from dataclasses import dataclass


@dataclass
class LoggingConfig:
    """Configuration for controlling message logging behavior

    Attributes:
        print_system_prompt (bool): Whether to print system prompts
        print_messages (bool): Whether to print user, AI, and tool messages
        print_usage_report (bool): Whether to print usage reports
    """
    print_system_prompt: bool = False
    print_messages: bool = True
    print_usage_report: bool = True

    @classmethod
    def create_default(cls) -> 'LoggingConfig':
        """Create the default logging configuration

        Returns:
            LoggingConfig: A new LoggingConfig instance with default values
        """
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
        LoggingConfig: The current default logging configuration
    """
    return DEFAULT_LOGGING_CONFIG
