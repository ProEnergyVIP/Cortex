from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

@dataclass
class Message:
    content: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SystemMessage(Message):
    pass

@dataclass
class UserMessage(Message):
    user_name: str = None

    def build_content(self):
        return f'[{self.user_name}] {self.content}' if self.user_name else self.content

@dataclass
class UserVisionMessage(UserMessage):
    image_urls: Optional[list[str]] = None   # list of image urls for vision models

@dataclass
class Function:
    name: str
    arguments: str | object

@dataclass
class ToolCalling:
    id: str
    type: str
    function: Function

@dataclass
class MessageUsage:
    '''Token usage information for a message'''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0

    def accumulate(self, other: 'MessageUsage') -> None:
        '''Add another MessageUsage's numbers to this one'''
        if other:
            self.prompt_tokens += other.prompt_tokens
            self.completion_tokens += other.completion_tokens
            self.cached_tokens += other.cached_tokens
            self.total_tokens += other.total_tokens

    def format(self) -> str:
        '''Return a formatted string of the usage information'''
        return (f"Token Usage Summary:\n"
                f"  Prompt tokens: {self.prompt_tokens:,}\n"
                f"  Completion tokens: {self.completion_tokens:,}\n"
                f"  Cached tokens: {self.cached_tokens:,}\n"
                f"  Total tokens: {self.total_tokens:,}\n")

@dataclass
class AgentUsage:
    '''Tracks usage across multiple models in an agent session'''
    model_usages: Dict[str, MessageUsage] = field(default_factory=dict)
    
    def add_usage(self, model_name: str, message_usage: MessageUsage) -> None:
        '''Add or update usage for a specific model'''
        if model_name in self.model_usages:
            self.model_usages[model_name].accumulate(message_usage)
        else:
            self.model_usages[model_name] = message_usage

    def merge(self, other: 'AgentUsage') -> None:
        '''Merge another AgentUsage object into this one'''
        if other:
            for model_name, usage in other.model_usages.items():
                self.add_usage(model_name, usage)

    def format(self) -> str:
        '''Return a formatted string of all model usages'''
        if not self.model_usages:
            return "No usage data available\n"
            
        result = "Agent Usage Summary:\n"
        for model_name, usage in self.model_usages.items():
            result += f"\nModel: {model_name}\n{usage.format()}"
        return result

@dataclass
class AIMessage(Message):
    '''Message from an AI model'''
    tool_calls: List[ToolCalling] = None
    usage: Optional[MessageUsage] = None
    model: Optional[str] = None  # Name of the model that generated this message

@dataclass
class ToolMessage(Message):
    tool_call_id: str = ''

# message to group several tool calls' result together
@dataclass
class ToolMessageGroup:
    tool_messages: list[ToolMessage] = field(default_factory=list)

def print_message(msg):
    '''print the message to the console, use different colors for different roles.
    '''
    from rich import print
    
    if isinstance(msg, SystemMessage):
        print(f'[bold blue]{msg.content}[/bold blue]')
    elif isinstance(msg, UserMessage):
        content = msg.build_content()
        print(f'[bold green]{content}[/bold green]')
    elif isinstance(msg, AIMessage):
        print(f'[bold yellow]{msg.content}[/bold yellow]')
    elif isinstance(msg, ToolMessage):
        print(f'[bold purple]{msg.content}[/bold purple]')
    elif isinstance(msg, ToolMessageGroup):
        for tm in msg.tool_messages:
            print(f'[bold purple]{tm.content}[/bold purple]')
    else:
        print(msg.content)