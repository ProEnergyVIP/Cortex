from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

@dataclass
class Message:
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def decorate(self) -> str:
        """Default decorated representation for a message."""
        return self.content

@dataclass
class SystemMessage(Message):
    def decorate(self) -> str:
        return f'[bold blue]{self.content}[/bold blue]'

@dataclass
class DeveloperMessage(Message):
    def decorate(self) -> str:
        return f'[bold red]{self.content}[/bold red]'

@dataclass
class UserMessage(Message):
    user_name: str = None
    # Optional structured inputs for images and files
    images: Optional[List['InputImage']] = None
    files: Optional[List['InputFile']] = None

    def build_content(self):
        return f'[{self.user_name}] {self.content}' if self.user_name else self.content
    
    def decorate(self) -> str:
        return f'[bold green]{self.build_content()}[/bold green]'

@dataclass
class UserVisionMessage(UserMessage):
    image_urls: Optional[list[str]] = None   # list of image urls for vision models


# Structured input items for Responses-style inputs
@dataclass
class InputImage:
    # The type of the input item. Always 'input_image'.
    type: str = 'input_image'
    # The detail level of the image: 'high' | 'low' | 'auto' (default 'auto')
    detail: str = 'auto'
    # Either a file_id or an image_url can be supplied
    file_id: Optional[str] = None
    image_url: Optional[str] = None


@dataclass
class InputFile:
    # The type of the input item. Always 'input_file'.
    type: str = 'input_file'
    # Provide file content directly, or via file_id or file_url
    file_data: Optional[str] = None
    file_id: Optional[str] = None
    file_url: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class FunctionCall:
    id: str
    type: str
    call_id: str
    name: str
    arguments: str | object
    status: Optional[str] = None

# deprecated ToolCalling and Function inside.
# Still used to support old messages encoded in it and stored in memory storage.
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
                f"  Cached tokens: {self.cached_tokens:,}\n"
                f"  Completion tokens: {self.completion_tokens:,}\n"
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
            result += f"Model: {model_name}\n{usage.format()}"
        return result

@dataclass
class AIMessage(Message):
    '''Message from an AI model'''
    model: Optional[str] = None  # Name of the model that generated this message
    function_calls: Optional[List[FunctionCall]] = None
    tool_calls: Optional[List[ToolCalling]] = None # deprecated, only for backward compatibility
    original_output: Optional[dict] = None
    usage: Optional[MessageUsage] = None
    
    def decorate(self) -> str:
        return f'[bold yellow]{self.content}[/bold yellow]'


@dataclass
class ToolMessage(Message):
    tool_call_id: str = ''
    
    def decorate(self) -> str:
        return f'[bold purple]{self.content}[/bold purple]'

# message to group several tool calls' result together
@dataclass
class ToolMessageGroup:
    tool_messages: list[ToolMessage] = field(default_factory=list)
    
    def decorate(self) -> str:
        return '\n'.join([tm.decorate() for tm in self.tool_messages])
