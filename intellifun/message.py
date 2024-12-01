from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

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
    arguments: str

@dataclass
class ToolCalling:
    id: str
    type: str
    function: Function

@dataclass
class AIMessage(Message):
    tool_calls: List[ToolCalling] = None

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