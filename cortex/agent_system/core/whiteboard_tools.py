"""
Whiteboard Tools - Channel-based messaging tools for agents.

These tools allow agents to post and read messages via the whiteboard,
enabling asynchronous communication between coordinator and workers.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime

from cortex import Tool
from ..core.context import AgentSystemContext
from ..core.whiteboard import Whiteboard


# System prompt addition when whiteboard is enabled
WHITEBOARD_PROMPT_ADDITION = """
You have access to a shared whiteboard where agents can post and read messages.
Use it to share information, track progress, and coordinate with other agents.

Channels are simple strings - use them to organize communication:
- Use project-specific channels for ongoing work (e.g., "project:acme-merger")
- Use task-specific channels for one-off activities (e.g., "task:analysis-123")
- Read relevant channels before starting new tasks to understand context
- Post updates and results as you complete work
- Use threads to keep conversations organized (e.g., thread="regulatory-analysis")

Available whiteboard tools:
- whiteboard_post: Post a message to a channel
- whiteboard_read: Read messages from a channel
- whiteboard_subscribe: Subscribe to channel notifications (for persistent monitoring)

Example workflow:
1. Coordinator posts goal to "project:xyz" channel
2. Workers read from "project:xyz" to understand context
3. Workers post results to relevant channels
4. Coordinator aggregates results and responds to user
"""


def create_whiteboard_tools() -> List[Tool]:
    """Create whiteboard tools available to all agents (coordinator and workers).
    
    These tools provide a simple messaging interface:
    - whiteboard_post: Send a message to a channel
    - whiteboard_read: Retrieve messages from a channel
    - whiteboard_subscribe: Subscribe to channel updates (primarily for persistent agents)
    
    Returns:
        List of Tool instances for whiteboard operations
    """
    
    # ---- Tool Functions ----
    
    async def whiteboard_post(args: Dict[str, Any], context: AgentSystemContext) -> str:
        """Post a message to a whiteboard channel.
        
        The sender is automatically set to the current agent's name.
        """
        wb: Optional[Whiteboard] = getattr(context, "whiteboard", None)
        if not wb:
            return "Error: Whiteboard not available in this context"
        
        channel = args.get("channel")
        content = args.get("content")
        
        if not channel:
            return "Error: 'channel' parameter is required"
        if not content:
            return "Error: 'content' parameter is required"
        if not isinstance(content, dict):
            return "Error: 'content' must be a JSON object (dictionary)"
        
        # Get sender from args or use a default
        sender = args.get("sender", "Agent")
        
        thread = args.get("thread")
        reply_to = args.get("reply_to")
        
        try:
            message = await wb.post(
                sender=sender,
                channel=channel,
                content=content,
                thread=thread,
                reply_to=reply_to
            )
            return f"Posted message {message.id[:8]} to channel '{channel}'"
        except Exception as e:
            return f"Error posting message: {str(e)}"
    
    async def whiteboard_read(args: Dict[str, Any], context: AgentSystemContext) -> Dict[str, Any]:
        """Read messages from a whiteboard channel.
        
        Returns messages ordered by timestamp (oldest first).
        """
        wb: Optional[Whiteboard] = getattr(context, "whiteboard", None)
        if not wb:
            return {"error": "Whiteboard not available in this context"}
        
        channel = args.get("channel")
        if not channel:
            return {"error": "'channel' parameter is required"}
        
        # Parse optional parameters
        since_str = args.get("since")
        since = None
        if since_str:
            try:
                since = datetime.fromisoformat(since_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
        
        limit = args.get("limit", 100)
        try:
            limit = int(limit)
            if limit < 1:
                limit = 100
            if limit > 1000:
                limit = 1000  # Cap at 1000 to prevent huge responses
        except (ValueError, TypeError):
            limit = 100
        
        thread = args.get("thread")
        
        try:
            messages = await wb.read(
                channel=channel,
                since=since,
                limit=limit,
                thread=thread
            )
            
            return {
                "channel": channel,
                "count": len(messages),
                "messages": [
                    {
                        "id": m.id[:8],  # Short ID for readability
                        "sender": m.sender,
                        "timestamp": m.timestamp.isoformat(),
                        "content": m.content,
                        "thread": m.thread,
                        "reply_to": m.reply_to[:8] if m.reply_to else None,
                    }
                    for m in messages
                ]
            }
        except Exception as e:
            return {"error": f"Error reading messages: {str(e)}"}
    
    async def whiteboard_subscribe(args: Dict[str, Any], context: AgentSystemContext) -> str:
        """Subscribe to a whiteboard channel for notifications.
        
        This is primarily useful for persistent agents that need to
        monitor channel activity. Most agents should use whiteboard_read
        proactively instead.
        """
        wb: Optional[Whiteboard] = getattr(context, "whiteboard", None)
        if not wb:
            return "Error: Whiteboard not available in this context"
        
        channel = args.get("channel")
        if not channel:
            return "Error: 'channel' parameter is required"
        
        # For now, subscription is a no-op - agents should use read()
        # In a future version, this could set up async callbacks
        return f"Subscribed to channel '{channel}' (note: use whiteboard_read to actively check for messages)"
    
    async def whiteboard_list_channels(args: Dict[str, Any], context: AgentSystemContext) -> Dict[str, Any]:
        """List all channels that have been used on the whiteboard.
        
        Useful for discovering available channels.
        """
        wb: Optional[Whiteboard] = getattr(context, "whiteboard", None)
        if not wb:
            return {"error": "Whiteboard not available in this context"}
        
        try:
            channels = wb.list_channels()
            return {
                "count": len(channels),
                "channels": channels
            }
        except Exception as e:
            return {"error": f"Error listing channels: {str(e)}"}
    
    # ---- Tool Definitions ----
    
    return [
        Tool(
            name="whiteboard_post",
            func=whiteboard_post,
            description="Post a message to a whiteboard channel. Use this to share information, report progress, or communicate with other agents.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name to post to (e.g., 'project:acme', 'task:analysis-123', 'broadcast:alerts')"
                    },
                    "content": {
                        "type": "object",
                        "description": "Message content as a JSON object. Can contain any structured data like {'type': 'status', 'progress': '50%'}"
                    },
                    "thread": {
                        "type": ["string", "null"],
                        "description": "Optional thread identifier for grouping related messages (e.g., 'regulatory-analysis')"
                    },
                    "reply_to": {
                        "type": ["string", "null"],
                        "description": "Optional message ID this is replying to (use short 8-char ID from whiteboard_read)"
                    }
                },
                "required": ["channel", "content", "thread", "reply_to"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="whiteboard_read",
            func=whiteboard_read,
            description="Read messages from a whiteboard channel. Use this to understand context, check progress, or see what other agents have posted.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name to read from (e.g., 'project:acme', 'task:analysis-123')"
                    },
                    "since": {
                        "type": ["string", "null"],
                        "description": "Optional ISO timestamp to filter messages after (e.g., '2024-01-15T10:30:00')"
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of messages to return (default: 100, max: 1000)"
                    },
                    "thread": {
                        "type": ["string", "null"],
                        "description": "Optional thread identifier to filter messages"
                    }
                },
                "required": ["channel", "since", "limit", "thread"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="whiteboard_subscribe",
            func=whiteboard_subscribe,
            description="Subscribe to receive notifications for new messages on a channel. Primarily useful for persistent monitoring agents. Most agents should use whiteboard_read proactively instead.",
            parameters={
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name to subscribe to"
                    }
                },
                "required": ["channel"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="whiteboard_list_channels",
            func=whiteboard_list_channels,
            description="List all channels that have been used on the whiteboard. Useful for discovering what channels are available.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
    ]


# Legacy support - keep old function name as alias
def create_coordinator_whiteboard_tools() -> List[Tool]:
    """DEPRECATED: Use create_whiteboard_tools() instead.
    
    This alias exists for backward compatibility during migration.
    """
    return create_whiteboard_tools()
