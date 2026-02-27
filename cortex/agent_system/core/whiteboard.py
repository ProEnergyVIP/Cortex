"""
Whiteboard - Channel-based messaging protocol for multi-agent systems.

This module provides a simple, optional communication layer that enables
asynchronous information sharing between agents through channels.

Design Principles:
1. Optional - Systems work exactly as before without it
2. Simple Core - Just post and read messages to/from channels
3. Protocol, Not Implementation - Defines how agents communicate, not what
4. Storage Agnostic - Works with any storage backend
5. Extensible by Design - Subclass to add domain-specific features
6. No Imposed Business Logic - Core has no interpretation or access control
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any, Callable, Set
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import json
import logging
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

class Message(BaseModel):
    """A message posted to a channel on the whiteboard.
    
    Attributes:
        id: Unique identifier for this message (UUID)
        timestamp: When the message was posted
        sender: Name of the agent that posted the message
        channel: Channel name where the message was posted
        content: The message payload (any structured data)
        thread: Optional thread identifier for grouping related messages
        reply_to: Optional reference to another message ID this replies to
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    channel: str
    content: Dict[str, Any]
    thread: Optional[str] = None
    reply_to: Optional[str] = None


class WhiteboardStorage(ABC):
    """Abstract storage backend for whiteboard persistence.
    
    Implementations must provide methods to save and query messages.
    """
    
    @abstractmethod
    async def save(self, message: Message) -> None:
        """Store a message.
        
        Args:
            message: The message to store
        """
        pass
    
    @abstractmethod
    async def query(
        self, 
        channel: str, 
        since: Optional[datetime] = None,
        limit: int = 100,
        thread: Optional[str] = None
    ) -> List[Message]:
        """Retrieve messages from a channel.
        
        Args:
            channel: The channel name to query
            since: Optional timestamp to filter messages after
            limit: Maximum number of messages to return
            thread: Optional thread identifier to filter by
            
        Returns:
            List of messages matching the criteria, ordered by timestamp
        """
        pass
    
    async def close(self) -> None:
        """Close the storage connection (optional, for cleanup)."""
        pass


class InMemoryStorage(WhiteboardStorage):
    """Default in-memory storage for development and testing.
    
    Messages are stored in memory and lost when the process restarts.
    """
    
    _messages: Dict[str, List[Message]] = PrivateAttr()
    
    def __init__(self, **data):
        super().__init__(**data)
        self._messages = defaultdict(list)
    
    async def save(self, message: Message) -> None:
        """Store a message in memory."""
        self._messages[message.channel].append(message)
        logger.debug(f"InMemoryStorage: Saved message to '{message.channel}' (total: {len(self._messages[message.channel])})")
    
    async def query(
        self, 
        channel: str, 
        since: Optional[datetime] = None,
        limit: int = 100,
        thread: Optional[str] = None
    ) -> List[Message]:
        """Retrieve messages from memory."""
        messages = list(self._messages.get(channel, []))
        
        if since:
            messages = [m for m in messages if m.timestamp >= since]
        
        if thread:
            messages = [m for m in messages if m.thread == thread]
        
        # Return most recent messages, limited
        result = messages[-limit:] if limit < len(messages) else messages
        logger.debug(f"InMemoryStorage: Queried '{channel}' - returned {len(result)} of {len(self._messages.get(channel, []))} messages")
        return result


class RedisStorage(WhiteboardStorage):
    """Redis-backed storage for production use.
    
    Requires an async Redis client. Messages are stored in Redis lists
    with one list per channel.
    """
    
    _redis: Any = PrivateAttr()
    _prefix: str = PrivateAttr()
    
    def __init__(self, redis_client: Any, key_prefix: str = "whiteboard", **data):
        super().__init__(**data)
        self._redis = redis_client
        self._prefix = key_prefix
    
    def _channel_key(self, channel: str) -> str:
        """Generate the Redis key for a channel."""
        return f"{self._prefix}:channel:{channel}"
    
    async def save(self, message: Message) -> None:
        """Store a message in Redis."""
        key = self._channel_key(message.channel)
        payload = message.model_dump_json()
        await self._redis.rpush(key, payload)
        logger.debug(f"RedisStorage: Saved message to '{key}'")
    
    async def query(
        self, 
        channel: str, 
        since: Optional[datetime] = None,
        limit: int = 100,
        thread: Optional[str] = None
    ) -> List[Message]:
        """Retrieve messages from Redis.
        
        Note: Since Redis stores raw JSON, we need to fetch and filter in memory.
        For large channels, consider implementing time-indexed storage.
        """
        key = self._channel_key(channel)
        
        # Get all messages from the list (last 'limit' entries)
        raw_messages = await self._redis.lrange(key, -limit, -1)
        
        messages = []
        for raw in raw_messages:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            data = json.loads(raw)
            msg = Message(**data)
            
            # Apply filters
            if since and msg.timestamp < since:
                continue
            if thread and msg.thread != thread:
                continue
            
            messages.append(msg)
        
        # Sort by timestamp
        messages.sort(key=lambda m: m.timestamp)
        logger.debug(f"RedisStorage: Queried '{key}' - returned {len(messages)} messages (fetched {len(raw_messages)} from Redis)")
        return messages
    
    async def close(self) -> None:
        """Close the Redis connection."""
        if hasattr(self._redis, 'close'):
            logger.debug("RedisStorage: Closing Redis connection")
            await self._redis.close()


# =============================================================================
# Whiteboard Class
# =============================================================================

class Whiteboard:
    """Simple whiteboard for agent communication via channels.
    
    The base class provides minimal functionality for posting and reading
    messages. Subclass to add features like access control, audit logging,
    automatic summarization, etc.
    
    Example:
        ```python
        # Basic usage with in-memory storage
        wb = Whiteboard()
        
        # Post a message
        msg = await wb.post(
            sender="Coordinator",
            channel="project:acme",
            content={"type": "goal", "description": "Analyze merger"}
        )
        
        # Read messages
        messages = await wb.read(channel="project:acme")
        ```
    """
    
    _storage: WhiteboardStorage
    _subscribers: Dict[str, List[Callable[[Message], None]]]
    _channels: Set[str]
    
    def __init__(self, storage: Optional[WhiteboardStorage] = None):
        """Initialize the whiteboard.
        
        Args:
            storage: Storage backend. If None, uses InMemoryStorage.
        """
        self._storage = storage or InMemoryStorage()
        self._subscribers = defaultdict(list)
        self._channels = set()
        logger.debug(f"Whiteboard initialized with {type(self._storage).__name__}")
    
    # ---- Core Public API ----
    
    async def post(
        self, 
        sender: str,
        channel: str, 
        content: Dict[str, Any],
        thread: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> Message:
        """Post a message to a channel.
        
        Args:
            sender: Name of the agent posting the message
            channel: Channel name to post to
            content: Message payload (any JSON-serializable dict)
            thread: Optional thread identifier for grouping
            reply_to: Optional message ID this is replying to
            
        Returns:
            The created Message object
            
        Raises:
            ValueError: If sender or channel is empty
        """
        if not sender:
            raise ValueError("sender cannot be empty")
        if not channel:
            raise ValueError("channel cannot be empty")
        
        await self._before_post(sender, channel, content)
        
        message = Message(
            sender=sender,
            channel=channel,
            content=content,
            thread=thread,
            reply_to=reply_to
        )
        
        await self._storage.save(message)
        self._channels.add(channel)
        
        logger.debug(f"Message posted to '{channel}' by '{sender}': {content}")
        
        # Notify subscribers asynchronously (don't block on errors)
        await self._notify_subscribers(channel, message)
        
        await self._after_post(message)
        return message
    
    async def read(
        self,
        channel: str,
        since: Optional[datetime] = None,
        limit: int = 100,
        thread: Optional[str] = None
    ) -> List[Message]:
        """Read messages from a channel.
        
        Args:
            channel: Channel name to read from
            since: Optional timestamp to filter messages after
            limit: Maximum number of messages to return (default: 100)
            thread: Optional thread identifier to filter by
            
        Returns:
            List of messages matching the criteria, ordered by timestamp
        """
        await self._before_read(channel, since, limit, thread)
        messages = await self._storage.query(channel, since, limit, thread)
        await self._after_read(channel, messages)
        logger.debug(f"Read {len(messages)} messages from '{channel}' (since={since}, limit={limit}, thread={thread})")
        return messages
    
    async def subscribe(
        self, 
        channel: str, 
        callback: Callable[[Message], None]
    ) -> None:
        """Subscribe to receive notifications for new messages.
        
        Note: This is primarily useful for persistent agents. Most agents
        should use read() proactively to get messages.
        
        Args:
            channel: Channel name to monitor
            callback: Function to call when a new message arrives
        """
        self._subscribers[channel].append(callback)
        logger.debug(f"Subscribed to channel '{channel}' ({len(self._subscribers[channel])} subscribers total)")
    
    async def unsubscribe(
        self, 
        channel: str, 
        callback: Callable[[Message], None]
    ) -> None:
        """Unsubscribe from a channel.
        
        Args:
            channel: Channel name
            callback: The callback function to remove
        """
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(callback)
                logger.debug(f"Unsubscribed from channel '{channel}'")
            except ValueError:
                pass
    
    def list_channels(self) -> List[str]:
        """Return a list of all channels that have been used.
        
        Returns:
            Sorted list of channel names
        """
        channels = sorted(self._channels)
        logger.debug(f"Listing {len(channels)} channels: {channels}")
        return channels
    
    # ---- Extension Hooks ----
    
    async def _before_post(
        self, 
        sender: str, 
        channel: str, 
        content: Dict[str, Any]
    ) -> None:
        """Hook called before posting. Override to add validation/enrichment.
        
        This is called before the message is created and saved. Use this to:
        - Validate sender/channel
        - Enrich content with metadata
        - Apply content filtering
        - Check access permissions
        
        Args:
            sender: The agent posting
            channel: The target channel
            content: The message content (can be modified)
        """
        pass
    
    async def _after_post(self, message: Message) -> None:
        """Hook called after posting. Override to add side effects.
        
        This is called after the message is saved. Use this to:
        - Log the activity
        - Trigger external workflows
        - Update indexes
        - Send notifications
        
        Args:
            message: The posted message (immutable)
        """
        pass
    
    async def _before_read(
        self, 
        channel: str, 
        since: Optional[datetime],
        limit: int,
        thread: Optional[str]
    ) -> None:
        """Hook called before reading. Override to add access control.
        
        This is called before the query is executed. Use this to:
        - Check read permissions
        - Log access attempts
        - Modify query parameters
        
        Args:
            channel: The channel being read
            since: Timestamp filter
            limit: Limit parameter
            thread: Thread filter
        """
        pass
    
    async def _after_read(
        self, 
        channel: str, 
        messages: List[Message]
    ) -> None:
        """Hook called after reading. Override to add filtering/enrichment.
        
        This is called after the query is executed. Use this to:
        - Filter results
        - Enrich messages
        - Log access
        - Inject summaries
        
        Args:
            channel: The channel that was read
            messages: The retrieved messages (can be modified)
        """
        pass
    
    async def _notify_subscribers(
        self, 
        channel: str, 
        message: Message
    ) -> None:
        """Notify subscribers of new message.
        
        Args:
            channel: The channel with a new message
            message: The new message
        """
        for callback in self._subscribers.get(channel, []):
            try:
                callback(message)
            except Exception:
                # Subscriber errors shouldn't break posting
                pass
    
    async def close(self) -> None:
        """Close the whiteboard and its storage.
        
        Call this when the whiteboard is no longer needed.
        """
        logger.debug("Closing whiteboard")
        await self._storage.close()


# =============================================================================
# Extension Examples (for reference)
# =============================================================================

class ComplianceWhiteboard(Whiteboard):
    """Example: Whiteboard that adds compliance metadata to all messages.
    
    This demonstrates how to extend the whiteboard for specific needs.
    """
    
    _project_id: str
    _retention_days: int
    
    def __init__(
        self, 
        storage: Optional[WhiteboardStorage] = None,
        project_id: str = "default",
        retention_days: int = 90
    ):
        super().__init__(storage)
        self._project_id = project_id
        self._retention_days = retention_days
    
    async def _before_post(
        self, 
        sender: str, 
        channel: str, 
        content: Dict[str, Any]
    ) -> None:
        """Add compliance metadata before posting."""
        content["_compliance"] = {
            "project_id": self._project_id,
            "retention_until": (
                datetime.now() + timedelta(days=self._retention_days)
            ).isoformat(),
            "classification": self._classify_content(content)
        }
    
    def _classify_content(self, content: Dict[str, Any]) -> str:
        """Classify content based on keywords."""
        text = str(content).lower()
        if "financial" in text or "revenue" in text or "cost" in text:
            return "confidential"
        if "personal" in text or "private" in text:
            return "restricted"
        return "internal"


__all__ = [
    "Message",
    "WhiteboardStorage",
    "InMemoryStorage", 
    "RedisStorage",
    "Whiteboard",
    "ComplianceWhiteboard",
]
