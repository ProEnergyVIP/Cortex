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
    
    @abstractmethod
    async def cleanup(
        self,
        channel: str,
        max_age: Optional[timedelta] = None,
        max_count: Optional[int] = None,
        keep_min: int = 10
    ) -> int:
        """Remove old messages from a channel.
        
        Args:
            channel: The channel to clean up
            max_age: Remove messages older than this (e.g., timedelta(hours=24))
            max_count: Keep only this many most recent messages
            keep_min: Always keep at least this many messages (safety)
            
        Returns:
            Number of messages removed
            
        Note:
            If both max_age and max_count are specified, the more restrictive
            (i.e., the one that removes more messages) is applied.
        """
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

    async def close(self) -> None:
        """Close the storage connection (optional, for cleanup)."""
        pass

    async def cleanup(
        self,
        channel: str,
        max_age: Optional[timedelta] = None,
        max_count: Optional[int] = None,
        keep_min: int = 10
    ) -> int:
        """Remove old messages from a channel.
        
        Args:
            channel: The channel to clean up
            max_age: Remove messages older than this (e.g., timedelta(hours=24))
            max_count: Keep only this many most recent messages
            keep_min: Always keep at least this many messages (safety)
            
        Returns:
            Number of messages removed
            
        Note:
            If both max_age and max_count are specified, the more restrictive
            (i.e., the one that removes more messages) is applied.
        """
        removed = 0
        messages = self._messages.get(channel, [])
        if not messages:
            return 0
        
        # Determine which messages to keep
        keep_indices = set(range(len(messages)))
        
        if max_age is not None:
            cutoff = datetime.now() - max_age
            for i, msg in enumerate(messages):
                if msg.timestamp < cutoff:
                    keep_indices.discard(i)
        
        if max_count is not None and len(messages) > max_count:
            # Keep only the most recent max_count messages
            keep_indices = set(range(len(messages) - max_count, len(messages)))
        
        # Ensure we keep at least keep_min messages
        if len(keep_indices) < keep_min:
            # Add back the most recent messages until we hit keep_min
            for i in range(len(messages) - 1, -1, -1):
                if len(keep_indices) >= keep_min:
                    break
                keep_indices.add(i)
        
        # Filter messages
        new_messages = [messages[i] for i in sorted(keep_indices)]
        removed = len(messages) - len(new_messages)
        self._messages[channel] = new_messages
        
        if removed > 0:
            logger.debug(f"InMemoryStorage: Cleaned up {removed} messages from '{channel}' (kept {len(new_messages)})")
        
        return removed


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
        
        logger.debug(f"RedisStorage: Queried '{key}' - returned {len(messages)} messages (fetched {len(raw_messages)} from Redis)")
        return messages

    async def cleanup(
        self,
        channel: str,
        max_age: Optional[timedelta] = None,
        max_count: Optional[int] = None,
        keep_min: int = 10
    ) -> int:
        """Remove old messages from a Redis channel.
        
        Args:
            channel: The channel to clean up
            max_age: Remove messages older than this
            max_count: Keep only this many most recent messages
            keep_min: Always keep at least this many messages
            
        Returns:
            Number of messages removed
        """
        key = self._channel_key(channel)
        
        # Get all messages
        total = await self._redis.llen(key)
        if total == 0:
            return 0
        
        raw_messages = await self._redis.lrange(key, 0, -1)
        
        # Determine which messages to keep
        keep_indices = set(range(len(raw_messages)))
        
        if max_age is not None:
            cutoff = datetime.now() - max_age
            for i, raw in enumerate(raw_messages):
                if isinstance(raw, bytes):
                    raw = raw.decode('utf-8')
                data = json.loads(raw)
                msg = Message(**data)
                if msg.timestamp < cutoff:
                    keep_indices.discard(i)
        
        if max_count is not None and len(raw_messages) > max_count:
            # Keep only the most recent max_count messages
            keep_indices = set(range(len(raw_messages) - max_count, len(raw_messages)))
        
        # Ensure we keep at least keep_min messages
        if len(keep_indices) < keep_min:
            for i in range(len(raw_messages) - 1, -1, -1):
                if len(keep_indices) >= keep_min:
                    break
                keep_indices.add(i)
        
        # If we need to remove messages, rebuild the list
        removed = len(raw_messages) - len(keep_indices)
        if removed > 0:
            # Keep only the messages we want
            messages_to_keep = [raw_messages[i] for i in sorted(keep_indices)]
            
            # Delete and rebuild the list atomically using a pipeline
            pipe = self._redis.pipeline()
            pipe.delete(key)
            if messages_to_keep:
                pipe.rpush(key, *messages_to_keep)
            await pipe.execute()
            
            logger.debug(f"RedisStorage: Cleaned up {removed} messages from '{key}' (kept {len(messages_to_keep)})")
        
        return removed

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
    _auto_cleanup_max_size: int
    _auto_cleanup_keep_min: int
    
    def __init__(
        self, 
        storage: Optional[WhiteboardStorage] = None,
        auto_cleanup_max_size: int = 100,
        auto_cleanup_keep_min: int = 20
    ):
        """Initialize the whiteboard.
        
        Args:
            storage: Storage backend. If None, uses InMemoryStorage.
            auto_cleanup_max_size: Trigger auto-cleanup when channel exceeds this size
            auto_cleanup_keep_min: Keep at least this many messages during auto-cleanup
        """
        self._storage = storage or InMemoryStorage()
        self._subscribers = defaultdict(list)
        self._channels = set()
        self._auto_cleanup_max_size = auto_cleanup_max_size
        self._auto_cleanup_keep_min = auto_cleanup_keep_min
        logger.debug(f"Whiteboard initialized with {type(self._storage).__name__} "
                    f"(auto_cleanup_max_size={auto_cleanup_max_size}, auto_cleanup_keep_min={auto_cleanup_keep_min})")
    
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
        # Auto-cleanup oversized channels (size-based only, not age-based)
        if self._auto_cleanup_max_size > 0:
            await self._auto_cleanup_if_needed(message.channel)
    
    async def _auto_cleanup_if_needed(self, channel: str) -> None:
        """Automatically clean up channel if it exceeds max size.
        
        This is automatic size-based cleanup that triggers transparently
        when a channel gets too large. It only uses max_count (not age).
        
        Args:
            channel: The channel to check and potentially clean
        """
        try:
            # Get current channel size (quick count query)
            all_messages = await self._storage.query(channel, since=None, limit=1000, thread=None)
            current_size = len(all_messages)
            
            if current_size > self._auto_cleanup_max_size:
                # Auto-cleanup: only use max_count (size-based), not age
                removed = await self._storage.cleanup(
                    channel=channel,
                    max_age=None,  # Don't use age-based cleanup for auto
                    max_count=self._auto_cleanup_max_size,
                    keep_min=self._auto_cleanup_keep_min
                )
                if removed > 0:
                    logger.info(f"Auto-cleanup: removed {removed} messages from '{channel}' "
                               f"(size was {current_size}, now {current_size - removed})")
        except Exception:
            # Auto-cleanup errors should not break posting
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
    
    async def cleanup(
        self,
        channel: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        max_count: Optional[int] = None,
        keep_min: int = 10
    ) -> Dict[str, int]:
        """Clean up old messages from whiteboard channels.
        
        This method can be called periodically to remove stale messages
        and prevent memory/storage bloat. It's safe to call - it will
        always keep at least keep_min messages per channel.
        
        Args:
            channel: Specific channel to clean (None = all channels)
            max_age_hours: Remove messages older than this many hours
            max_count: Keep only this many most recent messages per channel
            keep_min: Always keep at least this many messages (safety)
            
        Returns:
            Dict mapping channel name to number of messages removed
            
        Example:
            ```python
            # Clean up messages older than 24 hours
            removed = await wb.cleanup(max_age_hours=24)
            
            # Keep only last 100 messages per channel
            removed = await wb.cleanup(max_count=100)
            
            # Clean specific channel
            removed = await wb.cleanup(channel="project:acme", max_age_hours=48)
            ```
        """
        from datetime import timedelta
        
        max_age = timedelta(hours=max_age_hours) if max_age_hours else None
        channels_to_clean = [channel] if channel else self.list_channels()
        
        results = {}
        for ch in channels_to_clean:
            removed = await self._storage.cleanup(
                channel=ch,
                max_age=max_age,
                max_count=max_count,
                keep_min=keep_min
            )
            if removed > 0:
                results[ch] = removed
                logger.debug(f"Cleaned up {removed} messages from channel '{ch}'")
        
        if results:
            total = sum(results.values())
            logger.info(f"Whiteboard cleanup complete: removed {total} messages from {len(results)} channels")
        
        return results
    
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
