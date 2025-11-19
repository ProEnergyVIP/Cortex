"""
Pluggable persistence for the Whiteboard (team-level state a.k.a. AgentSystemContext).

This mirrors the pattern used by the agent memory bank, providing an
async store interface and both in-memory and Redis-backed implementations.
"""
from __future__ import annotations

import json
from typing import Optional, Protocol

from .whiteboard import Whiteboard


class AsyncWhiteboardStore(Protocol):
    """Abstract async store for the Whiteboard.

    Implementations should persist and retrieve a single Whiteboard per
    logical key (e.g., user_id:conversation_id).
    """

    async def load(self, key: str) -> Optional[Whiteboard]:
        """Load a whiteboard by key. Return None if it does not exist."""
        ...

    async def save(self, key: str, board: Whiteboard) -> None:
        """Persist the given Whiteboard under the key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete the whiteboard stored under the key, if any."""
        ...

    async def is_active(self, key: str) -> bool:
        """Return True if a whiteboard exists for the key."""
        ...


class InMemoryWhiteboardStore:
    """Simple in-memory store for the Whiteboard (useful for tests/dev)."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    async def load(self, key: str) -> Optional[Whiteboard]:
        payload = self._store.get(key)
        if payload is None:
            return None
        return Whiteboard.model_validate(payload)

    async def save(self, key: str, board: Whiteboard) -> None:
        self._store[key] = board.model_dump()

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def is_active(self, key: str) -> bool:
        return key in self._store


class AsyncRedisWhiteboardStore:
    """Redis-backed async store for the Whiteboard.

    Uses a provided async Redis client. Keys are namespaced with a configurable
    prefix, defaulting to "whiteboard".
    """

    # Static mapping of async Redis clients
    _async_redis_clients = {}

    def __init__(self, async_redis_client, key_prefix: str = "whiteboard"):
        self.async_redis_client = async_redis_client
        self.key_prefix = key_prefix

    def _key(self, key: str) -> str:
        # caller should pass a stable logical key like f"{user_id}:{conversation_id}"
        return f"{self.key_prefix}:{key}"

    async def load(self, key: str) -> Optional[Whiteboard]:
        data = await self.async_redis_client.get(self._key(key))
        if not data:
            return None
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
        return Whiteboard.model_validate(payload)

    async def save(self, key: str, board: Whiteboard) -> None:
        payload = json.dumps(board.model_dump())
        await self.async_redis_client.set(self._key(key), payload)

    async def delete(self, key: str) -> None:
        await self.async_redis_client.delete(self._key(key))

    async def is_active(self, key: str) -> bool:
        exists = await self.async_redis_client.exists(self._key(key))
        return exists > 0

    # ---- Client registry and factory helpers ----
    @classmethod
    def register_client(cls, client_name: str, async_redis_client) -> None:
        """Register an async Redis client for later use."""
        cls._async_redis_clients[client_name] = async_redis_client

    @classmethod
    def _resolve_client(cls, *, async_redis_client=None, client_name: str = "default"):
        if async_redis_client is not None:
            return async_redis_client
        client = cls._async_redis_clients.get(client_name)
        if client is None:
            raise ValueError(
                "No async Redis client available. Provide async_redis_client or register one with client_name."
            )
        return client

    @classmethod
    def store_for(
        cls,
        *,
        user_id: str,
        async_redis_client=None,
        client_name: str = "default",
        base_prefix: str = "whiteboard",
    ) -> "AsyncRedisWhiteboardStore":
        """Create a store bound to a user-specific key prefix.

        Keys written by the returned store will be namespaced under
        f"{base_prefix}:user:{user_id}".
        """
        client = cls._resolve_client(async_redis_client=async_redis_client, client_name=client_name)
        key_prefix = f"{base_prefix}:user:{user_id}"
        return cls(client, key_prefix=key_prefix)
