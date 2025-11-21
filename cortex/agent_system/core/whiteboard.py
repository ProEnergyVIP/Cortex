from __future__ import annotations
from typing import Optional, Dict, List, Any, ClassVar
import json
from datetime import datetime
from enum import Enum
import uuid
from pydantic import BaseModel, Field, PrivateAttr


class WhiteboardUpdateType(str, Enum):
    """Types of updates that can be made to the Whiteboard."""
    PROGRESS = "progress"
    FINDING = "finding"
    DECISION = "decision"
    ARTIFACT = "artifact"
    BLOCKER = "blocker"


class WhiteboardUpdate(BaseModel):
    """Represents an update to the Whiteboard by an agent.
    
    Attributes:
        id: Unique identifier for this update
        agent_name: Name of the agent making the update
        type: Type of update (progress, finding, decision, artifact, blocker)
        content: Flexible dictionary containing update data
        timestamp: When the update was created
        tags: List of tags for categorization and filtering
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str
    type: WhiteboardUpdateType
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True  # Serialize enums as their values


class WhiteboardTopic(BaseModel):
    """Per-topic Whiteboard state for a team.
    
    This holds all fields that are naturally scoped to a specific conversation
    topic (mission, progress, updates, blockers, artifacts, etc.).
    """
    
    mission: str = ""
    current_focus: str = ""
    progress: str = ""
    updates: List[WhiteboardUpdate] = Field(default_factory=list)
    artifacts: Dict[str, List[Dict]] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)
    active_blockers: List[str] = Field(default_factory=list)


class Whiteboard(BaseModel):
    """Container for team Whiteboard state, including topics and metadata."""
    # Team policy/metadata (not tied to a single topic)
    team_roles: Dict[str, str] = Field(default_factory=dict)  # agent_name -> role
    protocols: List[str] = Field(default_factory=list)
    default_topic: str = "general"
    max_updates_per_topic: int = 10
    # Topic state
    current_topic: str = "general"
    topics: Dict[str, WhiteboardTopic] = Field(default_factory=dict)

    # ---- Topic helpers ----
    def set_current_topic(self, topic: str) -> None:
        if topic not in self.topics:
            self.topics[topic] = WhiteboardTopic()
        self.current_topic = topic

    def _state(self) -> WhiteboardTopic:
        if self.current_topic not in self.topics:
            self.topics[self.current_topic] = WhiteboardTopic()
        return self.topics[self.current_topic]

    # ---- Topic routing helpers ----
    def known_topics(self) -> List[str]:
        """Return topic names that currently exist in the whiteboard."""
        return sorted(self.topics.keys())

    # ---- Operations ----
    def add_update(
        self,
        *,
        agent_name: str,
        update_type: WhiteboardUpdateType,
        content: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> WhiteboardUpdate:
        state = self._state()
        update = WhiteboardUpdate(
            agent_name=agent_name,
            type=update_type,
            content=content,
            tags=tags or [],
        )
        state.updates.append(update)
        state.last_activity = datetime.now()
        # Enforce per-topic update limit (oldest first) if configured
        while len(state.updates) > self.max_updates_per_topic:
            state.updates.pop(0)
        return update

    def clear_topic(self, topic: Optional[str] = None) -> None:
        """Reset the specified topic (or current topic) to a clean state."""
        target = topic or self.current_topic
        self.topics[target] = WhiteboardTopic()
        if topic is None:
            self.current_topic = target

    # High-level coordinator-oriented operations (sync)
    def set_mission_focus(self, *, mission: Optional[str], focus: Optional[str]) -> None:
        state = self._state()
        if mission is not None:
            state.mission = mission
        if focus is not None:
            state.current_focus = focus
        self.add_update(
            agent_name="Coordinator",
            update_type=WhiteboardUpdateType.DECISION,
            content={
                "action": "mission_updated",
                "mission": state.mission,
                "focus": state.current_focus,
            },
            tags=["coordinator", "mission", "planning"],
        )

    def update_progress(self, *, progress: str) -> None:
        state = self._state()
        state.progress = progress
        self.add_update(
            agent_name="Coordinator",
            update_type=WhiteboardUpdateType.PROGRESS,
            content={"action": "progress_updated", "progress": progress},
            tags=["coordinator", "progress", "status"],
        )

    def add_blocker(self, *, blocker: str) -> str:
        state = self._state()
        if blocker in state.active_blockers:
            status = "exists"
        else:
            state.active_blockers.append(blocker)
            status = "added"
        self.add_update(
            agent_name="Coordinator",
            update_type=WhiteboardUpdateType.BLOCKER,
            content={
                "action": "blocker_add",
                "blocker": blocker,
                "status": status,
                "active_blockers": list(state.active_blockers),
            },
            tags=["coordinator", "blocker", "add"],
        )
        return status

    def remove_blocker(self, *, blocker: str) -> str:
        state = self._state()
        if blocker in state.active_blockers:
            state.active_blockers.remove(blocker)
            status = "removed"
        else:
            status = "not_found"
        self.add_update(
            agent_name="Coordinator",
            update_type=WhiteboardUpdateType.PROGRESS,
            content={
                "action": "blocker_remove",
                "blocker": blocker,
                "status": status,
                "active_blockers": list(state.active_blockers),
            },
            tags=["coordinator", "blocker", "remove"],
        )
        return status

    def log_decision(self, *, decision: str, rationale: Optional[str] = None) -> None:
        self.add_update(
            agent_name="Coordinator",
            update_type=WhiteboardUpdateType.DECISION,
            content={"decision": decision, "rationale": rationale},
            tags=["coordinator", "decision"],
        )

    # No async wrappers; persistence is external.

    def get_agent_view(self, agent_name: str) -> Dict:
        state = self._state()
        return {
            "mission": state.mission,
            "current_focus": state.current_focus,
            "progress": state.progress,
            "my_role": self.team_roles.get(agent_name, ""),
            "team_roles": self.team_roles,
            "protocols": self.protocols,
            "recent_updates": [
                {
                    "id": u.id,
                    "agent_name": u.agent_name,
                    "type": u.type.value if isinstance(u.type, WhiteboardUpdateType) else u.type,
                    "content": u.content,
                    "tags": u.tags,
                    "timestamp": u.timestamp.isoformat(),
                }
                for u in state.updates[-10:]
            ],
            "artifacts": state.artifacts,
            "active_blockers": state.active_blockers,
            "last_activity": state.last_activity.isoformat(),
        }

    def get_recent_updates(
        self,
        *,
        since: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        update_type: Optional[WhiteboardUpdateType] = None,
    ) -> List[WhiteboardUpdate]:
        state = self._state()
        updates = state.updates
        if since:
            updates = [u for u in updates if u.timestamp >= since]
        if agent_name:
            updates = [u for u in updates if u.agent_name == agent_name]
        if update_type:
            updates = [u for u in updates if u.type == update_type]
        return updates

    def apply_suggestion(
        self,
        suggestion: Dict[str, Any],
        *,
        source_agent: str = "Coordinator",
    ) -> None:
        if not suggestion:
            return
        state = self._state()

        # progress
        progress = suggestion.get("progress")
        if isinstance(progress, str) and progress.strip():
            state.progress = progress
            self.add_update(
                agent_name=source_agent,
                update_type=WhiteboardUpdateType.PROGRESS,
                content={"action": "progress_suggested", "progress": progress},
                tags=["whiteboard", "progress"],
            )

        # blockers add
        blockers_add = suggestion.get("blockers_add") or []
        if isinstance(blockers_add, list):
            for blocker in blockers_add:
                if not isinstance(blocker, str):
                    continue
                b = blocker.strip()
                if not b or b in state.active_blockers:
                    continue
                state.active_blockers.append(b)
                self.add_update(
                    agent_name=source_agent,
                    update_type=WhiteboardUpdateType.BLOCKER,
                    content={
                        "action": "blocker_add_suggested",
                        "blocker": b,
                        "active_blockers": list(state.active_blockers),
                    },
                    tags=["whiteboard", "blocker", "add"],
                )

        # blockers remove
        blockers_remove = suggestion.get("blockers_remove") or []
        if isinstance(blockers_remove, list):
            for blocker in blockers_remove:
                if not isinstance(blocker, str):
                    continue
                b = blocker.strip()
                if not b or b not in state.active_blockers:
                    continue
                state.active_blockers.remove(b)
                self.add_update(
                    agent_name=source_agent,
                    update_type=WhiteboardUpdateType.PROGRESS,
                    content={
                        "action": "blocker_remove_suggested",
                        "blocker": b,
                        "active_blockers": list(state.active_blockers),
                    },
                    tags=["whiteboard", "blocker", "remove"],
                )

        # decisions
        decisions = suggestion.get("decisions") or []
        if isinstance(decisions, list):
            for d in decisions:
                if not isinstance(d, dict):
                    continue
                decision_text = d.get("decision")
                if not isinstance(decision_text, str) or not decision_text.strip():
                    continue
                rationale = d.get("rationale") if isinstance(d.get("rationale"), str) else None
                self.add_update(
                    agent_name=source_agent,
                    update_type=WhiteboardUpdateType.DECISION,
                    content={
                        "decision": decision_text,
                        "rationale": rationale,
                        "action": "decision_suggested",
                    },
                    tags=["whiteboard", "decision"],
                )


class RedisWhiteboard(Whiteboard):
    """Redis-backed Whiteboard that persists state to a Redis key on mutation."""

    _redis_client: Any = PrivateAttr()
    _key: str = PrivateAttr()

    def __init__(self, *, redis_client, key: str, **data):
        super().__init__(**data)
        self._redis_client = redis_client
        self._key = key

    # ---- Persistence helpers ----
    def _save(self) -> None:
        payload = self.model_dump_json()
        self._redis_client.set(self._key, payload)

    @classmethod
    def load(cls, *, redis_client, key: str) -> "RedisWhiteboard":
        data = redis_client.get(key)
        if not data:
            return cls(redis_client=redis_client, key=key)
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
        return cls(redis_client=redis_client, key=key, **payload)

    def delete(self) -> None:
        self._redis_client.delete(self._key)

    # ---- Mutations with auto-persist ----
    def set_current_topic(self, topic: str) -> None:
        super().set_current_topic(topic)
        self._save()

    def add_update(
        self,
        *,
        agent_name: str,
        update_type: WhiteboardUpdateType,
        content: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> WhiteboardUpdate:
        upd = super().add_update(
            agent_name=agent_name,
            update_type=update_type,
            content=content,
            tags=tags,
        )
        self._save()
        return upd

    def configure_topics(
        self,
        *,
        topic_keywords: Optional[Dict[str, List[str]]] = None,
        default_topic: Optional[str] = None,
    ) -> None:
        super().configure_topics(topic_keywords=topic_keywords, default_topic=default_topic)
        self._save()

    def update_progress(self, *, progress: str) -> None:
        super().update_progress(progress=progress)
        self._save()

    def add_blocker(self, *, blocker: str) -> str:
        status = super().add_blocker(blocker=blocker)
        self._save()
        return status

    def remove_blocker(self, *, blocker: str) -> str:
        status = super().remove_blocker(blocker=blocker)
        self._save()
        return status

    def log_decision(self, *, decision: str, rationale: Optional[str] = None) -> None:
        super().log_decision(decision=decision, rationale=rationale)
        self._save()

    def apply_suggestion(
        self,
        suggestion: Dict[str, Any],
        *,
        source_agent: str = "Coordinator",
    ) -> None:
        super().apply_suggestion(suggestion, source_agent=source_agent)
        self._save()

    def clear_topic(self, topic: Optional[str] = None) -> None:
        super().clear_topic(topic=topic)
        self._save()


class AsyncWhiteboard(Whiteboard):
    """Async base variant of Whiteboard with async methods.

    All operations mirror the sync API but are defined as async to enable
    truly async-backed implementations. Default behavior is in-memory.
    """

    # ---- Topic helpers ----
    async def set_current_topic(self, topic: str) -> None:
        super().set_current_topic(topic)

    async def known_topics(self) -> List[str]:
        return super().known_topics()

    async def configure_topics(
        self,
        *,
        topic_keywords: Optional[Dict[str, List[str]]] = None,
        default_topic: Optional[str] = None,
    ) -> None:
        super().configure_topics(topic_keywords=topic_keywords, default_topic=default_topic)

    async def detect_topic(self, message: str) -> str:
        return super().detect_topic(message)

    async def set_topic_for_message(self, message: str) -> str:
        return super().set_topic_for_message(message)

    # ---- Operations ----
    async def add_update(
        self,
        *,
        agent_name: str,
        update_type: WhiteboardUpdateType,
        content: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> WhiteboardUpdate:
        return super().add_update(
            agent_name=agent_name,
            update_type=update_type,
            content=content,
            tags=tags,
        )

    # High-level coordinator-oriented operations
    async def set_mission_focus(self, *, mission: Optional[str], focus: Optional[str]) -> None:
        super().set_mission_focus(mission=mission, focus=focus)

    async def update_progress(self, *, progress: str) -> None:
        super().update_progress(progress=progress)

    async def add_blocker(self, *, blocker: str) -> str:
        return super().add_blocker(blocker=blocker)

    async def remove_blocker(self, *, blocker: str) -> str:
        return super().remove_blocker(blocker=blocker)

    async def log_decision(self, *, decision: str, rationale: Optional[str] = None) -> None:
        super().log_decision(decision=decision, rationale=rationale)

    async def get_agent_view(self, agent_name: str) -> Dict:
        return super().get_agent_view(agent_name)

    async def get_recent_updates(
        self,
        *,
        since: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        update_type: Optional[WhiteboardUpdateType] = None,
    ) -> List[WhiteboardUpdate]:
        return super().get_recent_updates(
            since=since, agent_name=agent_name, update_type=update_type
        )

    async def apply_suggestion(
        self,
        suggestion: Dict[str, Any],
        *,
        source_agent: str = "Coordinator",
    ) -> None:
        super().apply_suggestion(suggestion, source_agent=source_agent)

    async def clear_topic(self, topic: Optional[str] = None) -> None:
        super().clear_topic(topic=topic)


class AsyncRedisWhiteboard(AsyncWhiteboard):
    """Async Redis-backed Whiteboard that persists to a Redis key on mutation."""

    _async_redis_client: Any = PrivateAttr()
    _key: str = PrivateAttr()
    # Static mapping of async Redis clients
    _async_redis_clients: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *, async_redis_client, key: str, **data):
        super().__init__(**data)
        self._async_redis_client = async_redis_client
        self._key = key

    # ---- Persistence helpers ----
    async def _save(self) -> None:
        payload = self.model_dump_json()
        await self._async_redis_client.set(self._key, payload)

    @classmethod
    async def load(cls, *, async_redis_client, key: str) -> "AsyncRedisWhiteboard":
        data = await async_redis_client.get(key)
        if not data:
            return cls(async_redis_client=async_redis_client, key=key)
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
        return cls(async_redis_client=async_redis_client, key=key, **payload)

    async def delete(self) -> None:
        await self._async_redis_client.delete(self._key)

    # ---- Client registry and factory helpers ----
    @classmethod
    def register_client(cls, client_name: str, async_redis_client) -> None:
        """Register an async Redis client for later use."""
        cls._async_redis_clients[client_name] = async_redis_client

    @classmethod
    def _resolve_client(cls, kwargs):
        """Resolve an async Redis client from kwargs or registry."""
        async_redis_client = kwargs.get("async_redis_client")
        client_name = kwargs.get("client_name", "default")
        if async_redis_client is None:
            async_redis_client = cls._async_redis_clients.get(client_name)
        if async_redis_client is None:
            raise ValueError(
                "No async Redis client available. Either provide 'async_redis_client' or register a client with 'client_name'"
            )
        return async_redis_client

    @staticmethod
    async def _delete_by_pattern(async_redis_client, key_pattern: str) -> None:
        """Delete all keys matching a pattern (async)."""
        keys = await async_redis_client.keys(key_pattern)
        if keys:
            # Some clients require splat expansion
            await async_redis_client.delete(*keys)

    @classmethod
    async def whiteboard_for(
        cls,
        user_id: str,
        **kwargs,
    ) -> "AsyncRedisWhiteboard":
        """Get or create an async Redis whiteboard for a user.

        Args:
            user_id: User ID
            **kwargs: Optional 'client_name', 'async_redis_client', and 'base_prefix'
        """
        async_redis_client = cls._resolve_client(kwargs)
        base_prefix = kwargs.get("base_prefix", "whiteboard")
        key = f"{base_prefix}:user:{user_id}"
        return await cls.load(async_redis_client=async_redis_client, key=key)

    @classmethod
    async def clear_whiteboard_for(cls, user_id: str, **kwargs) -> None:
        """Delete a user's whiteboard if it exists."""
        async_redis_client = cls._resolve_client(kwargs)
        base_prefix = kwargs.get("base_prefix", "whiteboard")
        key = f"{base_prefix}:user:{user_id}"
        await async_redis_client.delete(key)

    @classmethod
    async def reset_all(cls, **kwargs) -> None:
        """Delete all whiteboards for all users under the base prefix."""
        async_redis_client = cls._resolve_client(kwargs)
        base_prefix = kwargs.get("base_prefix", "whiteboard")
        key_pattern = f"{base_prefix}:user:*"
        await cls._delete_by_pattern(async_redis_client, key_pattern)

    @classmethod
    async def is_active(cls, user_id: str, **kwargs) -> bool:
        """Check if a user's whiteboard exists."""
        async_redis_client = cls._resolve_client(kwargs)
        base_prefix = kwargs.get("base_prefix", "whiteboard")
        key = f"{base_prefix}:user:{user_id}"
        exists = await async_redis_client.exists(key)
        return exists > 0

    # ---- Mutations with auto-persist ----
    async def set_current_topic(self, topic: str) -> None:
        await super().set_current_topic(topic)
        await self._save()

    async def add_update(
        self,
        *,
        agent_name: str,
        update_type: WhiteboardUpdateType,
        content: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> WhiteboardUpdate:
        upd = await super().add_update(
            agent_name=agent_name,
            update_type=update_type,
            content=content,
            tags=tags,
        )
        await self._save()
        return upd

    async def configure_topics(
        self,
        *,
        topic_keywords: Optional[Dict[str, List[str]]] = None,
        default_topic: Optional[str] = None,
    ) -> None:
        await super().configure_topics(topic_keywords=topic_keywords, default_topic=default_topic)
        await self._save()

    async def update_progress(self, *, progress: str) -> None:
        await super().update_progress(progress=progress)
        await self._save()

    async def add_blocker(self, *, blocker: str) -> str:
        status = await super().add_blocker(blocker=blocker)
        await self._save()
        return status

    async def remove_blocker(self, *, blocker: str) -> str:
        status = await super().remove_blocker(blocker=blocker)
        await self._save()
        return status

    async def log_decision(self, *, decision: str, rationale: Optional[str] = None) -> None:
        await super().log_decision(decision=decision, rationale=rationale)
        await self._save()

    async def apply_suggestion(
        self,
        suggestion: Dict[str, Any],
        *,
        source_agent: str = "Coordinator",
    ) -> None:
        await super().apply_suggestion(suggestion, source_agent=source_agent)
        await self._save()

    async def clear_topic(self, topic: Optional[str] = None) -> None:
        await super().clear_topic(topic=topic)
        await self._save()
