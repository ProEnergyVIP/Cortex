from __future__ import annotations
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid
from pydantic import BaseModel, Field


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
    # Topic routing configuration
    topic_keywords: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "solar": ["solar", "pv", "photovoltaic", "inverter", "panel"],
            "banking": ["bank", "loan", "credit", "account", "interest"],
            "general": [],
        }
    )
    default_topic: str = "general"
    max_updates_per_topic: int = 200
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
        """Return configured topic names plus topics with active state."""
        wb_topics = set(self.topics.keys())
        known = set(self.topic_keywords.keys()) | wb_topics
        return sorted(known)

    def configure_topics(
        self,
        *,
        topic_keywords: Optional[Dict[str, List[str]]] = None,
        default_topic: Optional[str] = None,
    ) -> None:
        """Configure keyword routing and default topic on the whiteboard."""
        if topic_keywords is not None:
            self.topic_keywords = topic_keywords
        if default_topic is not None:
            self.default_topic = default_topic
            self.set_current_topic(default_topic)

    def detect_topic(self, message: str) -> str:
        """Detect topic from a message with simple keyword matching."""
        text = message.lower()
        for topic, keywords in self.topic_keywords.items():
            for kw in keywords:
                if kw and kw.lower() in text:
                    return topic
        return self.default_topic

    def set_topic_for_message(self, message: str) -> str:
        """Detect the topic from message and set it as current, returning it."""
        topic = self.detect_topic(message)
        self.set_current_topic(topic)
        return topic

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
        return update

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
