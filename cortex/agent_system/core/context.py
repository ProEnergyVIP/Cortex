from functools import cached_property
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid
from cortex import LLM, AsyncAgentMemoryBank, GPTModels
from cortex.message import AgentUsage
from cortex.backend import ReasoningEffort
from pydantic import BaseModel, Field, PrivateAttr


class UpdateType(str, Enum):
    """Types of updates that can be made to shared context."""
    PROGRESS = "progress"
    FINDING = "finding"
    DECISION = "decision"
    ARTIFACT = "artifact"
    BLOCKER = "blocker"


class ContextUpdate(BaseModel):
    """Represents an update to the shared context by an agent.
    
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
    type: UpdateType
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True  # Serialize enums as their values


class TopicContext(BaseModel):
    """Per-topic shared context state for a team.

    This holds all fields that are naturally scoped to a specific conversation
    topic (mission, progress, updates, blockers, artifacts, etc.). The
    AgentSystemContext exposes a *view* of the current topic's TopicContext via
    its top-level fields.
    """

    mission: str = ""
    current_focus: str = ""
    progress: str = ""
    updates: List[ContextUpdate] = Field(default_factory=list)
    artifacts: Dict[str, List[Dict]] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)
    active_blockers: List[str] = Field(default_factory=list)


class AgentSystemContext(BaseModel):
    # Existing fields
    usage: Optional[AgentUsage] = None  # Usage tracking for the agent
    memory_bank: Optional[object] = None  # Memory bank for the agent

    # Shared context fields for multi-agent coordination (view of current topic)
    mission: str = ""
    current_focus: str = ""
    progress: str = ""
    team_roles: Dict[str, str] = Field(default_factory=dict)  # agent_name -> role
    protocols: List[str] = Field(default_factory=list)
    updates: List[ContextUpdate] = Field(default_factory=list)
    artifacts: Dict[str, List[Dict]] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)
    active_blockers: List[str] = Field(default_factory=list)

    # Topic management: a single context instance with per-topic state
    current_topic: str = "general"
    topics: Dict[str, TopicContext] = Field(default_factory=dict)

    # Topic routing configuration (not part of the serialized model)
    _topic_keywords: Dict[str, List[str]] = PrivateAttr(
        default_factory=lambda: {
            "solar": ["solar", "pv", "photovoltaic", "inverter", "panel"],
            "banking": ["bank", "loan", "credit", "account", "interest"],
            "general": [],
        }
    )
    _default_topic: str = PrivateAttr(default="general")

    async def get_memory_bank(self) -> AsyncAgentMemoryBank:
        """Get the agent memory bank for this context."""
        if not self.memory_bank:
            raise ValueError("Memory bank not initialized")
        return self.memory_bank
    
    @cached_property
    def llm_primary(self) -> LLM:
        """Primary, general-purpose reasoning LLM for agents.
        
        Centralized here so agents can reuse a consistent default model
        and temperature. Update this property to roll out model changes app-wide.
        """
        return LLM(model=GPTModels.GPT_5_MINI, reasoning_effort=ReasoningEffort.MINIMAL)

    @cached_property
    def llm_creative(self) -> LLM:
        """Creative/high-variance LLM for tasks benefiting from more exploration.
        
        Uses the same base model as `llm_primary` with a higher temperature.
        """
        return LLM(model=GPTModels.GPT_5_MINI, reasoning_effort=ReasoningEffort.MEDIUM)
    
    # --- Core shared context methods (operate on current topic) ---

    def _get_or_create_topic_state(self, topic: str) -> TopicContext:
        """Get or create the TopicContext for a given topic.

        For the very first topic created, if this AgentSystemContext already has
        non-empty mission/progress/updates/etc, those are treated as belonging
        to that topic and used to initialize its state for backward
        compatibility.
        """

        if topic in self.topics:
            return self.topics[topic]

        # Initialize fresh topic state
        state = TopicContext()

        # If this is the first topic and the current view has data, seed from it
        if not self.topics and (
            self.mission
            or self.current_focus
            or self.progress
            or self.updates
            or self.artifacts
            or self.active_blockers
        ):
            state.mission = self.mission
            state.current_focus = self.current_focus
            state.progress = self.progress
            state.updates = list(self.updates)
            state.artifacts = dict(self.artifacts)
            state.last_activity = self.last_activity
            state.active_blockers = list(self.active_blockers)

        self.topics[topic] = state
        return state

    def _sync_view_from_state(self, state: TopicContext) -> None:
        """Update top-level view fields from a TopicContext state."""

        self.mission = state.mission
        self.current_focus = state.current_focus
        self.progress = state.progress
        self.updates = list(state.updates)
        self.artifacts = dict(state.artifacts)
        self.last_activity = state.last_activity
        self.active_blockers = list(state.active_blockers)

    def set_current_topic(self, topic: str) -> None:
        """Switch the current topic and update the view fields accordingly."""

        state = self._get_or_create_topic_state(topic)
        self.current_topic = topic
        self._sync_view_from_state(state)

    def set_topic_for_message(self, message: str) -> str:
        """Detect the appropriate topic for a message and make it current.

        Returns the detected topic name.
        """

        topic = self.detect_topic(message)
        self.set_current_topic(topic)
        return topic

    def add_update(
        self, 
        agent_name: str, 
        update_type: UpdateType, 
        content: Dict[str, Any], 
        tags: Optional[List[str]] = None
    ) -> ContextUpdate:
        """Add an update to the shared context from an agent.
        
        Args:
            agent_name: Name of the agent making the update
            update_type: Type of update (UpdateType enum: PROGRESS, FINDING, DECISION, ARTIFACT, BLOCKER)
            content: Dictionary containing update data
            tags: Optional list of tags for categorization
            
        Returns:
            The created ContextUpdate object
        """

        state = self._get_or_create_topic_state(self.current_topic)

        update = ContextUpdate(
            agent_name=agent_name,
            type=update_type,
            content=content,
            tags=tags or [],
        )
        state.updates.append(update)
        state.last_activity = datetime.now()

        # Keep the top-level view in sync with the current topic state
        self._sync_view_from_state(state)
        return update
    
    def get_agent_view(self, agent_name: str) -> Dict:
        """Get a filtered view of the context relevant to a specific agent.
        
        Args:
            agent_name: name of the agent requesting the view
            
        Returns:
            Dictionary containing relevant context information for the agent
        """

        state = self._get_or_create_topic_state(self.current_topic)

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
                    "type": u.type.value if isinstance(u.type, UpdateType) else u.type,
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
        since: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        update_type: Optional[UpdateType] = None
    ) -> List[ContextUpdate]:
        """Get recent updates, optionally filtered by time, agent, or type.
        
        Args:
            since: Optional datetime to filter updates after this time
            agent_name: Optional agent name to filter updates by agent
            update_type: Optional UpdateType enum to filter by (PROGRESS, FINDING, DECISION, ARTIFACT, BLOCKER)
            
        Returns:
            List of ContextUpdate objects matching the filters
        """

        state = self._get_or_create_topic_state(self.current_topic)
        filtered_updates = state.updates

        if since:
            filtered_updates = [u for u in filtered_updates if u.timestamp >= since]

        if agent_name:
            filtered_updates = [u for u in filtered_updates if u.agent_name == agent_name]

        if update_type:
            filtered_updates = [u for u in filtered_updates if u.type == update_type]

        return filtered_updates

    # --- Topic-aware helpers ---

    @property
    def known_topics(self) -> List[str]:
        """Return the list of known topics (configured or with active state)."""

        known = set(self._topic_keywords.keys()) | set(self.topics.keys())
        return sorted(known)

    def configure_topics(
        self,
        *,
        topic_keywords: Optional[Dict[str, List[str]]] = None,
        default_topic: Optional[str] = None,
    ) -> None:
        """Configure topic keyword groups and default topic.

        Args:
            topic_keywords: Optional mapping of topic name to list of keywords.
                If provided, replaces the existing configuration.
            default_topic: Optional new default topic name.
        """

        if topic_keywords is not None:
            self._topic_keywords = topic_keywords
        if default_topic is not None:
            self._default_topic = default_topic

    def detect_topic(self, message: str) -> str:
        """Detect topic from a user message using simple keyword matching.

        The first topic whose keyword appears (case-insensitive) in the message is
        returned. If none match, the default topic is used.
        """

        text = message.lower()
        for topic, keywords in self._topic_keywords.items():
            for kw in keywords:
                if kw and kw.lower() in text:
                    return topic
        return self._default_topic