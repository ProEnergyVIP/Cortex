from functools import cached_property
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid
from cortex import LLM, AsyncAgentMemoryBank, GPTModels
from cortex.message import AgentUsage
from cortex.backend import ReasoningEffort
from pydantic import BaseModel, Field


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


class AgentSystemContext(BaseModel):
    # Existing fields
    usage: Optional[AgentUsage] = None  # Usage tracking for the agent
    memory_bank: Optional[object] = None  # Memory bank for the agent
    
    # Shared context fields for multi-agent coordination
    mission: str = ""
    current_focus: str = ""
    progress: str = ""
    team_roles: Dict[str, str] = Field(default_factory=dict)  # agent_id -> role
    protocols: List[str] = Field(default_factory=list)
    updates: List[ContextUpdate] = Field(default_factory=list)
    artifacts: Dict[str, List[Dict]] = Field(default_factory=dict)
    last_activity: datetime = Field(default_factory=datetime.now)
    active_blockers: List[str] = Field(default_factory=list)

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
    
    # Shared context methods
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
        update = ContextUpdate(
            agent_name=agent_name,
            type=update_type,
            content=content,
            tags=tags or []
        )
        self.updates.append(update)
        self.last_activity = datetime.now()
        return update
    
    def get_agent_view(self, agent_name: str) -> Dict:
        """Get a filtered view of the context relevant to a specific agent.
        
        Args:
            agent_name: name of the agent requesting the view
            
        Returns:
            Dictionary containing relevant context information for the agent
        """
        return {
            "mission": self.mission,
            "current_focus": self.current_focus,
            "progress": self.progress,
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
                    "timestamp": u.timestamp.isoformat()
                }
                for u in self.updates[-10:]  # Last 10 updates
            ],
            "artifacts": self.artifacts,
            "active_blockers": self.active_blockers,
            "last_activity": self.last_activity.isoformat()
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
        filtered_updates = self.updates
        
        if since:
            filtered_updates = [u for u in filtered_updates if u.timestamp >= since]
        
        if agent_name:
            filtered_updates = [u for u in filtered_updates if u.agent_name == agent_name]
        
        if update_type:
            filtered_updates = [u for u in filtered_updates if u.type == update_type]
        
        return filtered_updates