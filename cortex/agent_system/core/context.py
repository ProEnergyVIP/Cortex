from __future__ import annotations
from functools import cached_property
from typing import Optional, Dict, List, TYPE_CHECKING
from cortex import LLM, AsyncAgentMemoryBank, GPTModels
from cortex.message import AgentUsage
from cortex.backend import ReasoningEffort
from pydantic import BaseModel, PrivateAttr
from .whiteboard import Whiteboard, WhiteboardTopic

if TYPE_CHECKING:
    pass
 


class AgentSystemContext(BaseModel):
    # Existing fields
    usage: Optional[AgentUsage] = None  # Usage tracking for the agent
    memory_bank: Optional[object] = None  # Memory bank for the agent
    whiteboard: Optional[Whiteboard] = None

    # Topic routing configuration (not part of the serialized model)
    _topic_keywords: Dict[str, List[str]] = PrivateAttr(
        default_factory=lambda: {
            "solar": ["solar", "pv", "photovoltaic", "inverter", "panel"],
            "banking": ["bank", "loan", "credit", "account", "interest"],
            "general": [],
        }
    )
    _default_topic: str = PrivateAttr(default="general")
    _max_updates_per_topic: int = PrivateAttr(default=200)

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
    
    # --- Core Whiteboard methods (operate on current topic) ---

    def _get_or_create_topic_state(self, topic: str) -> WhiteboardTopic:
        """Get or create the WhiteboardTopic for a given topic.

        Raises if no whiteboard is configured.
        """
        if not self.whiteboard:
            raise ValueError("Whiteboard not configured on context")
        self.whiteboard.set_current_topic(topic)
        return self.whiteboard.topics[topic]

    def set_current_topic(self, topic: str) -> None:
        """Switch the current topic if a whiteboard is configured."""
        if self.whiteboard:
            self.whiteboard.set_current_topic(topic)

    def set_topic_for_message(self, message: str) -> str:
        """Detect the appropriate topic for a message and make it current.

        Returns the detected topic name.
        """

        topic = self.detect_topic(message)
        self.set_current_topic(topic)
        return topic

    # --- Topic-aware helpers ---

    @property
    def known_topics(self) -> List[str]:
        """Return the list of known topics (configured or with active state)."""
        wb_topics = set(self.whiteboard.topics.keys()) if self.whiteboard else set()
        known = set(self._topic_keywords.keys()) | wb_topics
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

    # Note: Whiteboard data should be accessed via context.whiteboard