from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Dict, ClassVar, Optional
import logging

from cortex.LLM import LLM
from cortex.message import Message, SystemMessage, UserMessage

logger = logging.getLogger(__name__)

# Default summarization prompt used by the built-in LLM summarizer
DEFAULT_SUMMARY_PROMPT = """You are a conversation summarizer. Your job is to produce a concise summary that retains all important information from the conversation.

Rules:
- Keep key facts, user preferences, decisions, names, numbers, and action items.
- Drop greetings, filler, and redundant back-and-forth.
- If a previous summary is provided, merge new information into it. Do not repeat what is already captured unless it has changed.
- Output ONLY the updated summary, nothing else."""


def _format_messages_for_summary(messages: List[Message]) -> str:
    """Format a list of messages into a readable string for the summarizer."""
    parts = []
    for msg in messages:
        role = type(msg).__name__.replace('Message', '')
        parts.append(f"{role}: {msg.content}")
    return "\n".join(parts)


def _build_default_summary_fn_sync(llm: LLM):
    """Build the default sync summarization function using the given LLM."""
    def _default_summary_fn(current_summary: str, messages: List[Message]) -> str:
        conversation_text = _format_messages_for_summary(messages)
        user_content = ""
        if current_summary:
            user_content += f"Previous summary:\n{current_summary}\n\n"
        user_content += f"New conversation to incorporate:\n{conversation_text}"

        sys_msg = SystemMessage(content=DEFAULT_SUMMARY_PROMPT)
        msgs = [UserMessage(content=user_content)]
        ai_msg = llm.call(sys_msg, msgs)
        return ai_msg.content.strip() if ai_msg and ai_msg.content else current_summary
    return _default_summary_fn


def _build_default_summary_fn_async(llm: LLM):
    """Build the default async summarization function using the given LLM."""
    async def _default_summary_fn(current_summary: str, messages: List[Message]) -> str:
        conversation_text = _format_messages_for_summary(messages)
        user_content = ""
        if current_summary:
            user_content += f"Previous summary:\n{current_summary}\n\n"
        user_content += f"New conversation to incorporate:\n{conversation_text}"

        sys_msg = SystemMessage(content=DEFAULT_SUMMARY_PROMPT)
        msgs = [UserMessage(content=user_content)]
        ai_msg = await llm.async_call(sys_msg, msgs)
        return ai_msg.content.strip() if ai_msg and ai_msg.content else current_summary
    return _default_summary_fn


@dataclass
class AgentMemory:
    """Base class for agent memory. Default implementation uses in-memory storage.
    
    Supports an optional conversation summary that periodically condenses older
    messages into a compact summary, preventing important information from being
    lost when the sliding window evicts old rounds.
    
    Args:
        k: Maximum number of conversation rounds to keep.
        enable_summary: If True, enable periodic conversation summarization.
        summary_fn: Custom sync summarization function with signature
            ``(current_summary: str, messages: List[Message]) -> str``.
            If not provided, a default LLM-based summarizer is used.
        summary_llm: LLM instance for the default summarizer. Defaults to
            ``LLM(model='gpt-5-nano')`` when enable_summary is True.
        summarize_every_n: Run summarization every N evictions. Default 3.
    """
    k: int
    chat_memory: deque = field(default_factory=deque)
    enable_summary: bool = False
    summary_fn: Optional[Callable[[str, List[Message]], str]] = None
    summary_llm: Optional[object] = None  # LLM instance; optional to avoid circular import at class level
    summarize_every_n: int = 3
    # Internal state
    _summary: str = field(default="", repr=False)
    _eviction_counter: int = field(default=0, repr=False)
    _eviction_buffer: List = field(default_factory=list, repr=False)
    _summary_fn_resolved: Optional[Callable] = field(default=None, repr=False)

    def _get_summary_fn(self) -> Callable[[str, List[Message]], str]:
        """Lazily resolve the summarization function."""
        if self._summary_fn_resolved is not None:
            return self._summary_fn_resolved
        if self.summary_fn is not None:
            self._summary_fn_resolved = self.summary_fn
        else:
            llm = self.summary_llm
            if llm is None:
                llm = LLM(model='gpt-5-nano')
                self.summary_llm = llm
            self._summary_fn_resolved = _build_default_summary_fn_sync(llm)
        return self._summary_fn_resolved

    def _run_summarization(self, buffered_msgs: List[Message]) -> None:
        """Run the summarization function on buffered evicted messages."""
        try:
            fn = self._get_summary_fn()
            self._summary = fn(self._summary, buffered_msgs)
            logger.debug("Conversation summary updated (%d chars)", len(self._summary))
        except Exception as e:
            logger.warning("Conversation summarization failed: %s", e)

    def add_messages(self, msgs: List[Message]) -> None:
        """Add messages to the memory."""
        self.chat_memory.append(msgs)

        if len(self.chat_memory) > self.k:
            evicted = self.chat_memory.popleft()
            if self.enable_summary:
                self._eviction_buffer.extend(evicted)
                self._eviction_counter += 1
                if self._eviction_counter >= self.summarize_every_n:
                    self._run_summarization(self._eviction_buffer)
                    self._eviction_buffer = []
                    self._eviction_counter = 0

    def load_memory(self) -> List[Message]:
        """Load all messages from memory.
        
        If a conversation summary exists, it is prepended as a SystemMessage
        so the agent has access to important context from earlier rounds.
        
        If there are buffered evicted messages not yet summarized, they are
        also prepended so the LLM never loses visibility of recent evictions.
        """
        msgs = [m for chat in self.chat_memory for m in chat]
        prefix = []
        if self._summary:
            prefix.append(SystemMessage(content=f"Summary of earlier conversation:\n{self._summary}"))
        if self._eviction_buffer:
            buffer_text = _format_messages_for_summary(self._eviction_buffer)
            prefix.append(SystemMessage(content=f"Recent conversation not yet in summary:\n{buffer_text}"))
        return prefix + msgs
    
    def get_summary(self) -> str:
        """Return the current conversation summary text."""
        return self._summary
    
    def set_summary(self, summary: str) -> None:
        """Manually set the conversation summary."""
        self._summary = summary

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.chat_memory) == 0 and not self._summary


@dataclass
class AsyncAgentMemory:
    """Asynchronous version of agent memory. Default implementation uses in-memory storage.
    
    Supports an optional conversation summary that periodically condenses older
    messages into a compact summary, preventing important information from being
    lost when the sliding window evicts old rounds.
    
    Args:
        k: Maximum number of conversation rounds to keep.
        enable_summary: If True, enable periodic conversation summarization.
        summary_fn: Custom async summarization function with signature
            ``async (current_summary: str, messages: List[Message]) -> str``.
            If not provided, a default LLM-based summarizer is used.
        summary_llm: LLM instance for the default summarizer. Defaults to
            ``LLM(model='gpt-5-nano')`` when enable_summary is True.
        summarize_every_n: Run summarization every N evictions. Default 3.
    """
    k: int
    chat_memory: deque = field(default_factory=deque)
    enable_summary: bool = False
    summary_fn: Optional[Callable] = None  # async callable
    summary_llm: Optional[object] = None
    summarize_every_n: int = 3
    # Internal state
    _summary: str = field(default="", repr=False)
    _eviction_counter: int = field(default=0, repr=False)
    _eviction_buffer: List = field(default_factory=list, repr=False)
    _summary_fn_resolved: Optional[Callable] = field(default=None, repr=False)

    def _get_summary_fn(self) -> Callable:
        """Lazily resolve the async summarization function."""
        if self._summary_fn_resolved is not None:
            return self._summary_fn_resolved
        if self.summary_fn is not None:
            self._summary_fn_resolved = self.summary_fn
        else:
            llm = self.summary_llm
            if llm is None:
                llm = LLM(model='gpt-5-nano')
                self.summary_llm = llm
            self._summary_fn_resolved = _build_default_summary_fn_async(llm)
        return self._summary_fn_resolved

    async def _run_summarization(self, buffered_msgs: List[Message]) -> None:
        """Run the async summarization function on buffered evicted messages."""
        try:
            fn = self._get_summary_fn()
            self._summary = await fn(self._summary, buffered_msgs)
            logger.debug("Conversation summary updated (%d chars)", len(self._summary))
        except Exception as e:
            logger.warning("Conversation summarization failed: %s", e)

    async def add_messages(self, msgs: List[Message]) -> None:
        """Add messages to the memory asynchronously."""
        self.chat_memory.append(msgs)

        if len(self.chat_memory) > self.k:
            evicted = self.chat_memory.popleft()
            if self.enable_summary:
                self._eviction_buffer.extend(evicted)
                self._eviction_counter += 1
                if self._eviction_counter >= self.summarize_every_n:
                    await self._run_summarization(self._eviction_buffer)
                    self._eviction_buffer = []
                    self._eviction_counter = 0
            
    async def load_memory(self) -> List[Message]:
        """Load all messages from memory asynchronously.
        
        If a conversation summary exists, it is prepended as a SystemMessage.
        
        If there are buffered evicted messages not yet summarized, they are
        also prepended so the LLM never loses visibility of recent evictions.
        """
        msgs = [m for chat in self.chat_memory for m in chat]
        prefix = []
        if self._summary:
            prefix.append(SystemMessage(content=f"Summary of earlier conversation:\n{self._summary}"))
        if self._eviction_buffer:
            buffer_text = _format_messages_for_summary(self._eviction_buffer)
            prefix.append(SystemMessage(content=f"Recent conversation not yet in summary:\n{buffer_text}"))
        return prefix + msgs
    
    def get_summary(self) -> str:
        """Return the current conversation summary text."""
        return self._summary
    
    def set_summary(self, summary: str) -> None:
        """Manually set the conversation summary."""
        self._summary = summary

    async def is_empty(self) -> bool:
        """Check if memory is empty asynchronously."""
        return len(self.chat_memory) == 0 and not self._summary

class AgentMemoryBank:
    """Memory bank for all agents for a user. Default implementation uses in-memory storage."""
    # Static mapping of user IDs to memory banks
    user_memories: ClassVar[Dict[str, 'AgentMemoryBank']] = {}
    
    def __init__(self):
        """Initialize an agent memory bank."""
        self.agent_memories: Dict[str, AgentMemory] = {}

    def get_agent_memory(self, agent_name: str, k: int = 5, **kwargs) -> AgentMemory:
        """Get memory for a named agent.
        
        Args:
            agent_name: Name of the agent.
            k: Maximum number of conversation rounds to keep.
            **kwargs: Additional keyword arguments forwarded to AgentMemory
                (e.g. enable_summary, summary_fn, summary_llm, summarize_every_n).
        """
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]

        mem = AgentMemory(k=k, **kwargs)
        self.agent_memories[agent_name] = mem
        return mem
    
    def reset_memory(self):
        '''Reset all agent memories in this bank.'''
        self.agent_memories = {}

    @classmethod
    def bank_for(cls, user_id: str, **kwargs) -> 'AgentMemoryBank':
        """Get user memory bank from global store. If it doesn't exist, create one."""
        if user_id in cls.user_memories:
            return cls.user_memories[user_id]

        memory_bank = AgentMemoryBank()
        cls.user_memories[user_id] = memory_bank
        return memory_bank

    @classmethod
    def clear_bank_for(cls, user_id: str) -> None:
        """Clear user memory bank from global store."""
        if user_id in cls.user_memories:
            del cls.user_memories[user_id]

    @classmethod
    def reset_all(cls) -> None:
        """Reset all memory banks for all users."""
        cls.user_memories.clear()

    @classmethod
    def is_active(cls, user_id: str) -> bool:
        """Check if user has memory bank in global store."""
        return user_id in cls.user_memories


class AsyncAgentMemoryBank:
    """Asynchronous memory bank for all agents for a user. Default implementation uses in-memory storage."""
    # Static mapping of user IDs to memory banks
    user_memories: ClassVar[Dict[str, 'AsyncAgentMemoryBank']] = {}
    
    def __init__(self):
        """Initialize an async agent memory bank."""
        self.agent_memories: Dict[str, AsyncAgentMemory] = {}

    async def get_agent_memory(self, agent_name: str, k: int = 5, **kwargs) -> AsyncAgentMemory:
        """Get memory for a named agent asynchronously.
        
        Args:
            agent_name: Name of the agent.
            k: Maximum number of conversation rounds to keep.
            **kwargs: Additional keyword arguments forwarded to AsyncAgentMemory
                (e.g. enable_summary, summary_fn, summary_llm, summarize_every_n).
        """
        if agent_name in self.agent_memories:
            return self.agent_memories[agent_name]

        mem = AsyncAgentMemory(k=k, **kwargs)
        self.agent_memories[agent_name] = mem
        return mem
    
    async def reset_memory(self):
        '''Reset all agent memories in this bank.'''
        self.agent_memories = {}

    @classmethod
    async def bank_for(cls, user_id: str, **kwargs) -> 'AsyncAgentMemoryBank':
        """Get user memory bank from global store asynchronously. If it doesn't exist, create one."""
        if user_id in cls.user_memories:
            return cls.user_memories[user_id]

        memory_bank = AsyncAgentMemoryBank()
        cls.user_memories[user_id] = memory_bank
        return memory_bank

    @classmethod
    async def clear_bank_for(cls, user_id: str) -> None:
        """Clear user memory bank from global store asynchronously."""
        if user_id in cls.user_memories:
            del cls.user_memories[user_id]

    @classmethod
    async def reset_all(cls) -> None:
        """Reset all memory banks for all users asynchronously."""
        cls.user_memories.clear()

    @classmethod
    async def is_active(cls, user_id: str) -> bool:
        """Check if user has memory bank in global store asynchronously."""
        return user_id in cls.user_memories
