"""Tests for the conversation summary feature in AgentMemory and AsyncAgentMemory."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from cortex.agent_memory import (
    AgentMemory,
    AsyncAgentMemory,
    AgentMemoryBank,
    AsyncAgentMemoryBank,
    DEFAULT_SUMMARY_PROMPT,
    _format_messages_for_summary,
    _build_default_summary_fn_sync,
    _build_default_summary_fn_async,
)
from cortex.message import UserMessage, AIMessage, SystemMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_round(user_text: str, ai_text: str):
    """Create a simple conversation round (list of messages)."""
    return [UserMessage(content=user_text), AIMessage(content=ai_text)]


def _custom_summary_fn(current_summary: str, messages: list) -> str:
    """Deterministic custom summarizer for testing."""
    new_info = "; ".join(m.content for m in messages)
    if current_summary:
        return f"{current_summary} | {new_info}"
    return new_info


async def _async_custom_summary_fn(current_summary: str, messages: list) -> str:
    """Async version of the deterministic custom summarizer."""
    new_info = "; ".join(m.content for m in messages)
    if current_summary:
        return f"{current_summary} | {new_info}"
    return new_info


# ===========================================================================
# AgentMemory (sync) tests
# ===========================================================================

class TestAgentMemorySummaryDisabled:
    """Verify that summary-disabled memory behaves exactly like the old implementation."""

    def test_no_summary_by_default(self):
        mem = AgentMemory(k=2)
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))
        mem.add_messages(_make_round("q2", "a2"))  # evicts round 1
        assert mem.get_summary() == ""
        loaded = mem.load_memory()
        # Should only have the last 2 rounds, no summary message
        assert len(loaded) == 4
        assert not any(isinstance(m, SystemMessage) for m in loaded)

    def test_is_empty_without_summary(self):
        mem = AgentMemory(k=2)
        assert mem.is_empty()
        mem.add_messages(_make_round("hi", "hello"))
        assert not mem.is_empty()


class TestAgentMemorySummaryEnabled:
    """Test the summary feature with a custom (deterministic) summarizer."""

    def test_summary_triggers_after_n_evictions(self):
        mem = AgentMemory(k=2, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=2)
        # Fill the window
        mem.add_messages(_make_round("hi", "hello"))       # round 0
        mem.add_messages(_make_round("q1", "a1"))           # round 1 — window full
        assert mem.get_summary() == ""

        # Eviction 1 (counter=1, threshold=2 → no summarization yet)
        mem.add_messages(_make_round("q2", "a2"))           # evicts round 0
        assert mem.get_summary() == ""
        assert mem._eviction_counter == 1

        # Eviction 2 (counter=2 → summarization fires)
        mem.add_messages(_make_round("q3", "a3"))           # evicts round 1
        assert mem._eviction_counter == 0  # reset after summarization
        assert mem.get_summary() != ""

    def test_summary_prepended_in_load_memory(self):
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1)
        mem.add_messages(_make_round("hi", "hello"))        # window full
        mem.add_messages(_make_round("q1", "a1"))           # evicts → summarize

        loaded = mem.load_memory()
        # First message should be the summary SystemMessage
        assert isinstance(loaded[0], SystemMessage)
        assert "Summary of earlier conversation" in loaded[0].content
        # Remaining messages are the current window
        assert loaded[1].content == "q1"
        assert loaded[2].content == "a1"

    def test_summary_merges_incrementally(self):
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1)
        mem.add_messages(_make_round("fact1", "ack1"))
        mem.add_messages(_make_round("fact2", "ack2"))  # evicts round 0 → summarize
        first_summary = mem.get_summary()
        assert "fact1" in first_summary

        mem.add_messages(_make_round("fact3", "ack3"))  # evicts round 1 → summarize
        second_summary = mem.get_summary()
        # The custom fn merges with " | ", so both should be present
        assert "fact2" in second_summary
        assert first_summary in second_summary  # old summary preserved

    def test_set_summary_manually(self):
        mem = AgentMemory(k=2)
        mem.set_summary("User prefers dark mode.")
        loaded = mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "dark mode" in loaded[0].content

    def test_is_empty_with_summary(self):
        mem = AgentMemory(k=2)
        assert mem.is_empty()
        mem.set_summary("something")
        assert not mem.is_empty()

    def test_summarize_every_n_default_is_3(self):
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn)
        assert mem.summarize_every_n == 3

        # Evictions 1 and 2 should NOT trigger summarization
        mem.add_messages(_make_round("r0", "a0"))
        mem.add_messages(_make_round("r1", "a1"))  # eviction 1
        assert mem.get_summary() == ""
        mem.add_messages(_make_round("r2", "a2"))  # eviction 2
        assert mem.get_summary() == ""

        # Eviction 3 should trigger
        mem.add_messages(_make_round("r3", "a3"))  # eviction 3
        summary = mem.get_summary()
        assert summary != ""
        # All 3 evicted rounds should be in the summary (buffered)
        assert "r0" in summary
        assert "r1" in summary
        assert "r2" in summary

    def test_eviction_buffer_captures_all_rounds(self):
        """Verify that evicted rounds between summarization runs are buffered,
        so no information is silently dropped."""
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=3)

        mem.add_messages(_make_round("important_fact_A", "ack_A"))
        mem.add_messages(_make_round("important_fact_B", "ack_B"))  # eviction 1
        mem.add_messages(_make_round("important_fact_C", "ack_C"))  # eviction 2
        # Neither eviction 1 nor 2 triggers summarization, but messages are buffered
        assert mem.get_summary() == ""
        assert len(mem._eviction_buffer) == 4  # 2 rounds × 2 msgs each

        mem.add_messages(_make_round("important_fact_D", "ack_D"))  # eviction 3 → summarize
        summary = mem.get_summary()
        # All three evicted rounds should appear in the summary
        assert "important_fact_A" in summary
        assert "important_fact_B" in summary
        assert "important_fact_C" in summary
        # Buffer should be cleared after summarization
        assert len(mem._eviction_buffer) == 0

    def test_buffered_messages_visible_in_load_memory(self):
        """Verify that not-yet-summarized buffered messages are included in
        load_memory() so the LLM never loses visibility of evicted info."""
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=3)

        mem.add_messages(_make_round("secret_code_42", "ack"))
        mem.add_messages(_make_round("q1", "a1"))  # eviction 1 — no summarization yet
        assert mem.get_summary() == ""
        assert len(mem._eviction_buffer) == 2

        loaded = mem.load_memory()
        # Should contain: buffer SystemMessage + current window (2 msgs)
        assert any("secret_code_42" in m.content for m in loaded)
        assert any(isinstance(m, SystemMessage) and "Recent conversation not yet in summary" in m.content for m in loaded)

    def test_summarization_failure_is_graceful(self):
        def _failing_fn(current_summary, messages):
            raise RuntimeError("LLM is down")

        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_failing_fn, summarize_every_n=1)
        mem.add_messages(_make_round("hi", "hello"))
        # Should not raise
        mem.add_messages(_make_round("q1", "a1"))
        assert mem.get_summary() == ""  # summary unchanged after failure


class TestAgentMemoryDefaultLLMSummarizer:
    """Test the default LLM-based summarizer wiring (mocked)."""

    def test_default_llm_is_created_lazily(self):
        mem = AgentMemory(k=1, enable_summary=True, summarize_every_n=1)
        assert mem.summary_llm is None
        assert mem._summary_fn_resolved is None

        # Trigger lazy init by accessing the fn
        with patch('cortex.agent_memory._build_default_summary_fn_sync') as mock_build:
            mock_build.return_value = _custom_summary_fn
            resolved = mem._get_summary_fn()
            assert mem.summary_llm is not None
            assert resolved is _custom_summary_fn
            mock_build.assert_called_once()

    def test_custom_llm_is_used(self):
        fake_llm = MagicMock()
        mem = AgentMemory(k=1, enable_summary=True, summary_llm=fake_llm, summarize_every_n=1)

        with patch('cortex.agent_memory._build_default_summary_fn_sync') as mock_build:
            mock_build.return_value = _custom_summary_fn
            mem._get_summary_fn()
            mock_build.assert_called_once_with(fake_llm)

    def test_custom_summary_fn_takes_precedence_over_llm(self):
        fake_llm = MagicMock()
        mem = AgentMemory(k=1, enable_summary=True, summary_fn=_custom_summary_fn, summary_llm=fake_llm, summarize_every_n=1)
        fn = mem._get_summary_fn()
        assert fn is _custom_summary_fn
        # LLM should never be used
        fake_llm.call.assert_not_called()


# ===========================================================================
# AsyncAgentMemory tests
# ===========================================================================

class TestAsyncAgentMemorySummary:
    """Mirror the sync tests for the async variant."""

    @pytest.mark.asyncio
    async def test_summary_triggers_after_n_evictions(self):
        mem = AsyncAgentMemory(k=2, enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=2)
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))
        assert mem.get_summary() == ""

        await mem.add_messages(_make_round("q2", "a2"))  # eviction 1
        assert mem.get_summary() == ""

        await mem.add_messages(_make_round("q3", "a3"))  # eviction 2 → summarize
        assert mem.get_summary() != ""
        assert mem._eviction_counter == 0

    @pytest.mark.asyncio
    async def test_summary_prepended_in_load_memory(self):
        mem = AsyncAgentMemory(k=1, enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=1)
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))

        loaded = await mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "Summary of earlier conversation" in loaded[0].content

    @pytest.mark.asyncio
    async def test_summary_merges_incrementally(self):
        mem = AsyncAgentMemory(k=1, enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=1)
        await mem.add_messages(_make_round("fact1", "ack1"))
        await mem.add_messages(_make_round("fact2", "ack2"))
        first_summary = mem.get_summary()
        assert "fact1" in first_summary

        await mem.add_messages(_make_round("fact3", "ack3"))
        second_summary = mem.get_summary()
        assert "fact2" in second_summary
        assert first_summary in second_summary

    @pytest.mark.asyncio
    async def test_set_summary_manually(self):
        mem = AsyncAgentMemory(k=2)
        mem.set_summary("User prefers dark mode.")
        loaded = await mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "dark mode" in loaded[0].content

    @pytest.mark.asyncio
    async def test_is_empty_with_summary(self):
        mem = AsyncAgentMemory(k=2)
        assert await mem.is_empty()
        mem.set_summary("something")
        assert not await mem.is_empty()

    @pytest.mark.asyncio
    async def test_eviction_buffer_captures_all_rounds(self):
        mem = AsyncAgentMemory(k=1, enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=3)

        await mem.add_messages(_make_round("fact_A", "ack_A"))
        await mem.add_messages(_make_round("fact_B", "ack_B"))  # eviction 1
        await mem.add_messages(_make_round("fact_C", "ack_C"))  # eviction 2
        assert mem.get_summary() == ""
        assert len(mem._eviction_buffer) == 4

        await mem.add_messages(_make_round("fact_D", "ack_D"))  # eviction 3 → summarize
        summary = mem.get_summary()
        assert "fact_A" in summary
        assert "fact_B" in summary
        assert "fact_C" in summary
        assert len(mem._eviction_buffer) == 0

    @pytest.mark.asyncio
    async def test_buffered_messages_visible_in_load_memory(self):
        mem = AsyncAgentMemory(k=1, enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=3)

        await mem.add_messages(_make_round("secret_code_42", "ack"))
        await mem.add_messages(_make_round("q1", "a1"))  # eviction 1
        assert mem.get_summary() == ""

        loaded = await mem.load_memory()
        assert any("secret_code_42" in m.content for m in loaded)
        assert any(isinstance(m, SystemMessage) and "Recent conversation not yet in summary" in m.content for m in loaded)

    @pytest.mark.asyncio
    async def test_summarization_failure_is_graceful(self):
        async def _failing_fn(current_summary, messages):
            raise RuntimeError("LLM is down")

        mem = AsyncAgentMemory(k=1, enable_summary=True, summary_fn=_failing_fn, summarize_every_n=1)
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))
        assert mem.get_summary() == ""

    @pytest.mark.asyncio
    async def test_default_async_llm_is_created_lazily(self):
        mem = AsyncAgentMemory(k=1, enable_summary=True, summarize_every_n=1)
        assert mem.summary_llm is None

        with patch('cortex.agent_memory._build_default_summary_fn_async') as mock_build:
            mock_build.return_value = _async_custom_summary_fn
            resolved = mem._get_summary_fn()
            assert mem.summary_llm is not None
            assert resolved is _async_custom_summary_fn
            mock_build.assert_called_once()


# ===========================================================================
# MemoryBank forwarding tests
# ===========================================================================

class TestAgentMemoryBankSummaryForwarding:
    """Verify that MemoryBank.get_agent_memory forwards summary kwargs."""

    def test_bank_forwards_summary_kwargs(self):
        bank = AgentMemoryBank()
        mem = bank.get_agent_memory(
            "test_agent", k=3,
            enable_summary=True,
            summary_fn=_custom_summary_fn,
            summarize_every_n=5,
        )
        assert isinstance(mem, AgentMemory)
        assert mem.enable_summary is True
        assert mem.summary_fn is _custom_summary_fn
        assert mem.summarize_every_n == 5

    def test_bank_returns_cached_memory(self):
        bank = AgentMemoryBank()
        mem1 = bank.get_agent_memory("agent_a", k=3, enable_summary=True)
        mem2 = bank.get_agent_memory("agent_a", k=10)  # kwargs ignored on cache hit
        assert mem1 is mem2


class TestAsyncAgentMemoryBankSummaryForwarding:

    @pytest.mark.asyncio
    async def test_bank_forwards_summary_kwargs(self):
        bank = AsyncAgentMemoryBank()
        mem = await bank.get_agent_memory(
            "test_agent", k=3,
            enable_summary=True,
            summary_fn=_async_custom_summary_fn,
            summarize_every_n=5,
        )
        assert isinstance(mem, AsyncAgentMemory)
        assert mem.enable_summary is True
        assert mem.summary_fn is _async_custom_summary_fn
        assert mem.summarize_every_n == 5


# ===========================================================================
# Helper function tests
# ===========================================================================

class TestHelpers:

    def test_format_messages_for_summary(self):
        msgs = [UserMessage(content="Hello"), AIMessage(content="Hi there")]
        result = _format_messages_for_summary(msgs)
        assert "User: Hello" in result
        assert "AI: Hi there" in result

    def test_default_summary_prompt_is_nonempty(self):
        assert len(DEFAULT_SUMMARY_PROMPT) > 50

    def test_build_default_summary_fn_sync_calls_llm(self):
        mock_llm = MagicMock()
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "Summary: user said hello"
        mock_llm.call.return_value = mock_ai_msg

        fn = _build_default_summary_fn_sync(mock_llm)
        result = fn("", [UserMessage(content="hello"), AIMessage(content="hi")])

        mock_llm.call.assert_called_once()
        assert result == "Summary: user said hello"

    def test_build_default_summary_fn_sync_with_existing_summary(self):
        mock_llm = MagicMock()
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "Merged summary"
        mock_llm.call.return_value = mock_ai_msg

        fn = _build_default_summary_fn_sync(mock_llm)
        result = fn("Old summary", [UserMessage(content="new info")])

        # Verify the user content includes the previous summary
        call_args = mock_llm.call.call_args
        user_msg = call_args[0][1][0]  # second positional arg, first message
        assert "Previous summary:" in user_msg.content
        assert "Old summary" in user_msg.content
        assert result == "Merged summary"

    @pytest.mark.asyncio
    async def test_build_default_summary_fn_async_calls_llm(self):
        mock_llm = MagicMock()
        mock_ai_msg = MagicMock()
        mock_ai_msg.content = "Async summary result"
        mock_llm.async_call = AsyncMock(return_value=mock_ai_msg)

        fn = _build_default_summary_fn_async(mock_llm)
        result = await fn("", [UserMessage(content="hello")])

        mock_llm.async_call.assert_called_once()
        assert result == "Async summary result"


# ===========================================================================
# Integration-style test: full conversation lifecycle
# ===========================================================================

class TestFullConversationLifecycle:
    """Simulate a multi-round conversation and verify summary behavior end-to-end."""

    def test_long_conversation_retains_info_via_summary(self):
        mem = AgentMemory(k=2, enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1)

        # Round 1: user shares their name
        mem.add_messages(_make_round("My name is Alice", "Nice to meet you, Alice!"))
        # Round 2
        mem.add_messages(_make_round("I like Python", "Python is great!"))
        # Round 3 — evicts round 1, summarizer captures it
        mem.add_messages(_make_round("What's the weather?", "It's sunny."))

        summary = mem.get_summary()
        assert "Alice" in summary

        loaded = mem.load_memory()
        # Summary message + 2 rounds × 2 messages = 5
        assert len(loaded) == 5
        assert isinstance(loaded[0], SystemMessage)

        # Round 4 — evicts round 2, summarizer captures it
        mem.add_messages(_make_round("Tell me a joke", "Why did the chicken..."))

        summary = mem.get_summary()
        assert "Python" in summary  # from round 2
        assert "Alice" in summary   # preserved from earlier summary
