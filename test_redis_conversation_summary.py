"""Tests for the conversation summary feature in RedisAgentMemory and AsyncRedisAgentMemory."""

import pytest

from cortex.redis_agent_memory import (
    RedisAgentMemory,
    AsyncRedisAgentMemory,
    RedisAgentMemoryBank,
    AsyncRedisAgentMemoryBank,
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


# ---------------------------------------------------------------------------
# Fake Redis client (sync) — mimics a subset of redis.Redis
# ---------------------------------------------------------------------------

class FakeRedis:
    """Minimal in-memory fake of a sync Redis client for testing."""

    def __init__(self):
        self._store: dict[str, object] = {}  # key -> value (str or list)

    # -- list operations --
    def rpush(self, key, *values):
        lst = self._store.setdefault(key, [])
        lst.extend(values)
        return len(lst)

    def llen(self, key):
        lst = self._store.get(key, [])
        return len(lst)

    def ltrim(self, key, start, stop):
        lst = self._store.get(key, [])
        self._store[key] = lst[start:] if stop == -1 else lst[start:stop + 1]

    def lrange(self, key, start, stop):
        lst = self._store.get(key, [])
        if stop == -1:
            return lst[start:]
        return lst[start:stop + 1]

    def lindex(self, key, index):
        lst = self._store.get(key, [])
        if 0 <= index < len(lst):
            return lst[index]
        return None

    # -- string operations --
    def get(self, key):
        val = self._store.get(key)
        if isinstance(val, list):
            return None
        return val

    def set(self, key, value):
        self._store[key] = value

    def delete(self, key):
        self._store.pop(key, None)

    # -- set operations --
    def sadd(self, key, *members):
        s = self._store.setdefault(key, set())
        s.update(members)

    def exists(self, key):
        return 1 if key in self._store else 0

    def eval(self, script, numkeys, *args):
        # Simplified: just delete matching keys for the pattern
        import fnmatch
        pattern = args[0] if args else ""
        to_delete = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
        for k in to_delete:
            del self._store[k]
        return len(to_delete)


# ---------------------------------------------------------------------------
# Fake async Redis client — wraps FakeRedis with async methods
# ---------------------------------------------------------------------------

class FakeAsyncRedis:
    """Minimal in-memory fake of an async Redis client for testing."""

    def __init__(self):
        self._sync = FakeRedis()

    async def rpush(self, key, *values):
        return self._sync.rpush(key, *values)

    async def llen(self, key):
        return self._sync.llen(key)

    async def ltrim(self, key, start, stop):
        return self._sync.ltrim(key, start, stop)

    async def lrange(self, key, start, stop):
        return self._sync.lrange(key, start, stop)

    async def lindex(self, key, index):
        return self._sync.lindex(key, index)

    async def get(self, key):
        return self._sync.get(key)

    async def set(self, key, value):
        return self._sync.set(key, value)

    async def delete(self, key):
        return self._sync.delete(key)

    async def sadd(self, key, *members):
        return self._sync.sadd(key, *members)

    async def exists(self, key):
        return self._sync.exists(key)

    async def eval(self, script, numkeys, *args):
        return self._sync.eval(script, numkeys, *args)


# ===========================================================================
# RedisAgentMemory (sync) tests
# ===========================================================================

class TestRedisAgentMemorySummaryDisabled:
    """Verify that summary-disabled Redis memory behaves like the old implementation."""

    def test_no_summary_by_default(self):
        r = FakeRedis()
        mem = RedisAgentMemory(k=2, redis_client=r, key="test:agent")
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))
        mem.add_messages(_make_round("q2", "a2"))  # evicts round 0
        assert mem.get_summary() == ""
        loaded = mem.load_memory()
        assert len(loaded) == 4
        assert not any(isinstance(m, SystemMessage) for m in loaded)

    def test_is_empty_without_summary(self):
        r = FakeRedis()
        mem = RedisAgentMemory(k=2, redis_client=r, key="test:agent")
        assert mem.is_empty()
        mem.add_messages(_make_round("hi", "hello"))
        assert not mem.is_empty()


class TestRedisAgentMemorySummaryEnabled:
    """Test the summary feature with a custom (deterministic) summarizer."""

    def test_summary_triggers_after_n_evictions(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=2, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=2,
        )
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))
        assert mem.get_summary() == ""

        # Eviction 1
        mem.add_messages(_make_round("q2", "a2"))
        assert mem.get_summary() == ""
        assert mem._eviction_counter == 1

        # Eviction 2 → summarize
        mem.add_messages(_make_round("q3", "a3"))
        assert mem._eviction_counter == 0
        assert mem.get_summary() != ""

    def test_summary_prepended_in_load_memory(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1,
        )
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))  # evicts → summarize

        loaded = mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "Summary of earlier conversation" in loaded[0].content
        assert loaded[1].content == "q1"
        assert loaded[2].content == "a1"

    def test_summary_merges_incrementally(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1,
        )
        mem.add_messages(_make_round("fact1", "ack1"))
        mem.add_messages(_make_round("fact2", "ack2"))
        first_summary = mem.get_summary()
        assert "fact1" in first_summary

        mem.add_messages(_make_round("fact3", "ack3"))
        second_summary = mem.get_summary()
        assert "fact2" in second_summary
        assert first_summary in second_summary

    def test_summary_persisted_in_redis(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1,
        )
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))

        # Verify the summary is stored in Redis under the expected key
        raw = r.get("test:agent:summary")
        assert raw is not None
        assert "hi" in raw

    def test_set_summary_manually(self):
        r = FakeRedis()
        mem = RedisAgentMemory(k=2, redis_client=r, key="test:agent")
        mem.set_summary("User prefers dark mode.")
        loaded = mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "dark mode" in loaded[0].content
        # Also persisted in Redis
        assert r.get("test:agent:summary") == "User prefers dark mode."

    def test_set_summary_empty_deletes_key(self):
        r = FakeRedis()
        mem = RedisAgentMemory(k=2, redis_client=r, key="test:agent")
        mem.set_summary("something")
        assert r.get("test:agent:summary") is not None
        mem.set_summary("")
        assert r.get("test:agent:summary") is None

    def test_is_empty_with_summary(self):
        r = FakeRedis()
        mem = RedisAgentMemory(k=2, redis_client=r, key="test:agent")
        assert mem.is_empty()
        mem.set_summary("something")
        assert not mem.is_empty()

    def test_summarize_every_n_default_is_3(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn,
        )
        assert mem.summarize_every_n == 3

        mem.add_messages(_make_round("r0", "a0"))
        mem.add_messages(_make_round("r1", "a1"))  # eviction 1
        assert mem.get_summary() == ""
        mem.add_messages(_make_round("r2", "a2"))  # eviction 2
        assert mem.get_summary() == ""
        mem.add_messages(_make_round("r3", "a3"))  # eviction 3 → summarize
        assert mem.get_summary() != ""

    def test_summarization_failure_is_graceful(self):
        def _failing_fn(current_summary, messages):
            raise RuntimeError("LLM is down")

        r = FakeRedis()
        mem = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_failing_fn, summarize_every_n=1,
        )
        mem.add_messages(_make_round("hi", "hello"))
        mem.add_messages(_make_round("q1", "a1"))
        assert mem.get_summary() == ""


# ===========================================================================
# AsyncRedisAgentMemory tests
# ===========================================================================

class TestAsyncRedisAgentMemorySummary:

    @pytest.mark.asyncio
    async def test_summary_triggers_after_n_evictions(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(
            k=2, async_redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=2,
        )
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))
        assert await mem.get_summary() == ""

        await mem.add_messages(_make_round("q2", "a2"))  # eviction 1
        assert await mem.get_summary() == ""

        await mem.add_messages(_make_round("q3", "a3"))  # eviction 2 → summarize
        assert await mem.get_summary() != ""
        assert mem._eviction_counter == 0

    @pytest.mark.asyncio
    async def test_summary_prepended_in_load_memory(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(
            k=1, async_redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=1,
        )
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))

        loaded = await mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "Summary of earlier conversation" in loaded[0].content

    @pytest.mark.asyncio
    async def test_summary_merges_incrementally(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(
            k=1, async_redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=1,
        )
        await mem.add_messages(_make_round("fact1", "ack1"))
        await mem.add_messages(_make_round("fact2", "ack2"))
        first_summary = await mem.get_summary()
        assert "fact1" in first_summary

        await mem.add_messages(_make_round("fact3", "ack3"))
        second_summary = await mem.get_summary()
        assert "fact2" in second_summary
        assert first_summary in second_summary

    @pytest.mark.asyncio
    async def test_summary_persisted_in_redis(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(
            k=1, async_redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_async_custom_summary_fn, summarize_every_n=1,
        )
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))

        raw = r._sync.get("test:agent:summary")
        assert raw is not None
        assert "hi" in raw

    @pytest.mark.asyncio
    async def test_set_summary_manually(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(k=2, async_redis_client=r, key="test:agent")
        await mem.set_summary("User prefers dark mode.")
        loaded = await mem.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "dark mode" in loaded[0].content

    @pytest.mark.asyncio
    async def test_is_empty_with_summary(self):
        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(k=2, async_redis_client=r, key="test:agent")
        assert await mem.is_empty()
        await mem.set_summary("something")
        assert not await mem.is_empty()

    @pytest.mark.asyncio
    async def test_summarization_failure_is_graceful(self):
        async def _failing_fn(current_summary, messages):
            raise RuntimeError("LLM is down")

        r = FakeAsyncRedis()
        mem = AsyncRedisAgentMemory(
            k=1, async_redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_failing_fn, summarize_every_n=1,
        )
        await mem.add_messages(_make_round("hi", "hello"))
        await mem.add_messages(_make_round("q1", "a1"))
        assert await mem.get_summary() == ""


# ===========================================================================
# RedisAgentMemoryBank forwarding tests
# ===========================================================================

class TestRedisAgentMemoryBankSummaryForwarding:

    def test_bank_forwards_summary_kwargs(self):
        r = FakeRedis()
        bank = RedisAgentMemoryBank(redis_client=r)
        mem = bank.get_agent_memory(
            "test_agent", k=3,
            enable_summary=True,
            summary_fn=_custom_summary_fn,
            summarize_every_n=5,
        )
        assert isinstance(mem, RedisAgentMemory)
        assert mem.enable_summary is True
        assert mem.summary_fn is _custom_summary_fn
        assert mem.summarize_every_n == 5

    def test_bank_returns_cached_memory(self):
        r = FakeRedis()
        bank = RedisAgentMemoryBank(redis_client=r)
        mem1 = bank.get_agent_memory("agent_a", k=3, enable_summary=True)
        mem2 = bank.get_agent_memory("agent_a", k=10)
        assert mem1 is mem2


class TestAsyncRedisAgentMemoryBankSummaryForwarding:

    @pytest.mark.asyncio
    async def test_bank_forwards_summary_kwargs(self):
        r = FakeAsyncRedis()
        bank = AsyncRedisAgentMemoryBank(async_redis_client=r)
        mem = await bank.get_agent_memory(
            "test_agent", k=3,
            enable_summary=True,
            summary_fn=_async_custom_summary_fn,
            summarize_every_n=5,
        )
        assert isinstance(mem, AsyncRedisAgentMemory)
        assert mem.enable_summary is True
        assert mem.summary_fn is _async_custom_summary_fn
        assert mem.summarize_every_n == 5


# ===========================================================================
# Integration-style test: full conversation lifecycle with Redis
# ===========================================================================

class TestRedisFullConversationLifecycle:

    def test_long_conversation_retains_info_via_summary(self):
        r = FakeRedis()
        mem = RedisAgentMemory(
            k=2, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1,
        )

        mem.add_messages(_make_round("My name is Alice", "Nice to meet you, Alice!"))
        mem.add_messages(_make_round("I like Python", "Python is great!"))
        mem.add_messages(_make_round("What's the weather?", "It's sunny."))

        summary = mem.get_summary()
        assert "Alice" in summary

        loaded = mem.load_memory()
        assert len(loaded) == 5  # summary + 2 rounds × 2 msgs
        assert isinstance(loaded[0], SystemMessage)

        mem.add_messages(_make_round("Tell me a joke", "Why did the chicken..."))
        summary = mem.get_summary()
        assert "Python" in summary
        assert "Alice" in summary

    def test_summary_survives_new_memory_instance(self):
        """Summary is in Redis, so a fresh RedisAgentMemory with the same key sees it."""
        r = FakeRedis()
        mem1 = RedisAgentMemory(
            k=1, redis_client=r, key="test:agent",
            enable_summary=True, summary_fn=_custom_summary_fn, summarize_every_n=1,
        )
        mem1.add_messages(_make_round("My name is Bob", "Hi Bob!"))
        mem1.add_messages(_make_round("q1", "a1"))
        assert mem1.get_summary() != ""

        # Create a brand-new instance pointing at the same Redis key
        mem2 = RedisAgentMemory(k=1, redis_client=r, key="test:agent")
        loaded = mem2.load_memory()
        assert isinstance(loaded[0], SystemMessage)
        assert "Bob" in loaded[0].content
