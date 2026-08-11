"""Response cache edge cases.

A wrong cache is worse than no cache: it serves confidently incorrect answers
and hides the bug behind a fast response.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.cache import (
    CacheEntry,
    InMemoryCache,
    RedisCache,
    SingleFlight,
    make_cache_key,
)


def _entry(text="hello", model="small") -> CacheEntry:
    return CacheEntry(text=text, input_tokens=3, output_tokens=2, model=model)


# ---- key construction -----------------------------------------------------


def test_same_inputs_produce_the_same_key():
    a = make_cache_key(model="small", prompt="hi", max_output_tokens=100)
    b = make_cache_key(model="small", prompt="hi", max_output_tokens=100)
    assert a == b


def test_model_is_part_of_the_key():
    """Otherwise a cheap model's answer gets served to a request that paid for
    the expensive one."""
    a = make_cache_key(model="small", prompt="hi", max_output_tokens=100)
    b = make_cache_key(model="large", prompt="hi", max_output_tokens=100)
    assert a != b


def test_max_output_tokens_is_part_of_the_key():
    """A 50-token answer must not satisfy a request that asked for 2000."""
    a = make_cache_key(model="small", prompt="hi", max_output_tokens=50)
    b = make_cache_key(model="small", prompt="hi", max_output_tokens=2000)
    assert a != b


def test_whitespace_and_case_differences_are_distinct_keys():
    """Normalising prompts would be a correctness bug: whitespace and case can
    change a model's output."""
    keys = {
        make_cache_key(model="small", prompt=p, max_output_tokens=10)
        for p in ["hi", "Hi", " hi", "hi "]
    }
    assert len(keys) == 4


def test_unicode_prompts_hash_without_error():
    k = make_cache_key(model="small", prompt="🙂 日本語 ünïcödé", max_output_tokens=10)
    assert k.startswith("v1:")


def test_namespace_bump_invalidates_everything():
    a = make_cache_key(model="small", prompt="hi", max_output_tokens=10)
    b = make_cache_key(model="small", prompt="hi", max_output_tokens=10, namespace="v2")
    assert a != b


# ---- in-memory behaviour --------------------------------------------------


async def test_hit_returns_the_stored_entry():
    c = InMemoryCache()
    await c.set("k", _entry(), ttl_s=60)
    got = await c.get("k")
    assert got is not None and got.text == "hello"
    assert c.hits == 1


async def test_miss_on_unknown_key():
    c = InMemoryCache()
    assert await c.get("nope") is None
    assert c.misses == 1


async def test_entries_expire_after_ttl():
    c = InMemoryCache()
    clock = [1000.0]
    c._now = lambda: clock[0]
    await c.set("k", _entry(), ttl_s=10)
    clock[0] += 9.9
    assert await c.get("k") is not None
    clock[0] += 0.2  # now past the TTL
    assert await c.get("k") is None


async def test_zero_or_negative_ttl_does_not_cache():
    """A caller signalling 'do not cache' must be obeyed, not rounded up."""
    c = InMemoryCache()
    await c.set("k", _entry(), ttl_s=0)
    assert await c.get("k") is None
    await c.set("k", _entry(), ttl_s=-5)
    assert await c.get("k") is None


async def test_lru_eviction_bounds_memory():
    """An unbounded response cache is a memory leak that ends in an OOM kill."""
    c = InMemoryCache(max_entries=3)
    for i in range(5):
        await c.set(f"k{i}", _entry(text=str(i)), ttl_s=60)
    assert await c.get("k0") is None  # evicted
    assert await c.get("k1") is None
    assert await c.get("k4") is not None


async def test_reading_an_entry_makes_it_survive_eviction():
    c = InMemoryCache(max_entries=2)
    await c.set("a", _entry(), ttl_s=60)
    await c.set("b", _entry(), ttl_s=60)
    await c.get("a")  # touch a, so b becomes least-recently-used
    await c.set("c", _entry(), ttl_s=60)
    assert await c.get("a") is not None
    assert await c.get("b") is None


# ---- stampede protection --------------------------------------------------


async def test_single_flight_collapses_concurrent_identical_work():
    """Ten simultaneous misses on one key must produce one generation, not ten.

    This is what stops an expired popular key from stampeding the GPU.
    """
    sf = SingleFlight()
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        await asyncio.sleep(0.01)
        return "result"

    results = await asyncio.gather(*[sf.run("k", factory) for _ in range(10)])
    assert results == ["result"] * 10
    assert calls["n"] == 1


async def test_single_flight_releases_the_key_after_completion():
    sf = SingleFlight()

    async def factory():
        return 1

    await sf.run("k", factory)
    assert sf.inflight_count() == 0
    await sf.run("k", factory)  # a later call runs fresh


async def test_single_flight_propagates_errors_to_all_waiters():
    """Followers must see the failure, not hang or get a bogus success."""
    sf = SingleFlight()

    async def factory():
        await asyncio.sleep(0.01)
        raise RuntimeError("upstream died")

    results = await asyncio.gather(
        *[sf.run("k", factory) for _ in range(5)], return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert sf.inflight_count() == 0


async def test_follower_cancellation_does_not_kill_the_shared_work():
    """A client disconnecting must not cancel the generation other clients are
    still waiting on."""
    sf = SingleFlight()
    started = asyncio.Event()

    async def factory():
        started.set()
        await asyncio.sleep(0.05)
        return "done"

    leader = asyncio.create_task(sf.run("k", factory))
    await started.wait()
    follower = asyncio.create_task(sf.run("k", factory))
    await asyncio.sleep(0)
    follower.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower
    assert await leader == "done"


async def test_different_keys_run_independently():
    sf = SingleFlight()
    calls = {"n": 0}

    async def factory():
        calls["n"] += 1
        return calls["n"]

    await asyncio.gather(sf.run("a", factory), sf.run("b", factory))
    assert calls["n"] == 2


# ---- Redis cache ----------------------------------------------------------


class _FakeRedis:
    def __init__(self, fail: bool = False):
        self.data = {}
        self.fail = fail

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        if self.fail:
            raise ConnectionError("redis down")
        self.data[key] = value

    async def aclose(self):
        return None


async def test_redis_cache_roundtrip():
    c = RedisCache(client=_FakeRedis())
    await c.set("k", _entry(), ttl_s=60)
    got = await c.get("k")
    assert got is not None and got.model == "small"


async def test_redis_outage_degrades_to_a_miss_not_an_error():
    """A cache is an optimisation. Losing it must cost latency and money, never
    correctness or availability."""
    c = RedisCache(client=_FakeRedis(fail=True))
    assert await c.get("k") is None
    await c.set("k", _entry(), ttl_s=60)  # must not raise


async def test_poisoned_cache_entry_is_treated_as_a_miss():
    """A schema change or a manual redis-cli edit must not serve garbage or
    crash the request path."""
    fake = _FakeRedis()
    c = RedisCache(client=fake)
    fake.data[c._key("k")] = "{not valid json"
    assert await c.get("k") is None
    fake.data[c._key("k")] = json.dumps({"unexpected": "shape"})
    assert await c.get("k") is None
