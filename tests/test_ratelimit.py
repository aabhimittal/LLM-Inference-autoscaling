"""Rate limiter edge cases.

Spend budgets bound total damage; rate limits bound *instantaneous* damage.
A client firing thousands of cheap requests saturates the queue long before any
dollar budget notices.
"""
from __future__ import annotations

import asyncio

import pytest

from app.ratelimit import (
    RateLimitExceededError,
    RedisRateLimiter,
    TokenBucketLimiter,
)


def _limiter(clock, capacity=5.0, rate=1.0):
    return TokenBucketLimiter(
        capacity=capacity, refill_per_second=rate, clock=lambda: clock[0]
    )


def test_allows_a_burst_up_to_capacity():
    """Real clients are bursty. A strict per-second limiter would reject traffic
    the system could absorb."""
    lim = _limiter([0.0], capacity=5.0)
    for _ in range(5):
        lim.check("u1")
    with pytest.raises(RateLimitExceededError):
        lim.check("u1")


def test_refills_over_time():
    clock = [0.0]
    lim = _limiter(clock, capacity=5.0, rate=1.0)
    for _ in range(5):
        lim.check("u1")
    clock[0] = 2.0  # 2 tokens back
    lim.check("u1")
    lim.check("u1")
    with pytest.raises(RateLimitExceededError):
        lim.check("u1")


def test_idle_time_does_not_bank_unlimited_burst():
    """Refill must clamp at capacity, or a user idle overnight returns with an
    unbounded burst allowance."""
    clock = [0.0]
    lim = _limiter(clock, capacity=5.0, rate=1.0)
    lim.check("u1")
    clock[0] = 100_000.0
    for _ in range(5):
        lim.check("u1")
    with pytest.raises(RateLimitExceededError):
        lim.check("u1")


def test_buckets_are_per_user():
    """One noisy tenant must not throttle everyone else."""
    lim = _limiter([0.0], capacity=2.0)
    lim.check("noisy")
    lim.check("noisy")
    with pytest.raises(RateLimitExceededError):
        lim.check("noisy")
    lim.check("quiet")  # unaffected


def test_retry_after_reflects_the_actual_deficit():
    clock = [0.0]
    lim = _limiter(clock, capacity=1.0, rate=2.0)
    lim.check("u1")
    with pytest.raises(RateLimitExceededError) as exc:
        lim.check("u1")
    assert exc.value.retry_after_s == pytest.approx(0.5)  # 1 token / 2 per sec


def test_variable_cost_requests():
    """An expensive request can consume several tokens."""
    lim = _limiter([0.0], capacity=10.0)
    lim.check("u1", cost=7.0)
    with pytest.raises(RateLimitExceededError):
        lim.check("u1", cost=5.0)
    lim.check("u1", cost=3.0)


def test_cost_larger_than_capacity_is_always_rejected():
    """Must reject cleanly rather than deadlock waiting for a refill that can
    never satisfy it."""
    lim = _limiter([0.0], capacity=5.0)
    with pytest.raises(RateLimitExceededError):
        lim.check("u1", cost=10.0)


def test_zero_refill_rate_reports_infinite_retry_rather_than_dividing_by_zero():
    lim = TokenBucketLimiter(capacity=1.0, refill_per_second=0.0, clock=lambda: 0.0)
    lim.check("u1")
    with pytest.raises(RateLimitExceededError) as exc:
        lim.check("u1")
    assert exc.value.retry_after_s == float("inf")


def test_rejected_requests_do_not_consume_tokens():
    """A throttled caller must not be pushed further into debt by retrying —
    otherwise a retry loop makes recovery impossible."""
    clock = [0.0]
    lim = _limiter(clock, capacity=1.0, rate=1.0)
    lim.check("u1")
    for _ in range(10):
        with pytest.raises(RateLimitExceededError):
            lim.check("u1")
    clock[0] = 1.0
    lim.check("u1")  # one second, one token, one request


def test_reset_clears_a_users_bucket():
    lim = _limiter([0.0], capacity=1.0)
    lim.check("u1")
    lim.reset("u1")
    lim.check("u1")


# ---- Redis limiter --------------------------------------------------------


fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa", reason="fakeredis needs lupa to execute Lua scripts")


def _redis_limiter(**kw):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisRateLimiter(client=client, **kw)


async def test_redis_limiter_enforces_capacity():
    lim = _redis_limiter(capacity=3.0, refill_per_second=1.0)
    for _ in range(3):
        await lim.check("u1")
    with pytest.raises(RateLimitExceededError):
        await lim.check("u1")


async def test_redis_limiter_is_atomic_under_concurrency():
    """Without the Lua script, concurrent callers all read the same token count
    and all pass — the limit silently multiplies by the replica count."""
    lim = _redis_limiter(capacity=10.0, refill_per_second=0.0)
    results = await asyncio.gather(
        *[lim.check("u1") for _ in range(50)], return_exceptions=True
    )
    allowed = [r for r in results if r is None]
    assert len(allowed) == 10


async def test_redis_limiter_shares_one_bucket_across_instances():
    """Two 'replicas' must share one limit, which is the whole point."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    a = RedisRateLimiter(client=client, capacity=2.0, refill_per_second=0.0)
    b = RedisRateLimiter(client=client, capacity=2.0, refill_per_second=0.0)
    await a.check("u1")
    await b.check("u1")
    with pytest.raises(RateLimitExceededError):
        await a.check("u1")


async def test_redis_limiter_refills():
    lim = _redis_limiter(capacity=2.0, refill_per_second=10.0)
    now = [1000.0]
    lim._now = lambda: now[0]
    await lim.check("u1")
    await lim.check("u1")
    with pytest.raises(RateLimitExceededError):
        await lim.check("u1")
    now[0] += 1.0
    await lim.check("u1")


class _DeadRedis:
    async def script_load(self, *a, **kw):
        raise ConnectionError("redis down")

    async def evalsha(self, *a, **kw):
        raise ConnectionError("redis down")

    async def eval(self, *a, **kw):
        raise ConnectionError("redis down")

    async def aclose(self):
        return None


async def test_redis_limiter_fails_open_on_outage():
    """A rate limiter is abuse control, not correctness control. Rejecting all
    traffic because the limiter is unreachable turns a blip into an outage —
    the opposite trade-off from the spend ledger, deliberately."""
    lim = RedisRateLimiter(client=_DeadRedis(), capacity=1.0)
    for _ in range(10):
        await lim.check("u1")  # must not raise
