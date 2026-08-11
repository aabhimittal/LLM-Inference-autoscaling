"""Redis ledger edge cases.

Run against ``fakeredis`` with Lua enabled, so the actual reserve/settle/release
scripts execute — this exercises the atomicity logic, not a Python stand-in.
Tests skip cleanly if fakeredis is not installed.
"""
from __future__ import annotations

import asyncio

import pytest

from app.cost import LedgerUnavailableError, RedisLedger, UserBudgetExceededError
from app.cost.redis_ledger import _sum_members

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa", reason="fakeredis needs lupa to execute Lua scripts")


@pytest.fixture
def ledger():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return RedisLedger(window_seconds=100.0, client=client)


# ---- core protocol --------------------------------------------------------


async def test_reserve_and_settle(ledger):
    res = await ledger.reserve("u1", 0.10, limit=1.0)
    assert await ledger.spent("u1") == pytest.approx(0.10)
    await ledger.settle(res, 0.04, model="small", input_tokens=1, output_tokens=1)
    assert await ledger.spent("u1") == pytest.approx(0.04)


async def test_release_refunds(ledger):
    res = await ledger.reserve("u1", 0.5, limit=1.0)
    await ledger.release(res)
    assert await ledger.spent("u1") == pytest.approx(0.0)


async def test_limit_is_enforced(ledger):
    await ledger.reserve("u1", 0.9, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        await ledger.reserve("u1", 0.2, limit=1.0)


async def test_lua_reserve_is_atomic_under_concurrency(ledger):
    """The whole reason for the Lua script: 50 racing reservations against a
    $1.00 cap must grant exactly 10, not 50."""
    results = await asyncio.gather(
        *[ledger.reserve("u1", 0.10, limit=1.0) for _ in range(50)],
        return_exceptions=True,
    )
    granted = [r for r in results if not isinstance(r, Exception)]
    assert len(granted) == 10
    assert await ledger.spent("u1") <= 1.0 + 1e-9


async def test_budgets_are_shared_across_ledger_instances(ledger):
    """Two 'replicas' pointing at the same Redis must share one budget.

    This is the bug the Redis backend exists to fix: with the in-memory ledger
    each replica would grant the full limit independently.
    """
    replica_a = ledger
    replica_b = RedisLedger(window_seconds=100.0, client=ledger._client)
    await replica_a.reserve("u1", 0.7, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        await replica_b.reserve("u1", 0.5, limit=1.0)


async def test_settle_is_idempotent(ledger):
    res = await ledger.reserve("u1", 0.10, limit=1.0)
    await ledger.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    await ledger.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    assert await ledger.spent("u1") == pytest.approx(0.05)


async def test_window_expiry_frees_budget(ledger):
    ledger._now = lambda: 1000.0
    res = await ledger.reserve("u1", 1.0, limit=1.0)
    await ledger.settle(res, 1.0, model="small", input_tokens=1, output_tokens=1)
    assert await ledger.spent("u1") == pytest.approx(1.0)
    ledger._now = lambda: 1200.0  # past the 100s window
    assert await ledger.spent("u1") == pytest.approx(0.0)
    await ledger.reserve("u1", 1.0, limit=1.0)


async def test_summary_reports_each_user(ledger):
    a = await ledger.reserve("alice", 0.2, limit=1.0)
    await ledger.settle(a, 0.2, model="small", input_tokens=1, output_tokens=1)
    b = await ledger.reserve("bob", 0.3, limit=1.0)
    await ledger.settle(b, 0.3, model="large", input_tokens=1, output_tokens=1)
    summary = await ledger.summary()
    assert summary["alice"] == pytest.approx(0.2)
    assert summary["bob"] == pytest.approx(0.3)


# ---- outage behaviour -----------------------------------------------------


class _DeadClient:
    """Every call fails, like a Redis that just went away."""

    async def script_load(self, *a, **kw):
        raise ConnectionError("redis is down")

    async def evalsha(self, *a, **kw):
        raise ConnectionError("redis is down")

    async def eval(self, *a, **kw):
        raise ConnectionError("redis is down")

    async def zremrangebyscore(self, *a, **kw):
        raise ConnectionError("redis is down")

    async def zrange(self, *a, **kw):
        raise ConnectionError("redis is down")

    def scan_iter(self, *a, **kw):
        raise ConnectionError("redis is down")

    async def aclose(self):
        return None


async def test_fail_closed_rejects_when_redis_is_down():
    """Default policy: a cost-control outage must not become an unmetered-spend
    incident."""
    led = RedisLedger(client=_DeadClient(), fail_closed=True)
    with pytest.raises(LedgerUnavailableError):
        await led.reserve("u1", 0.1, limit=1.0)


async def test_fail_open_allows_when_redis_is_down():
    """Opt-in policy: availability over accuracy. The request proceeds unmetered."""
    led = RedisLedger(client=_DeadClient(), fail_closed=False)
    res = await led.reserve("u1", 0.1, limit=1.0)
    assert res.user_id == "u1"


async def test_settle_failure_marks_reservation_closed():
    """A failed settle must not leave a reservation that a retry could
    double-charge."""
    led = RedisLedger(client=_DeadClient(), fail_closed=True)
    from app.cost import Reservation

    res = Reservation(user_id="u1", amount=0.1, created_at=0.0)
    with pytest.raises(LedgerUnavailableError):
        await led.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    assert res.closed is True


async def test_noscript_error_falls_back_to_eval():
    """After a Redis restart the cached script SHA is gone. The ledger must
    reload it instead of failing every request."""
    calls = {"evalsha": 0, "eval": 0}

    class _RestartedClient(_DeadClient):
        async def script_load(self, *a, **kw):
            return "deadbeef"

        async def evalsha(self, *a, **kw):
            calls["evalsha"] += 1
            raise RuntimeError("NOSCRIPT No matching script")

        async def eval(self, *a, **kw):
            calls["eval"] += 1
            return [1, "0"]

    led = RedisLedger(client=_RestartedClient())
    await led.reserve("u1", 0.1, limit=1.0)
    assert calls["evalsha"] == 1 and calls["eval"] == 1


# ---- data corruption ------------------------------------------------------


def test_corrupt_members_are_skipped_not_fatal():
    """A member written by an older schema (or a manual redis-cli edit) must not
    break billing for the whole user."""
    members = ["abc:0.5", "garbage", "def:notanumber", "ghi:0.25", ""]
    assert _sum_members(members) == pytest.approx(0.75)


def test_sum_members_handles_empty_and_none():
    assert _sum_members([]) == 0.0
    assert _sum_members(None) == 0.0
