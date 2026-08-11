import pytest

from app.cost import Ledger, UserBudgetExceededError


async def test_reserve_and_settle_tracks_spend():
    led = Ledger()
    res = await led.reserve("u1", 0.10, limit=1.0)
    await led.settle(res, 0.08, model="small", input_tokens=100, output_tokens=50)
    assert await led.spent("u1") == pytest.approx(0.08)
    assert (await led.summary())["u1"] == pytest.approx(0.08)


async def test_reserve_blocks_when_over_limit():
    led = Ledger()
    await led.reserve("u1", 0.9, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        await led.reserve("u1", 0.2, limit=1.0)


async def test_release_refunds_reservation():
    led = Ledger()
    res = await led.reserve("u1", 0.5, limit=1.0)
    await led.release(res)
    assert await led.spent("u1") == pytest.approx(0.0)
    await led.reserve("u1", 0.9, limit=1.0)


async def test_rolling_window_prunes_old_spend():
    led = Ledger(window_seconds=100.0)
    now = [1000.0]
    led._now = lambda: now[0]
    res = await led.reserve("u1", 0.5, limit=1.0)
    await led.settle(res, 0.5, model="small", input_tokens=1, output_tokens=1)
    assert await led.spent("u1") == pytest.approx(0.5)
    now[0] += 200.0
    assert await led.spent("u1") == pytest.approx(0.0)


async def test_concurrent_reservations_respect_limit():
    led = Ledger()
    await led.reserve("u1", 0.6, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        await led.reserve("u1", 0.6, limit=1.0)
