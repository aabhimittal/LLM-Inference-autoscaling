import pytest

from app.cost import Ledger, UserBudgetExceededError


def test_reserve_and_settle_tracks_spend():
    led = Ledger()
    res = led.reserve("u1", 0.10, limit=1.0)
    led.settle(res, 0.08, model="small", input_tokens=100, output_tokens=50)
    assert abs(led.spent("u1") - 0.08) < 1e-9
    assert led.summary()["u1"] == pytest.approx(0.08)


def test_reserve_blocks_when_over_limit():
    led = Ledger()
    led.reserve("u1", 0.9, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        led.reserve("u1", 0.2, limit=1.0)


def test_release_refunds_reservation():
    led = Ledger()
    res = led.reserve("u1", 0.5, limit=1.0)
    led.release(res)
    assert led.spent("u1") == pytest.approx(0.0)
    # After release the user can spend again.
    led.reserve("u1", 0.9, limit=1.0)


def test_rolling_window_prunes_old_spend():
    led = Ledger(window_seconds=100.0)
    now = [1000.0]
    led._now = lambda: now[0]  # type: ignore[assignment]
    res = led.reserve("u1", 0.5, limit=1.0)
    led.settle(res, 0.5, model="small", input_tokens=1, output_tokens=1)
    assert led.spent("u1") == pytest.approx(0.5)
    now[0] += 200.0  # move past the window
    assert led.spent("u1") == pytest.approx(0.0)


def test_concurrent_reservations_respect_limit():
    led = Ledger()
    led.reserve("u1", 0.6, limit=1.0)
    # Second reservation of 0.6 would exceed 1.0 -> must fail.
    with pytest.raises(UserBudgetExceededError):
        led.reserve("u1", 0.6, limit=1.0)
