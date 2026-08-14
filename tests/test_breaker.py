"""Circuit breaker edge cases.

A breaker that trips too eagerly removes healthy capacity; one that trips too
late lets a dead backend consume every worker slot. Both are outages.
"""
from __future__ import annotations

import pytest

from app.breaker import (
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
)


def _breaker(clock, **kw):
    return CircuitBreaker(
        name="vllm", failure_threshold=3, recovery_timeout_s=30.0,
        clock=lambda: clock[0], **kw
    )


def test_starts_closed_and_allows_traffic():
    b = _breaker([0.0])
    assert b.state is BreakerState.CLOSED
    assert b.allows()


def test_trips_open_after_threshold_failures():
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    assert b.state is BreakerState.OPEN
    assert not b.allows()


def test_does_not_trip_below_threshold():
    b = _breaker([0.0])
    b.record_failure()
    b.record_failure()
    assert b.state is BreakerState.CLOSED
    assert b.allows()


def test_a_success_resets_the_failure_run():
    """The threshold counts *consecutive* failures. Occasional isolated errors
    are normal and must not accumulate into a trip over hours."""
    b = _breaker([0.0])
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    b.record_failure()
    assert b.state is BreakerState.CLOSED


def test_open_breaker_fails_fast_without_calling_the_backend():
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    with pytest.raises(CircuitOpenError) as exc:
        b.check()
    assert exc.value.retry_after_s > 0


def test_moves_to_half_open_after_the_recovery_timeout():
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    assert not b.allows()
    clock[0] = 30.0
    assert b.allows()  # this caller is the probe
    assert b.state is BreakerState.HALF_OPEN


def test_half_open_admits_exactly_one_probe():
    """Releasing full load onto a recovering backend knocks it straight back
    down. Only one request is allowed to test the water."""
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock[0] = 30.0
    assert b.allows()      # probe
    assert not b.allows()  # everyone else waits
    assert not b.allows()


def test_successful_probe_closes_the_circuit():
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock[0] = 30.0
    b.allows()
    b.record_success()
    assert b.state is BreakerState.CLOSED
    assert b.allows() and b.allows()  # full traffic restored


def test_failed_probe_reopens_and_restarts_the_timer():
    """A backend that fails its probe must get another full recovery window,
    not be retried immediately."""
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    clock[0] = 30.0
    b.allows()
    b.record_failure()  # probe failed
    assert b.state is BreakerState.OPEN
    assert not b.allows()
    clock[0] = 45.0
    assert not b.allows(), "timer must restart from the reopen, not the first trip"
    clock[0] = 60.0
    assert b.allows()


def test_retry_after_shrinks_as_the_window_elapses():
    clock = [0.0]
    b = _breaker(clock)
    for _ in range(3):
        b.record_failure()
    assert b.retry_after_s() == pytest.approx(30.0)
    clock[0] = 20.0
    assert b.retry_after_s() == pytest.approx(10.0)
    clock[0] = 100.0
    assert b.retry_after_s() == 0.0


def test_closed_breaker_reports_no_retry_delay():
    assert _breaker([0.0]).retry_after_s() == 0.0


# ---- registry -------------------------------------------------------------


def test_registry_isolates_backends():
    """A dead 70B backend must not stop the 1.5B backend from serving."""
    clock = [0.0]
    reg = BreakerRegistry(failure_threshold=2, clock=lambda: clock[0])
    for _ in range(2):
        reg.get("large").record_failure()
    assert not reg.get("large").allows()
    assert reg.get("small").allows()


def test_registry_returns_the_same_breaker_for_a_name():
    reg = BreakerRegistry()
    assert reg.get("small") is reg.get("small")


def test_registry_reports_state_and_open_backends():
    clock = [0.0]
    reg = BreakerRegistry(failure_threshold=1, clock=lambda: clock[0])
    reg.get("large").record_failure()
    reg.get("small")
    assert reg.states() == {"large": "open", "small": "closed"}
    assert reg.open_backends() == ["large"]
