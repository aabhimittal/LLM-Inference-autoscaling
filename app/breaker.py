"""Circuit breaker for upstream model servers.

When a vLLM replica dies, every request still queues against it, waits out the
full timeout, and fails. That turns one dead backend into fleet-wide latency:
worker slots fill with doomed requests, the queue grows, the autoscaler adds
pods that also fail. The breaker cuts that loop by failing fast once a backend
has clearly stopped working, and by probing carefully before trusting it again.

States:

    CLOSED ──(failure_threshold consecutive failures)──► OPEN
      ▲                                                   │
      │                                          (recovery_timeout elapses)
      │                                                   ▼
      └──────────(probe succeeds)───────────────── HALF_OPEN
                                                          │
                                      (probe fails) ──────┘ back to OPEN

The important subtlety is **what counts as a failure**. A 400 from the model
server means *we* sent something invalid; tripping the breaker on client errors
would take a healthy backend out of rotation because of one malformed prompt.
Only server-side faults (unreachable, timeout, 5xx) count.

In HALF_OPEN exactly one probe is admitted. Letting the full load back in at
once is how a recovering backend gets knocked straight back down.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """The breaker is open; the call was rejected without touching the backend."""

    def __init__(self, name: str, retry_after_s: float):
        self.name = name
        self.retry_after_s = retry_after_s
        super().__init__(
            f"circuit for {name!r} is open; retry in {retry_after_s:.1f}s"
        )


@dataclass
class CircuitBreaker:
    """One breaker per backend (here: per model).

    Kept synchronous and lock-free: every transition is a single attribute
    write, and the cost of a rare double-probe under concurrency is far lower
    than the cost of serialising every request through a lock.
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    clock: Callable[[], float] = time.monotonic

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    _half_open_probe_taken: bool = field(default=False, repr=False)

    def allows(self) -> bool:
        """True if a call may proceed. Advances OPEN → HALF_OPEN when due."""
        if self.state is BreakerState.CLOSED:
            return True

        if self.state is BreakerState.OPEN:
            if self.clock() - self.opened_at >= self.recovery_timeout_s:
                self.state = BreakerState.HALF_OPEN
                self._half_open_probe_taken = True
                return True  # this caller is the probe
            return False

        # HALF_OPEN: admit exactly one probe, hold everyone else back.
        if not self._half_open_probe_taken:
            self._half_open_probe_taken = True
            return True
        return False

    def check(self) -> None:
        """Raise :class:`CircuitOpenError` if the call must not proceed."""
        if not self.allows():
            raise CircuitOpenError(self.name, self.retry_after_s())

    def retry_after_s(self) -> float:
        if self.state is BreakerState.CLOSED:
            return 0.0
        remaining = self.recovery_timeout_s - (self.clock() - self.opened_at)
        return max(0.0, remaining)

    def record_success(self) -> None:
        self.state = BreakerState.CLOSED
        self.consecutive_failures = 0
        self._half_open_probe_taken = False

    def record_failure(self) -> None:
        """Count a server-side fault. Callers must not report client errors here."""
        self.consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN:
            # The probe failed: the backend is still sick. Restart the timer.
            self._trip()
            return
        if self.consecutive_failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self.state = BreakerState.OPEN
        self.opened_at = self.clock()
        self._half_open_probe_taken = False


class BreakerRegistry:
    """Breakers keyed by backend name, created on first use.

    Per-model breakers matter because a fleet usually serves several models: a
    dead 70B backend should not stop the 1.5B backend from serving traffic.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.clock = clock
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name=name,
                failure_threshold=self.failure_threshold,
                recovery_timeout_s=self.recovery_timeout_s,
                clock=self.clock,
            )
            self._breakers[name] = breaker
        return breaker

    def states(self) -> Dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}

    def open_backends(self) -> list[str]:
        return [
            name
            for name, b in self._breakers.items()
            if b.state is BreakerState.OPEN
        ]
