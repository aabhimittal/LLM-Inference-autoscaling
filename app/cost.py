"""Cost controls: a per-user spend ledger with a rolling daily window.

The ledger enforces the "cost control" half of the service. Routing decides what
a request *would* cost; the ledger decides whether the user is *allowed* to spend
it, reserves the budget before the call, and reconciles against actual usage
after the call returns.

The implementation is an in-memory, thread-safe ledger — good enough for a single
node and for tests. A production deployment would back this with Redis/Postgres,
but the interface would not change.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class UserBudgetExceededError(Exception):
    def __init__(self, user_id: str, spent: float, limit: float, requested: float):
        self.user_id = user_id
        self.spent = spent
        self.limit = limit
        self.requested = requested
        super().__init__(
            f"user {user_id!r} would spend ${spent + requested:.4f} exceeding "
            f"daily limit ${limit:.4f}"
        )


@dataclass
class Reservation:
    """A budget hold taken before a call. Reconcile it with the real cost after."""

    user_id: str
    amount: float
    created_at: float


@dataclass
class UsageRecord:
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    at: float


@dataclass
class Ledger:
    """Tracks spend per user over a rolling window (default 24h)."""

    window_seconds: float = 24 * 3600.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # user_id -> list of (timestamp, cost)
    _spend: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _records: List[UsageRecord] = field(default_factory=list, repr=False)
    # Injectable clock so tests can control the rolling window deterministically.
    _now = staticmethod(time.time)

    def _prune(self, user_id: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._spend[user_id] = [
            (t, c) for (t, c) in self._spend[user_id] if t >= cutoff
        ]

    def spent(self, user_id: str) -> float:
        now = self._now()
        with self._lock:
            self._prune(user_id, now)
            return sum(c for _, c in self._spend[user_id])

    def reserve(self, user_id: str, amount: float, limit: float) -> Reservation:
        """Atomically check-and-hold ``amount`` against the user's daily limit.

        Reserving *before* the call prevents concurrent requests from each
        passing the check and collectively blowing the budget.
        """
        now = self._now()
        with self._lock:
            self._prune(user_id, now)
            current = sum(c for _, c in self._spend[user_id])
            if current + amount > limit:
                raise UserBudgetExceededError(user_id, current, limit, amount)
            self._spend[user_id].append((now, amount))
            return Reservation(user_id=user_id, amount=amount, created_at=now)

    def settle(
        self,
        reservation: Reservation,
        actual_cost: float,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Replace a reservation's held amount with the real post-call cost."""
        now = self._now()
        with self._lock:
            entries = self._spend[reservation.user_id]
            # Remove the specific reservation hold (match by timestamp+amount).
            for i, (t, c) in enumerate(entries):
                if t == reservation.created_at and abs(c - reservation.amount) < 1e-12:
                    entries.pop(i)
                    break
            entries.append((now, actual_cost))
            self._records.append(
                UsageRecord(
                    user_id=reservation.user_id,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=actual_cost,
                    at=now,
                )
            )

    def release(self, reservation: Reservation) -> None:
        """Drop a reservation without charging (used when a call fails)."""
        with self._lock:
            entries = self._spend[reservation.user_id]
            for i, (t, c) in enumerate(entries):
                if t == reservation.created_at and abs(c - reservation.amount) < 1e-12:
                    entries.pop(i)
                    break

    def records(self) -> List[UsageRecord]:
        with self._lock:
            return list(self._records)

    def summary(self) -> Dict[str, float]:
        with self._lock:
            out: Dict[str, float] = defaultdict(float)
            for r in self._records:
                out[r.user_id] += r.cost
            return dict(out)
