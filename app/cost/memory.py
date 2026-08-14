"""In-memory ledger: correct for a single process, the default for tests."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .base import Reservation, UsageRecord, UserBudgetExceededError


@dataclass
class Ledger:
    """Tracks spend per user over a rolling window (24h by default).

    Entries are ``(timestamp, amount, entry_id)``; the id lets settle/release
    target the exact reservation even when two holds share a timestamp and
    amount.
    """

    window_seconds: float = 24 * 3600.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _spend: Dict[str, List[Tuple[float, float, str]]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _records: List[UsageRecord] = field(default_factory=list, repr=False)
    # Injectable clock so tests can drive the rolling window deterministically.
    _now = staticmethod(time.time)

    def _prune(self, user_id: str, now: float) -> None:
        cutoff = now - self.window_seconds
        self._spend[user_id] = [e for e in self._spend[user_id] if e[0] >= cutoff]

    async def spent(self, user_id: str) -> float:
        now = self._now()
        async with self._lock:
            self._prune(user_id, now)
            return sum(amount for _, amount, _ in self._spend[user_id])

    async def reserve(self, user_id: str, amount: float, limit: float) -> Reservation:
        now = self._now()
        async with self._lock:
            self._prune(user_id, now)
            current = sum(a for _, a, _ in self._spend[user_id])
            # Strictly-greater comparison with a small epsilon so a request that
            # lands exactly on the limit is allowed rather than rejected by
            # float representation error.
            if current + amount > limit + 1e-9:
                raise UserBudgetExceededError(user_id, current, limit, amount)
            reservation = Reservation(user_id=user_id, amount=amount, created_at=now)
            self._spend[user_id].append((now, amount, reservation.id))
            return reservation

    def _drop(self, user_id: str, entry_id: str) -> None:
        entries = self._spend[user_id]
        for i, (_, _, eid) in enumerate(entries):
            if eid == entry_id:
                entries.pop(i)
                return

    async def settle(
        self,
        reservation: Reservation,
        actual_cost: float,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if reservation.closed:
            return  # idempotent: never double-charge a reservation
        now = self._now()
        async with self._lock:
            self._drop(reservation.user_id, reservation.id)
            self._spend[reservation.user_id].append((now, actual_cost, reservation.id))
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
            reservation.closed = True

    async def release(self, reservation: Reservation) -> None:
        if reservation.closed:
            return
        async with self._lock:
            self._drop(reservation.user_id, reservation.id)
            reservation.closed = True

    async def records(self) -> List[UsageRecord]:
        async with self._lock:
            return list(self._records)

    async def summary(self) -> Dict[str, float]:
        async with self._lock:
            out: Dict[str, float] = defaultdict(float)
            for r in self._records:
                out[r.user_id] += r.cost
            return dict(out)

    async def aclose(self) -> None:
        return None
