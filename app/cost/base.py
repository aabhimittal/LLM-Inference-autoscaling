"""Ledger interface shared by the in-memory and Redis backends.

The ledger is the enforcement point for cost control. Its contract is a
three-phase protocol:

    reserve(user, projected_cost, limit)   -> hold budget BEFORE the call
    settle(reservation, actual_cost, ...)  -> replace the hold with real spend
    release(reservation)                   -> drop the hold, charge nothing

Reserving first is what makes the check atomic: without it, N concurrent
requests can each read a stale "spent" value, all pass the limit check, and
collectively overspend. This matters more, not less, with Redis — a distributed
fleet has many more racing writers than a single process.

All methods are async because the Redis backend does network I/O; the in-memory
backend implements the same signatures so the two are drop-in interchangeable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, runtime_checkable


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


class LedgerUnavailableError(Exception):
    """The backing store could not be reached and the policy is fail-closed."""


@dataclass
class Reservation:
    """A budget hold. ``id`` makes settle/release idempotent and unambiguous."""

    user_id: str
    amount: float
    created_at: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Set once the reservation has been settled or released, so a duplicate call
    # is a no-op instead of a double charge/refund.
    closed: bool = False


@dataclass
class UsageRecord:
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    at: float


@runtime_checkable
class LedgerBackend(Protocol):
    async def spent(self, user_id: str) -> float: ...

    async def reserve(
        self, user_id: str, amount: float, limit: float
    ) -> Reservation: ...

    async def settle(
        self,
        reservation: Reservation,
        actual_cost: float,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None: ...

    async def release(self, reservation: Reservation) -> None: ...

    async def summary(self) -> Dict[str, float]: ...

    async def records(self) -> List[UsageRecord]: ...

    async def aclose(self) -> None: ...
