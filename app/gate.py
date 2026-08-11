"""A concurrency gate whose capacity can be raised *and lowered* at runtime.

``asyncio.Semaphore`` cannot shrink: releasing extra permits to grow it is easy,
but taking permits back requires acquiring them, which blocks. The autoscaler
needs both directions, so this gate tracks capacity explicitly and lets waiters
re-check it under a condition variable.

Shrinking is graceful: in-flight requests are never interrupted. The gate simply
stops admitting new ones until ``active`` falls below the new capacity.

The gate also owns the ``waiting`` counter, which *is* the queue-depth signal the
autoscaler scales on — the number of requests that wanted a worker and could not
get one is the most direct measure of being under-provisioned.
"""
from __future__ import annotations

import asyncio


class CapacityGate:
    def __init__(self, capacity: int):
        self._capacity = max(1, capacity)
        self._active = 0
        self._waiting = 0
        self._cond = asyncio.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> None:
        async with self._cond:
            if self._active < self._capacity:
                self._active += 1
                return
            self._waiting += 1
            try:
                await self._cond.wait_for(lambda: self._active < self._capacity)
                self._active += 1
            finally:
                self._waiting -= 1

    async def release(self) -> None:
        async with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify()

    async def resize(self, capacity: int) -> None:
        capacity = max(1, capacity)
        async with self._cond:
            grew = capacity > self._capacity
            self._capacity = capacity
            if grew:
                # Wake every waiter that the new headroom can admit.
                self._cond.notify(capacity)
