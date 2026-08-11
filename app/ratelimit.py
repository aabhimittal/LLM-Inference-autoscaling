"""Per-user rate limiting via token bucket.

Spend budgets and rate limits solve different problems and you need both. A
dollar budget caps *total* damage but does nothing about a client that fires
10,000 cheap requests in a second — that request rate saturates the queue and
degrades everyone else long before the budget notices. Conversely a rate limit
alone lets a patient client burn the whole budget on expensive calls.

Token bucket is the right shape here because it permits **bursts** (real clients
are bursty; a strict requests-per-second limiter rejects traffic that the system
could easily absorb) while bounding the sustained rate.

    capacity      how large a burst is tolerated
    refill_rate   sustained requests per second

The Redis variant is atomic via Lua for the same reason the ledger is: without
it, replicas each maintain their own bucket and the effective limit multiplies
by the replica count.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple


class RateLimitExceededError(Exception):
    def __init__(self, user_id: str, retry_after_s: float):
        self.user_id = user_id
        self.retry_after_s = retry_after_s
        super().__init__(
            f"rate limit exceeded for {user_id!r}; retry in {retry_after_s:.2f}s"
        )


@dataclass
class TokenBucketLimiter:
    """In-process token bucket, one bucket per user.

    Buckets are refilled lazily on access rather than by a background timer:
    no task to supervise, and an idle user costs nothing until they return.
    """

    capacity: float = 20.0
    refill_per_second: float = 5.0
    clock: Callable[[], float] = time.monotonic
    # user -> (tokens remaining, last refill timestamp)
    _buckets: Dict[str, Tuple[float, float]] = field(default_factory=dict, repr=False)

    def _current(self, user_id: str, now: float) -> float:
        tokens, last = self._buckets.get(user_id, (self.capacity, now))
        elapsed = max(0.0, now - last)
        # Refill, never above capacity — otherwise a long idle period would
        # bank unlimited burst.
        return min(self.capacity, tokens + elapsed * self.refill_per_second)

    def check(self, user_id: str, cost: float = 1.0) -> None:
        """Consume ``cost`` tokens or raise :class:`RateLimitExceededError`."""
        now = self.clock()
        available = self._current(user_id, now)
        if available < cost:
            deficit = cost - available
            retry_after = (
                deficit / self.refill_per_second
                if self.refill_per_second > 0
                else float("inf")
            )
            # Persist the refill so the next call sees accurate state.
            self._buckets[user_id] = (available, now)
            raise RateLimitExceededError(user_id, retry_after)
        self._buckets[user_id] = (available - cost, now)

    def tokens_remaining(self, user_id: str) -> float:
        return self._current(user_id, self.clock())

    def reset(self, user_id: str) -> None:
        self._buckets.pop(user_id, None)


# Atomically refill and consume. Doing this in Lua is what makes the limit hold
# across replicas; a read-modify-write from Python lets concurrent callers each
# see the same token count and all pass.
# KEYS[1] = bucket hash;  ARGV: now, capacity, refill_rate, cost, ttl
# returns {allowed(0|1), tokens_remaining}
_CONSUME_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local rate = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then tokens = capacity end
if ts == nil then ts = now end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(capacity, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(tokens)}
"""


class RedisRateLimiter:
    """Fleet-wide token bucket.

    Fails **open** on a Redis outage: a rate limiter is an abuse control, not a
    correctness control. Rejecting all traffic because the limiter is unreachable
    converts a dependency blip into a full outage, which is a worse trade than
    briefly under-limiting. (The spend ledger makes the opposite choice, and for
    the opposite reason — see ``cost/redis_ledger.py``.)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        capacity: float = 20.0,
        refill_per_second: float = 5.0,
        namespace: str = "llm:rl",
        client: Any = None,
    ):
        if client is None:
            import redis.asyncio as aioredis

            client = aioredis.from_url(url, decode_responses=True)
        self._client = client
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.namespace = namespace
        self._sha = None
        self._now = time.time

    def _key(self, user_id: str) -> str:
        return f"{self.namespace}:{user_id}"

    async def _eval(self, keys, args):
        try:
            if self._sha is None:
                self._sha = await self._client.script_load(_CONSUME_LUA)
            return await self._client.evalsha(self._sha, len(keys), *keys, *args)
        except Exception as e:
            if "NOSCRIPT" in str(e).upper():
                self._sha = None
                return await self._client.eval(_CONSUME_LUA, len(keys), *keys, *args)
            raise

    async def check(self, user_id: str, cost: float = 1.0) -> None:
        ttl = int(max(60.0, self.capacity / max(self.refill_per_second, 0.001) * 2))
        try:
            allowed, remaining = await self._eval(
                [self._key(user_id)],
                [self._now(), self.capacity, self.refill_per_second, cost, ttl],
            )
        except Exception:
            return  # fail open
        if int(allowed) == 0:
            deficit = cost - float(remaining)
            retry_after = (
                deficit / self.refill_per_second
                if self.refill_per_second > 0
                else float("inf")
            )
            raise RateLimitExceededError(user_id, max(0.0, retry_after))

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass
