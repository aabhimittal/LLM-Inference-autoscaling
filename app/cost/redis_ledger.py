"""Redis-backed distributed ledger.

Why Redis: the in-memory ledger enforces budgets per *process*. Run four
replicas and a user gets four times their budget. Redis makes the budget global
across the fleet.

Why Lua: the check-and-hold must be atomic. A read-then-write from Python has a
race window in which N replicas all read the same "spent" total and all decide
there is room. The reserve script below does prune → sum → compare → write
inside a single Redis execution, so exactly one caller can win the last dollar
of budget.

Data model (per user):

    ZSET  llm:spend:{user}   member "{entry_id}:{amount}"  score = unix_ts
    HASH  llm:usage:{user}   field "total" -> lifetime settled spend

The ZSET doubles as the rolling window: pruning is ``ZREMRANGEBYSCORE`` by
timestamp, and the key carries a TTL so idle users cost nothing.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import (
    LedgerUnavailableError,
    Reservation,
    UsageRecord,
    UserBudgetExceededError,
)

try:
    import redis.asyncio as aioredis

    _HAVE_REDIS = True
except Exception:  # pragma: no cover
    _HAVE_REDIS = False


# Atomically prune the window, sum live spend, and add a hold if it fits.
# KEYS[1] = spend zset
# ARGV: now, window_seconds, amount, limit, member
# returns {allowed(0|1), current_total}
_RESERVE_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local amount = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])
local member = ARGV[5]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)

local total = 0
local members = redis.call('ZRANGE', key, 0, -1)
for i = 1, #members do
  local sep = string.find(members[i], ':', 1, true)
  if sep then
    local v = tonumber(string.sub(members[i], sep + 1))
    if v then total = total + v end
  end
end

if total + amount > limit + 0.000000001 then
  return {0, tostring(total)}
end

redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, math.ceil(window) + 60)
return {1, tostring(total)}
"""

# Replace a hold with the real post-call cost, atomically.
# KEYS[1] = spend zset, KEYS[2] = usage hash
# ARGV: now, entry_id, actual_cost, window_seconds
_SETTLE_LUA = """
local key = KEYS[1]
local usage_key = KEYS[2]
local now = tonumber(ARGV[1])
local entry_id = ARGV[2]
local actual = tonumber(ARGV[3])
local window = tonumber(ARGV[4])

local members = redis.call('ZRANGE', key, 0, -1)
local prefix = entry_id .. ':'
local found = 0
for i = 1, #members do
  if string.sub(members[i], 1, #prefix) == prefix then
    redis.call('ZREM', key, members[i])
    found = 1
    break
  end
end

redis.call('ZADD', key, now, entry_id .. ':' .. string.format('%.10f', actual))
redis.call('EXPIRE', key, math.ceil(window) + 60)
redis.call('HINCRBYFLOAT', usage_key, 'total', actual)
redis.call('EXPIRE', usage_key, math.ceil(window) + 60)
return found
"""

# Drop a hold without charging.
_RELEASE_LUA = """
local key = KEYS[1]
local entry_id = ARGV[1]
local members = redis.call('ZRANGE', key, 0, -1)
local prefix = entry_id .. ':'
for i = 1, #members do
  if string.sub(members[i], 1, #prefix) == prefix then
    redis.call('ZREM', key, members[i])
    return 1
  end
end
return 0
"""


class RedisLedger:
    """Distributed budget ledger.

    ``fail_closed`` decides what happens when Redis is unreachable:

    * ``True`` (default) — reject the request. Cost control is the point of this
      component; a Redis outage must not become an unmetered-spend incident.
    * ``False`` — allow the request through unmetered. Choose this only when
      availability genuinely outranks spend accuracy, and alert on it.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        window_seconds: float = 24 * 3600.0,
        namespace: str = "llm",
        fail_closed: bool = True,
        client: Any = None,
        max_records: int = 1000,
    ):
        if client is None and not _HAVE_REDIS:
            raise RuntimeError("RedisLedger requires redis: pip install redis")
        self.window_seconds = window_seconds
        self.namespace = namespace
        self.fail_closed = fail_closed
        self.max_records = max_records
        self._owns_client = client is None
        self._client = client or aioredis.from_url(url, decode_responses=True)
        self._reserve_sha: Optional[str] = None
        self._settle_sha: Optional[str] = None
        self._release_sha: Optional[str] = None
        self._local_records: List[UsageRecord] = []
        self._now = time.time

    # ---- keys ----------------------------------------------------------

    def _spend_key(self, user_id: str) -> str:
        return f"{self.namespace}:spend:{user_id}"

    def _usage_key(self, user_id: str) -> str:
        return f"{self.namespace}:usage:{user_id}"

    # ---- script loading ------------------------------------------------

    async def _eval(self, script: str, sha_attr: str, keys: List[str], args: List[Any]):
        """Run a Lua script, loading it on first use and after a Redis restart.

        We cache the SHA and use EVALSHA; a NOSCRIPT error (Redis restarted or
        the script cache was flushed) transparently falls back to EVAL.
        """
        sha = getattr(self, sha_attr)
        try:
            if sha is None:
                sha = await self._client.script_load(script)
                setattr(self, sha_attr, sha)
            return await self._client.evalsha(sha, len(keys), *keys, *args)
        except Exception as e:
            if "NOSCRIPT" in str(e).upper():
                setattr(self, sha_attr, None)
                return await self._client.eval(script, len(keys), *keys, *args)
            raise

    # ---- API -----------------------------------------------------------

    async def spent(self, user_id: str) -> float:
        now = self._now()
        try:
            await self._client.zremrangebyscore(
                self._spend_key(user_id), "-inf", now - self.window_seconds
            )
            members = await self._client.zrange(self._spend_key(user_id), 0, -1)
        except Exception as e:
            if self.fail_closed:
                raise LedgerUnavailableError(f"redis unavailable: {e}") from e
            return 0.0
        return _sum_members(members)

    async def reserve(self, user_id: str, amount: float, limit: float) -> Reservation:
        now = self._now()
        reservation = Reservation(user_id=user_id, amount=amount, created_at=now)
        member = f"{reservation.id}:{amount:.10f}"
        try:
            allowed, current = await self._eval(
                _RESERVE_LUA,
                "_reserve_sha",
                [self._spend_key(user_id)],
                [now, self.window_seconds, amount, limit, member],
            )
        except Exception as e:
            if self.fail_closed:
                raise LedgerUnavailableError(f"redis unavailable: {e}") from e
            return reservation  # fail-open: unmetered, but the request proceeds
        if int(allowed) == 0:
            raise UserBudgetExceededError(user_id, float(current), limit, amount)
        return reservation

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
            return
        now = self._now()
        try:
            await self._eval(
                _SETTLE_LUA,
                "_settle_sha",
                [
                    self._spend_key(reservation.user_id),
                    self._usage_key(reservation.user_id),
                ],
                [now, reservation.id, actual_cost, self.window_seconds],
            )
        except Exception as e:
            # Never fail a completed request because bookkeeping failed — the
            # model call already happened and the user already got their answer.
            if self.fail_closed:
                # Still surface it for alerting, but only after marking closed so
                # a retry cannot double-charge.
                reservation.closed = True
                raise LedgerUnavailableError(f"redis unavailable: {e}") from e
        reservation.closed = True
        self._local_records.append(
            UsageRecord(
                user_id=reservation.user_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=actual_cost,
                at=now,
            )
        )
        if len(self._local_records) > self.max_records:
            del self._local_records[: -self.max_records]

    async def release(self, reservation: Reservation) -> None:
        if reservation.closed:
            return
        try:
            await self._eval(
                _RELEASE_LUA,
                "_release_sha",
                [self._spend_key(reservation.user_id)],
                [reservation.id],
            )
        except Exception as e:
            if self.fail_closed:
                reservation.closed = True
                raise LedgerUnavailableError(f"redis unavailable: {e}") from e
        reservation.closed = True

    async def summary(self) -> Dict[str, float]:
        """Live per-user spend read straight from Redis (fleet-wide totals)."""
        out: Dict[str, float] = {}
        try:
            pattern = f"{self.namespace}:spend:*"
            async for key in self._client.scan_iter(match=pattern):
                user = key.split(":", 2)[-1]
                members = await self._client.zrange(key, 0, -1)
                out[user] = round(_sum_members(members), 10)
        except Exception as e:
            if self.fail_closed:
                raise LedgerUnavailableError(f"redis unavailable: {e}") from e
        return out

    async def records(self) -> List[UsageRecord]:
        return list(self._local_records)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _sum_members(members: List[str]) -> float:
    """Sum amounts encoded in ZSET members, skipping anything malformed.

    A corrupt member (written by an older schema, or a manual redis-cli edit)
    must not take down billing for the whole user.
    """
    total = 0.0
    for m in members or []:
        _, _, raw = m.partition(":")
        try:
            total += float(raw)
        except (TypeError, ValueError):
            continue
    return total
