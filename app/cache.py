"""Response cache — the cheapest token is the one you never generate.

Identical prompts are extremely common in production (retries, shared prompt
templates, polling clients, evaluation loops). Serving those from cache is the
single largest cost lever available: a hit costs **$0** and returns in
microseconds instead of seconds of GPU time.

Two things make a cache correct here rather than merely fast:

* **The key must cover everything that changes the output.** A key built from
  the prompt alone will happily serve a ``small``-model answer to a request that
  asked for ``large``, or a 50-token answer to one that asked for 2000. The key
  therefore includes model, prompt, and output ceiling.
* **Stampede protection.** When a cache entry expires under load, N concurrent
  identical requests all miss and all call the model. :class:`SingleFlight`
  collapses them into one generation whose result everybody shares — without it,
  a popular expired key produces a thundering herd against the GPU.

Errors are never cached: a transient upstream failure must not become a
permanently wrong answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol

CACHE_KEY_VERSION = "v1"


@dataclass
class CacheEntry:
    """A cached generation. Token counts are kept so a hit can still be
    reported (and observed) accurately even though it is billed at $0."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str


def make_cache_key(
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    namespace: str = CACHE_KEY_VERSION,
) -> str:
    """Hash every input that can change the output.

    The version prefix lets a deploy that changes generation behaviour
    invalidate the whole cache by bumping one constant, rather than needing a
    flush.
    """
    payload = json.dumps(
        {
            "v": namespace,
            "model": model,
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class ResponseCache(Protocol):
    async def get(self, key: str) -> Optional[CacheEntry]: ...

    async def set(self, key: str, entry: CacheEntry, ttl_s: float) -> None: ...

    async def aclose(self) -> None: ...


class InMemoryCache:
    """LRU + TTL cache, correct for a single process.

    Bounded on purpose: an unbounded response cache is a memory leak that looks
    like a feature until the pod is OOM-killed mid-generation.
    """

    def __init__(self, max_entries: int = 1024):
        self.max_entries = max(1, max_entries)
        self._data: "OrderedDict[str, tuple[CacheEntry, float]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._now = time.monotonic
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self._lock:
            found = self._data.get(key)
            if found is None:
                self.misses += 1
                return None
            entry, expires_at = found
            if expires_at <= self._now():
                del self._data[key]  # expired entries are evicted on read
                self.misses += 1
                return None
            self._data.move_to_end(key)  # LRU touch
            self.hits += 1
            return entry

    async def set(self, key: str, entry: CacheEntry, ttl_s: float) -> None:
        if ttl_s <= 0:
            return  # a non-positive TTL means "do not cache"
        async with self._lock:
            self._data[key] = (entry, self._now() + ttl_s)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict least-recently-used

    async def aclose(self) -> None:
        return None


class RedisCache:
    """Shared cache across replicas.

    A per-process cache has a hit rate that degrades with every replica you add
    (each pod warms its own copy). Redis makes one pod's generation available to
    the whole fleet, which is where the real savings appear at scale.

    Cache failures are always non-fatal: a cache is an optimisation, and a Redis
    outage must degrade latency and cost, never correctness or availability.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        namespace: str = "llm:cache",
        client: Any = None,
    ):
        if client is None:
            import redis.asyncio as aioredis

            client = aioredis.from_url(url, decode_responses=True)
        self._client = client
        self._owns_client = True
        self.namespace = namespace
        self.hits = 0
        self.misses = 0

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    async def get(self, key: str) -> Optional[CacheEntry]:
        try:
            raw = await self._client.get(self._key(key))
        except Exception:
            self.misses += 1
            return None  # degrade to a miss, never fail the request
        if not raw:
            self.misses += 1
            return None
        try:
            data = json.loads(raw)
            entry = CacheEntry(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            # Poisoned or schema-drifted entry: treat as a miss rather than
            # serving garbage or raising.
            self.misses += 1
            return None
        self.hits += 1
        return entry

    async def set(self, key: str, entry: CacheEntry, ttl_s: float) -> None:
        if ttl_s <= 0:
            return
        try:
            await self._client.set(
                self._key(key), json.dumps(asdict(entry)), ex=max(1, int(ttl_s))
            )
        except Exception:
            return  # best-effort

    async def aclose(self) -> None:
        if self._owns_client:
            try:
                await self._client.aclose()
            except Exception:
                pass


class SingleFlight:
    """Collapse concurrent identical work into one execution.

    The first caller for a key runs the factory; everyone arriving while it is
    in flight awaits the same result. Followers use ``shield`` so that a
    follower giving up (client disconnect) does not cancel the shared work the
    other callers are still waiting on.
    """

    def __init__(self) -> None:
        self._inflight: Dict[str, asyncio.Future] = {}

    def inflight_count(self) -> int:
        return len(self._inflight)

    async def run(self, key: str, factory: Callable[[], Awaitable[Any]]) -> Any:
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)

        task = asyncio.ensure_future(factory())
        self._inflight[key] = task
        try:
            return await task
        finally:
            self._inflight.pop(key, None)
