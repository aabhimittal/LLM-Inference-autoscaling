"""Build a fully wired :class:`InferenceService` from :class:`Settings`.

Keeping construction in one place means ``main.py`` never imports a concrete
backend, and switching from mock to vLLM (or memory to Redis) is an environment
variable rather than a code change.
"""
from __future__ import annotations

import logging

from .cache import InMemoryCache, RedisCache
from .config import Settings
from .cost import Ledger, RedisLedger
from .providers import MockProvider, VLLMProvider
from .ratelimit import RedisRateLimiter, TokenBucketLimiter
from .service import InferenceService

log = logging.getLogger(__name__)


def build_provider(settings: Settings):
    backend = (settings.provider_backend or "mock").strip().lower()
    if backend == "mock":
        return MockProvider(chars_per_token=settings.chars_per_token)
    if backend == "vllm":
        return VLLMProvider(
            settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model_map=settings.vllm_model_map,
            connect_timeout_s=settings.vllm_connect_timeout_s,
            read_timeout_s=settings.vllm_read_timeout_s,
            max_retries=settings.vllm_max_retries,
            chars_per_token=settings.chars_per_token,
        )
    raise ValueError(f"unknown provider backend {backend!r} (use 'mock' or 'vllm')")


def build_ledger(settings: Settings):
    backend = (settings.ledger_backend or "memory").strip().lower()
    if backend == "memory":
        return Ledger()
    if backend == "redis":
        return RedisLedger(
            settings.redis_url,
            fail_closed=settings.redis_fail_closed,
        )
    raise ValueError(f"unknown ledger backend {backend!r} (use 'memory' or 'redis')")


def build_cache(settings: Settings):
    backend = (settings.cache_backend or "memory").strip().lower()
    if backend == "none":
        return None
    if backend == "memory":
        return InMemoryCache(max_entries=settings.cache_max_entries)
    if backend == "redis":
        # A shared cache is what makes the hit rate hold up as replicas are
        # added; a per-process cache dilutes with every pod.
        return RedisCache(settings.redis_url)
    raise ValueError(
        f"unknown cache backend {backend!r} (use 'none', 'memory', or 'redis')"
    )


def build_rate_limiter(settings: Settings):
    if not settings.rate_limit_enabled:
        return None
    if (settings.ledger_backend or "").strip().lower() == "redis":
        # Match the ledger: if budgets are fleet-wide, rate limits must be too,
        # or the effective limit multiplies by the replica count.
        return RedisRateLimiter(
            settings.redis_url,
            capacity=settings.rate_limit_burst,
            refill_per_second=settings.rate_limit_per_second,
        )
    return TokenBucketLimiter(
        capacity=settings.rate_limit_burst,
        refill_per_second=settings.rate_limit_per_second,
    )


def build_service(settings: Settings | None = None) -> InferenceService:
    settings = settings or Settings.from_env()
    log.info(
        "starting service provider=%s ledger=%s cache=%s replicas=%d-%d",
        settings.provider_backend,
        settings.ledger_backend,
        settings.cache_backend,
        settings.min_replicas,
        settings.max_replicas,
    )
    return InferenceService(
        settings,
        provider=build_provider(settings),
        ledger=build_ledger(settings),
        cache=build_cache(settings),
        rate_limiter=build_rate_limiter(settings),
    )
