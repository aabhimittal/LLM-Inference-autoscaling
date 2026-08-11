"""Build a fully wired :class:`InferenceService` from :class:`Settings`.

Keeping construction in one place means ``main.py`` never imports a concrete
backend, and switching from mock to vLLM (or memory to Redis) is an environment
variable rather than a code change.
"""
from __future__ import annotations

import logging

from .config import Settings
from .cost import Ledger, RedisLedger
from .providers import MockProvider, VLLMProvider
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


def build_service(settings: Settings | None = None) -> InferenceService:
    settings = settings or Settings.from_env()
    provider = build_provider(settings)
    ledger = build_ledger(settings)
    log.info(
        "starting service provider=%s ledger=%s replicas=%d-%d",
        settings.provider_backend,
        settings.ledger_backend,
        settings.min_replicas,
        settings.max_replicas,
    )
    return InferenceService(settings, provider=provider, ledger=ledger)
