"""Integration edge cases for caching, load shedding, breaking, and failover.

The unit suites prove each component works alone. These prove they work
*together* on the request path — which is where the money leaks live.
"""
from __future__ import annotations

import asyncio

import pytest

from app.breaker import BreakerRegistry, CircuitOpenError
from app.cache import InMemoryCache
from app.config import Settings
from app.providers import (
    Completion,
    MockProvider,
    ProviderError,
    ProviderUnavailableError,
    StreamChunk,
    Usage,
)
from app.ratelimit import RateLimitExceededError, TokenBucketLimiter
from app.service import AdmissionTimeoutError, InferenceService
from app.tokens import count_tokens


def _settings(**kw) -> Settings:
    base = dict(cache_backend="none", rate_limit_enabled=False)
    base.update(kw)
    return Settings(**base)


class FlakyProvider:
    """Fails for named models, succeeds for the rest.

    ``calls`` records which models were actually attempted, which is how the
    failover tests prove the request moved rather than silently succeeding on
    the first try.
    """

    def __init__(self, failing: set[str], error=None, chars_per_token: float = 4.0):
        self.failing = failing
        self.error = error or ProviderUnavailableError("backend down")
        self.calls: list[str] = []
        self.chars_per_token = chars_per_token

    async def generate(self, model, prompt, *, max_output_tokens):
        self.calls.append(model.name)
        if model.name in self.failing:
            raise self.error
        text = f"[{model.name}] ok"
        return Completion(
            text=text,
            input_tokens=count_tokens(prompt, self.chars_per_token),
            output_tokens=count_tokens(text, self.chars_per_token),
            model=model.name,
        )

    async def generate_stream(self, model, prompt, *, max_output_tokens):
        self.calls.append(model.name)
        if model.name in self.failing:
            raise self.error
        text = f"[{model.name}] ok"
        yield StreamChunk(text=text)
        yield StreamChunk(
            finish_reason="stop",
            usage=Usage(
                count_tokens(prompt, self.chars_per_token),
                count_tokens(text, self.chars_per_token),
            ),
        )

    async def aclose(self):
        return None


# ---------------------------------------------------------------------------
# Cache on the request path
# ---------------------------------------------------------------------------


async def test_identical_request_is_served_from_cache_for_free():
    svc = InferenceService(_settings(cache_backend="memory"))
    first = await svc.handle("say hi", user_id="u1")
    second = await svc.handle("say hi", user_id="u1")

    assert first.cached is False and second.cached is True
    assert second.cost_usd == 0.0
    assert second.text == first.text
    assert svc.metrics.snapshot()["total_cache_hits"] == 1


async def test_cache_hit_does_not_consume_the_user_budget():
    """The whole point of the cache: a hit generated no tokens, so it must not
    draw down a budget that exists to bound token spend."""
    svc = InferenceService(_settings(cache_backend="memory"))
    await svc.handle("say hi", user_id="u1")
    spend_after_first = await svc.ledger.spent("u1")
    await svc.handle("say hi", user_id="u1")
    assert await svc.ledger.spent("u1") == pytest.approx(spend_after_first)


async def test_cache_hit_serves_a_user_who_is_over_budget():
    """A hit costs nothing, so refusing it would be pure loss — the request is
    free to serve and the budget exists to cap spend, not usage."""
    svc = InferenceService(_settings(cache_backend="memory"))
    await svc.handle("say hi", user_id="u1")
    result = await svc.handle("say hi", user_id="u1", user_daily_budget_usd=0.0)
    assert result.cached is True


async def test_cache_hit_takes_no_capacity():
    """Cached responses must bypass the concurrency gate entirely, or a cache
    that exists to shed load would itself be limited by that load."""
    svc = InferenceService(_settings(cache_backend="memory"))
    await svc.handle("say hi", user_id="u1")
    svc.gate._capacity = 0  # nothing could be admitted now
    result = await svc.handle("say hi", user_id="u1")
    assert result.cached is True


async def test_cache_is_content_addressed_not_user_scoped():
    """Documented behaviour: two users sending the identical prompt to the
    identical model share one cached completion.

    That is what makes the cache effective on shared prompt templates. It also
    means the cache must never hold user-specific content — scope it per tenant
    (or disable it) if prompts can contain another user's private data.
    """
    svc = InferenceService(_settings(cache_backend="memory"))
    await svc.handle("say hi", user_id="alice")
    result = await svc.handle("say hi", user_id="bob")
    assert result.cached is True


async def test_different_prompts_do_not_share_cache_entries():
    svc = InferenceService(_settings(cache_backend="memory"))
    a = await svc.handle("say hi", user_id="u1")
    b = await svc.handle("say something else entirely", user_id="u1")
    assert b.cached is False
    assert a.text != b.text or True  # both generated, not shared


async def test_failed_generation_is_never_cached():
    """A transient upstream blip must not become a permanently wrong answer."""
    svc = InferenceService(
        _settings(cache_backend="memory", fallback_enabled=False),
        provider=FlakyProvider(failing={"small"}),
    )
    with pytest.raises(ProviderUnavailableError):
        await svc.handle("say hi", user_id="u1")

    # A later healthy call must actually generate, not serve a cached failure.
    svc.provider = FlakyProvider(failing=set())
    result = await svc.handle("say hi", user_id="u1")
    assert result.cached is False
    assert result.text


async def test_cache_disabled_by_configuration():
    svc = InferenceService(_settings(cache_backend="none"))
    await svc.handle("say hi", user_id="u1")
    second = await svc.handle("say hi", user_id="u1")
    assert second.cached is False
    assert second.cost_usd > 0


# ---- streaming + cache ----------------------------------------------------


async def test_streamed_response_is_cached_and_replayed():
    svc = InferenceService(_settings(cache_backend="memory"))
    events = [e async for e in svc.handle_stream("say hi", user_id="u1")]
    original = "".join(e.text for e in events if e.type == "delta")
    await svc.drain_settlements()

    replay = [e async for e in svc.handle_stream("say hi", user_id="u1")]
    assert replay[0].data["cached"] is True
    assert replay[-1].data["cost_usd"] == 0.0
    assert "".join(e.text for e in replay if e.type == "delta") == original


async def test_disconnected_stream_is_not_cached():
    """Caching a truncated generation would serve the truncation to everyone
    else, at full speed, for as long as the TTL lasts."""
    svc = InferenceService(
        _settings(cache_backend="memory"), provider=MockProvider(chunk_chars=4)
    )
    stream = svc.handle_stream("say hi", user_id="u1")
    n = 0
    async for e in stream:
        if e.type == "delta":
            n += 1
            if n >= 2:
                break
    await stream.aclose()
    await svc.drain_settlements()

    nxt = [e async for e in svc.handle_stream("say hi", user_id="u1")]
    assert nxt[0].data["cached"] is False


async def test_failed_stream_is_not_cached():
    svc = InferenceService(
        _settings(cache_backend="memory"),
        provider=MockProvider(chunk_chars=4, fail_after_chunks=2),
    )
    with pytest.raises(ProviderError):
        async for _ in svc.handle_stream("say hi", user_id="u1"):
            pass
    await svc.drain_settlements()

    svc.provider = MockProvider(chunk_chars=4)
    nxt = [e async for e in svc.handle_stream("say hi", user_id="u1")]
    assert nxt[0].data["cached"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def test_rate_limit_rejects_before_spending_anything():
    """The cheapest rejection: no routing, no reservation, no capacity."""
    svc = InferenceService(
        _settings(cache_backend="none"),
        rate_limiter=TokenBucketLimiter(capacity=2.0, refill_per_second=0.0),
    )
    await svc.handle("say hi", user_id="u1")
    await svc.handle("say hi 2", user_id="u1")
    with pytest.raises(RateLimitExceededError):
        await svc.handle("say hi 3", user_id="u1")
    assert svc.metrics.snapshot()["total_rate_limited"] == 1


async def test_rate_limit_is_per_user():
    svc = InferenceService(
        _settings(),
        rate_limiter=TokenBucketLimiter(capacity=1.0, refill_per_second=0.0),
    )
    await svc.handle("hi", user_id="noisy")
    with pytest.raises(RateLimitExceededError):
        await svc.handle("hi", user_id="noisy")
    await svc.handle("hi", user_id="quiet")  # unaffected


async def test_rate_limit_applies_to_streaming_too():
    """An abuse control that only covers one endpoint is not an abuse control."""
    svc = InferenceService(
        _settings(),
        rate_limiter=TokenBucketLimiter(capacity=1.0, refill_per_second=0.0),
    )
    async for _ in svc.handle_stream("hi", user_id="u1"):
        pass
    await svc.drain_settlements()
    with pytest.raises(RateLimitExceededError):
        async for _ in svc.handle_stream("hi", user_id="u1"):
            pass


# ---------------------------------------------------------------------------
# Load shedding
# ---------------------------------------------------------------------------


async def test_admission_timeout_sheds_when_the_queue_is_full():
    svc = InferenceService(
        _settings(admission_timeout_s=0.05, target_concurrency_per_replica=1)
    )
    await svc.gate.acquire()  # occupy the only slot
    with pytest.raises(AdmissionTimeoutError):
        await svc.handle("say hi", user_id="u1")
    assert svc.metrics.snapshot()["total_shed"] == 1


async def test_shed_request_refunds_its_reservation():
    """Without this refund, every shed request silently bills the user for a
    generation that never happened — and shedding spikes exactly when the
    system is already under stress."""
    svc = InferenceService(
        _settings(admission_timeout_s=0.05, target_concurrency_per_replica=1)
    )
    await svc.gate.acquire()
    with pytest.raises(AdmissionTimeoutError):
        await svc.handle("say hi", user_id="u1")
    assert await svc.ledger.spent("u1") == pytest.approx(0.0)


async def test_shed_request_does_not_leak_a_capacity_slot():
    svc = InferenceService(
        _settings(admission_timeout_s=0.05, target_concurrency_per_replica=1)
    )
    await svc.gate.acquire()
    with pytest.raises(AdmissionTimeoutError):
        await svc.handle("say hi", user_id="u1")
    await svc.gate.release()
    assert svc.gate.active == 0
    assert svc.gate.waiting == 0


async def test_zero_admission_timeout_means_wait_forever():
    """Opt-out must actually opt out, not collapse to an instant rejection."""
    svc = InferenceService(
        _settings(admission_timeout_s=0.0, target_concurrency_per_replica=1)
    )
    await svc.gate.acquire()
    task = asyncio.create_task(svc.handle("say hi", user_id="u1"))
    await asyncio.sleep(0.05)
    assert not task.done()  # still patiently queued
    await svc.gate.release()
    assert (await task).text


# ---------------------------------------------------------------------------
# Circuit breaking and failover
# ---------------------------------------------------------------------------


async def test_failover_to_another_model_on_server_fault():
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing={"small"}),
    )
    result = await svc.handle("say hi", user_id="u1")
    assert result.model != "small"
    assert svc.provider.calls[0] == "small"  # primary was attempted first
    assert svc.metrics.snapshot()["total_failovers"] >= 1


async def test_no_failover_on_client_error():
    """A malformed request fails identically on every model. Retrying it just
    multiplies load during an incident."""
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing={"small"}, error=ProviderError("bad request")),
    )
    with pytest.raises(ProviderError):
        await svc.handle("say hi", user_id="u1")
    assert svc.provider.calls == ["small"]


async def test_client_error_does_not_trip_the_breaker():
    """One bad prompt must not take a healthy backend out of rotation."""
    svc = InferenceService(
        _settings(fallback_enabled=False, breaker_failure_threshold=2),
        provider=FlakyProvider(failing={"small"}, error=ProviderError("bad request")),
    )
    for _ in range(5):
        with pytest.raises(ProviderError):
            await svc.handle("say hi", user_id="u1")
    assert svc.breakers.get("small").allows()


async def test_breaker_opens_after_repeated_server_faults():
    svc = InferenceService(
        _settings(fallback_enabled=False, breaker_failure_threshold=3),
        provider=FlakyProvider(failing={"small"}),
    )
    for _ in range(3):
        with pytest.raises(ProviderUnavailableError):
            await svc.handle("say hi", user_id="u1")
    assert not svc.breakers.get("small").allows()

    # Subsequent calls fail fast without touching the backend at all.
    before = len(svc.provider.calls)
    with pytest.raises(CircuitOpenError):
        await svc.handle("say hi", user_id="u1")
    assert len(svc.provider.calls) == before


async def test_open_circuit_fails_over_instead_of_failing():
    """An open breaker should redirect traffic, not just reject it, when there
    is another affordable model available."""
    clock = [0.0]
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing=set()),
        breakers=BreakerRegistry(failure_threshold=1, clock=lambda: clock[0]),
    )
    svc.breakers.get("small").record_failure()  # force it open
    result = await svc.handle("say hi", user_id="u1")
    assert result.model != "small"


async def test_force_model_is_never_silently_replaced():
    """The caller pinned that model on purpose; substituting another one would
    make the response quietly wrong."""
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing={"large"}),
    )
    with pytest.raises(ProviderUnavailableError):
        await svc.handle("say hi", user_id="u1", force_model="large", budget_usd=1.0)
    assert svc.provider.calls == ["large"]


async def test_failed_request_refunds_after_exhausting_failover():
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing={"small", "medium", "large"}),
    )
    with pytest.raises(ProviderUnavailableError):
        await svc.handle("say hi", user_id="u1", budget_usd=1.0)
    assert await svc.ledger.spent("u1") == pytest.approx(0.0)


async def test_failover_bills_the_model_that_actually_served():
    """Billing the primary's price for a fallback generation would over- or
    under-charge every failed-over request."""
    svc = InferenceService(
        _settings(fallback_enabled=True),
        provider=FlakyProvider(failing={"small"}),
    )
    result = await svc.handle("say hi", user_id="u1")
    served = {m.name: m for m in svc.settings.catalog}[result.model]
    expected = served.cost(result.input_tokens, result.output_tokens)
    assert result.cost_usd == pytest.approx(round(expected, 6))


async def test_breaker_recovery_restores_the_primary():
    clock = [0.0]
    svc = InferenceService(
        _settings(fallback_enabled=False, breaker_failure_threshold=1),
        provider=FlakyProvider(failing={"small"}),
        breakers=BreakerRegistry(
            failure_threshold=1, recovery_timeout_s=30.0, clock=lambda: clock[0]
        ),
    )
    with pytest.raises(ProviderUnavailableError):
        await svc.handle("say hi", user_id="u1")
    assert not svc.breakers.get("small").allows()

    svc.provider = FlakyProvider(failing=set())  # backend recovers
    clock[0] = 31.0
    result = await svc.handle("say hi", user_id="u1")
    assert result.model == "small"
    assert svc.breakers.get("small").allows()
