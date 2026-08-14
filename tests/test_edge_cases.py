"""Industrial edge cases.

These are the failure modes that show up in production but never in a happy-path
demo: money leaking on disconnects, budgets racing across concurrent requests,
upstream servers that hang or lie, and inputs that no one designed for.

Each test names the real-world incident it guards against.
"""
from __future__ import annotations

import asyncio

import pytest

from app.autoscaler import Autoscaler, LoadSample
from app.complexity import estimate_complexity
from app.config import Complexity, ModelSpec, Settings
from app.cost import Ledger, UserBudgetExceededError
from app.gate import CapacityGate
from app.providers import MockProvider, ProviderError
from app.router import BudgetExceededError, ContextTooLargeError, route
from app.service import InferenceService
from app.tokens import count_tokens


def _settings(**kw) -> Settings:
    return Settings(**kw)


# ---------------------------------------------------------------------------
# Billing integrity — the expensive class of bug
# ---------------------------------------------------------------------------


async def test_client_disconnect_midstream_still_bills_partial_output():
    """A user who hangs up after 3 chunks still consumed GPU time upstream.

    If billing only happened on clean completion, anyone could stream forever and
    disconnect to get it free.
    """
    svc = InferenceService(_settings(), provider=MockProvider(chunk_chars=4))
    stream = svc.handle_stream("say hi", user_id="u1")

    seen = 0
    async for event in stream:
        if event.type == "delta":
            seen += 1
            if seen >= 3:
                break  # simulate the client going away mid-generation
    await stream.aclose()

    await svc.drain_settlements()
    spend = (await svc.ledger.summary()).get("u1", 0.0)
    assert spend > 0, "partial stream must still be billed"


async def test_disconnect_releases_capacity():
    """A disconnect must not leak a slot in the concurrency gate.

    Leaked slots are silent: capacity erodes request by request until the service
    deadlocks with an idle GPU.
    """
    svc = InferenceService(_settings(), provider=MockProvider(chunk_chars=4))
    stream = svc.handle_stream("say hi", user_id="u1")
    await stream.__anext__()  # "start" — gate is now held
    assert svc.gate.active == 1
    await stream.aclose()
    await svc.drain_settlements()
    assert svc.gate.active == 0


async def test_provider_error_midstream_bills_only_what_was_produced():
    """An upstream crash after N tokens should charge for N tokens, not the
    full reservation and not zero."""
    svc = InferenceService(
        _settings(),
        provider=MockProvider(chunk_chars=4, fail_after_chunks=2),
    )
    events = []
    with pytest.raises(ProviderError):
        async for e in svc.handle_stream("say hi", user_id="u1"):
            events.append(e)

    assert events[-1].type == "error"
    await svc.drain_settlements()
    spend = (await svc.ledger.summary()).get("u1", 0.0)
    assert spend > 0  # billed for the partial output
    # ...but less than a full-length generation would have cost.
    full = InferenceService(_settings(), provider=MockProvider(chunk_chars=4))
    res = await full.handle("say hi", user_id="u2")
    assert spend < res.cost_usd


async def test_failed_buffered_call_refunds_the_reservation():
    """A provider that dies before producing anything must cost the user $0."""
    svc = InferenceService(
        _settings(), provider=MockProvider(raise_error=ProviderError("boom"))
    )
    with pytest.raises(ProviderError):
        await svc.handle("say hi", user_id="u1")
    assert await svc.ledger.spent("u1") == pytest.approx(0.0)


async def test_settle_is_idempotent():
    """Retry logic that settles twice must not double-charge."""
    led = Ledger()
    res = await led.reserve("u1", 0.10, limit=1.0)
    await led.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    await led.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    assert await led.spent("u1") == pytest.approx(0.05)


async def test_release_after_settle_does_not_refund():
    """A late release must not erase a legitimate charge."""
    led = Ledger()
    res = await led.reserve("u1", 0.10, limit=1.0)
    await led.settle(res, 0.05, model="small", input_tokens=1, output_tokens=1)
    await led.release(res)
    assert await led.spent("u1") == pytest.approx(0.05)


async def test_actual_cost_may_exceed_reservation():
    """Projections are estimates. If a model overruns them, the ledger records
    the real cost rather than silently capping it."""
    led = Ledger()
    res = await led.reserve("u1", 0.10, limit=100.0)
    await led.settle(res, 0.40, model="large", input_tokens=1, output_tokens=1)
    assert await led.spent("u1") == pytest.approx(0.40)


async def test_concurrent_reservations_never_exceed_the_limit():
    """The classic overspend race: N requests read the same balance at once.

    Without an atomic check-and-hold, all 50 would pass a $1.00 check.
    """
    led = Ledger()
    results = await asyncio.gather(
        *[led.reserve("u1", 0.10, limit=1.0) for _ in range(50)],
        return_exceptions=True,
    )
    granted = [r for r in results if not isinstance(r, Exception)]
    rejected = [r for r in results if isinstance(r, UserBudgetExceededError)]
    assert len(granted) == 10  # exactly $1.00 worth
    assert len(rejected) == 40
    assert await led.spent("u1") <= 1.0 + 1e-9


async def test_budget_exactly_at_limit_is_allowed():
    """Float arithmetic must not reject a request that lands exactly on the cap."""
    led = Ledger()
    await led.reserve("u1", 0.1 + 0.2, limit=0.3)  # 0.30000000000000004
    assert await led.spent("u1") > 0


# ---------------------------------------------------------------------------
# Rolling-window boundaries
# ---------------------------------------------------------------------------


async def test_spend_expires_exactly_at_window_edge():
    led = Ledger(window_seconds=100.0)
    now = [1000.0]
    led._now = lambda: now[0]
    res = await led.reserve("u1", 0.5, limit=1.0)
    await led.settle(res, 0.5, model="small", input_tokens=1, output_tokens=1)

    now[0] = 1099.9  # still inside the window
    assert await led.spent("u1") == pytest.approx(0.5)
    now[0] = 1100.1  # just past it
    assert await led.spent("u1") == pytest.approx(0.0)


async def test_budget_frees_up_after_window_rolls():
    """A user capped out yesterday must be able to spend again today."""
    led = Ledger(window_seconds=100.0)
    now = [1000.0]
    led._now = lambda: now[0]
    await led.reserve("u1", 1.0, limit=1.0)
    with pytest.raises(UserBudgetExceededError):
        await led.reserve("u1", 0.1, limit=1.0)
    now[0] += 200.0
    await led.reserve("u1", 1.0, limit=1.0)  # must succeed


# ---------------------------------------------------------------------------
# Hostile / degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", ["", "   ", "\n\n\t", "\x00"])
async def test_degenerate_prompts_do_not_crash(prompt):
    svc = InferenceService(_settings())
    res = await svc.handle(prompt, user_id="u1")
    assert res.model  # routed somewhere rather than raising


@pytest.mark.parametrize(
    "prompt",
    [
        "🙂🙂🙂 explain this",
        "日本語のテキストを分析してください",
        "Ünïcödé ãccènts",
        "𝕄𝕒𝕥𝕙𝕖𝕞𝕒𝕥𝕚𝕔𝕒𝕝 𝕓𝕠𝕝𝕕",
    ],
)
def test_unicode_token_counting_is_stable(prompt):
    """Emoji and multi-byte scripts must not produce zero or negative counts —
    a zero count means a free request."""
    n = count_tokens(prompt)
    assert n >= 1


def test_prompt_exceeding_every_context_window_is_rejected():
    s = _settings()
    huge = "word " * 300_000
    with pytest.raises(ContextTooLargeError):
        route(huge, s, budget_usd=1e9)


def test_max_output_tokens_larger_than_context_is_rejected():
    s = _settings()
    with pytest.raises(ContextTooLargeError):
        route("hi", s, task_type="simple", max_output_tokens=10_000_000,
              budget_usd=1e9)


def test_zero_budget_rejects_rather_than_serving_free():
    s = _settings()
    with pytest.raises(BudgetExceededError):
        route("hi", s, budget_usd=0.0)


def test_negative_budget_is_treated_as_unaffordable():
    s = _settings()
    with pytest.raises(BudgetExceededError):
        route("hi", s, budget_usd=-5.0)


def test_single_model_catalog_cannot_downgrade():
    """With one model there is nowhere to switch down to — the router must raise
    rather than loop or return the same model as a 'downgrade'."""
    only_large = ModelSpec(
        name="large", provider="mock", tier=Complexity.COMPLEX,
        input_price_per_1k=0.015, output_price_per_1k=0.075,
        max_context_tokens=200_000, speed=1.0,
    )
    s = Settings(catalog=[only_large])
    with pytest.raises(BudgetExceededError):
        route("hi", s, budget_usd=1e-9, max_output_tokens=1000)


def test_explicit_task_type_with_garbage_value_falls_back_to_heuristic():
    """An unknown task_type must not silently route everything to SIMPLE."""
    r = estimate_complexity(
        "Analyze and design and prove this step by step", task_type="banana"
    )
    assert r.tier == Complexity.COMPLEX


# ---------------------------------------------------------------------------
# Capacity gate
# ---------------------------------------------------------------------------


async def test_gate_blocks_beyond_capacity_and_reports_queue_depth():
    gate = CapacityGate(2)
    await gate.acquire()
    await gate.acquire()
    assert gate.active == 2

    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)  # let the waiter park
    await asyncio.sleep(0)
    assert gate.waiting == 1, "a blocked request must show up as queue depth"

    await gate.release()
    await waiter
    assert gate.waiting == 0
    assert gate.active == 2


async def test_gate_shrink_does_not_interrupt_inflight_work():
    """Scaling down must drain gracefully, never kill running generations."""
    gate = CapacityGate(4)
    for _ in range(4):
        await gate.acquire()
    await gate.resize(1)
    assert gate.active == 4  # in-flight work survives
    assert gate.capacity == 1

    # No new admissions until we drop below the new capacity.
    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not waiter.done()
    for _ in range(4):
        await gate.release()
    await waiter
    assert gate.active == 1


async def test_gate_grow_wakes_waiters():
    gate = CapacityGate(1)
    await gate.acquire()
    waiter = asyncio.create_task(gate.acquire())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not waiter.done()
    await gate.resize(5)
    await waiter
    assert gate.active == 2


# ---------------------------------------------------------------------------
# Autoscaler
# ---------------------------------------------------------------------------


def test_autoscaler_does_not_flap_on_a_load_spike():
    """A one-off burst must not cause scale-up-then-immediate-scale-down
    churn — every cycle costs a cold start."""
    clock = [0.0]
    s = _settings(scale_up_cooldown_s=0.0, scale_down_cooldown_s=60.0)
    a = Autoscaler(s, clock=lambda: clock[0])
    a.step(LoadSample(in_flight=40, queue_depth=0))
    assert a.current_replicas == 10
    for _ in range(5):
        clock[0] += 5.0
        a.step(LoadSample(in_flight=0, queue_depth=0))
    assert a.current_replicas == 10, "scale-down must wait out the cooldown"


def test_autoscaler_survives_min_greater_than_max_misconfiguration():
    """An operator typo must not produce a negative or zero replica target."""
    s = _settings(min_replicas=10, max_replicas=2)
    a = Autoscaler(s, clock=lambda: 0.0)
    d = a.decide(LoadSample(in_flight=100, queue_depth=100))
    assert d.desired_replicas >= 1


def test_autoscaler_handles_zero_load_without_dividing_by_zero():
    a = Autoscaler(_settings(), clock=lambda: 0.0)
    d = a.decide(LoadSample(in_flight=0, queue_depth=0))
    assert d.desired_replicas == 1


def test_autoscaler_counts_queue_depth_not_just_inflight():
    """Scaling on in-flight alone is a classic bug: a saturated service has
    constant in-flight and an exploding queue, and never scales."""
    s = _settings(target_concurrency_per_replica=4.0, scale_up_cooldown_s=0.0)
    a = Autoscaler(s, clock=lambda: 0.0)
    d = a.decide(LoadSample(in_flight=4, queue_depth=36))
    assert d.desired_replicas == 10
