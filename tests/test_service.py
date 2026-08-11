import pytest

from app.config import Settings
from app.cost import UserBudgetExceededError
from app.service import InferenceService


def _svc():
    return InferenceService(Settings())


async def test_end_to_end_simple_request():
    svc = _svc()
    res = await svc.handle("say hi", user_id="u1")
    assert res.model == "small"
    assert res.cost_usd >= 0
    assert svc.metrics.snapshot()["total_requests"] == 1


async def test_complexity_switches_model():
    svc = _svc()
    res = await svc.handle(
        "Analyze and design an algorithm and prove correctness step by step",
        user_id="u1",
        budget_usd=1.0,
    )
    assert res.model == "large"
    assert res.complexity == "COMPLEX"


async def test_user_budget_enforced():
    svc = _svc()
    with pytest.raises(UserBudgetExceededError):
        await svc.handle("say hi", user_id="u1", user_daily_budget_usd=0.0)
    assert svc.metrics.snapshot()["total_rejected"] == 1


async def test_spend_is_recorded_in_ledger():
    svc = _svc()
    await svc.handle("say hi", user_id="u1")
    assert (await svc.ledger.summary())["u1"] > 0


async def test_autoscale_tick_returns_decision():
    svc = _svc()
    decision = await svc.autoscale_tick()
    assert decision.current_replicas >= 1


async def test_stream_yields_start_deltas_and_end():
    svc = _svc()
    events = [e async for e in svc.handle_stream("say hi", user_id="u1")]
    assert events[0].type == "start"
    assert events[-1].type == "end"
    assert any(e.type == "delta" for e in events)
    text = "".join(e.text for e in events if e.type == "delta")
    assert text
    await svc.drain_settlements()
    assert (await svc.ledger.summary())["u1"] > 0
