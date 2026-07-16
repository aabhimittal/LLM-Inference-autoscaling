import asyncio

import pytest

from app.config import Settings
from app.cost import UserBudgetExceededError
from app.service import InferenceService


def _svc():
    return InferenceService(Settings())


def test_end_to_end_simple_request():
    svc = _svc()
    res = asyncio.run(svc.handle("say hi", user_id="u1"))
    assert res.model == "small"
    assert res.cost_usd >= 0
    assert svc.metrics.snapshot()["total_requests"] == 1


def test_complexity_switches_model():
    svc = _svc()
    res = asyncio.run(
        svc.handle(
            "Analyze and design an algorithm and prove correctness step by step",
            user_id="u1",
            budget_usd=1.0,
        )
    )
    assert res.model == "large"
    assert res.complexity == "COMPLEX"


def test_user_budget_enforced():
    svc = _svc()
    with pytest.raises(UserBudgetExceededError):
        asyncio.run(
            svc.handle("say hi", user_id="u1", user_daily_budget_usd=0.0)
        )
    # Rejection is recorded in metrics.
    assert svc.metrics.snapshot()["total_rejected"] == 1


def test_spend_is_recorded_in_ledger():
    svc = _svc()
    asyncio.run(svc.handle("say hi", user_id="u1"))
    assert svc.ledger.summary()["u1"] > 0


def test_autoscale_tick_returns_decision():
    svc = _svc()
    decision = svc.autoscale_tick()
    assert decision.current_replicas >= 1
