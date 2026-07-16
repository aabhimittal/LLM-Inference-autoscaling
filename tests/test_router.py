import pytest

from app.config import Complexity, Settings
from app.router import (
    BudgetExceededError,
    ContextTooLargeError,
    route,
)


def test_simple_task_routes_to_small_model():
    s = Settings()
    d = route("say hi", s)
    assert d.model.name == "small"
    assert not d.downgraded


def test_complex_task_routes_to_large_model_with_budget():
    s = Settings()
    d = route(
        "Analyze and design an algorithm, prove correctness step by step",
        s,
        budget_usd=1.0,
    )
    assert d.model.name == "large"
    assert d.complexity.tier == Complexity.COMPLEX


def test_tight_budget_forces_downgrade():
    s = Settings()
    # A complex task would pick "large", but a tiny budget forces a switch down.
    d = route(
        "Analyze and design an algorithm, prove correctness step by step",
        s,
        budget_usd=0.001,
        max_output_tokens=100,
    )
    assert d.downgraded
    assert d.model.name != "large"


def test_impossible_budget_raises():
    s = Settings()
    with pytest.raises(BudgetExceededError):
        route(
            "Analyze and design an algorithm, prove correctness step by step",
            s,
            budget_usd=0.0,
            max_output_tokens=1000,
        )


def test_force_model_bypasses_routing():
    s = Settings()
    d = route("say hi", s, force_model="large", budget_usd=1.0)
    assert d.model.name == "large"


def test_force_unknown_model_raises():
    s = Settings()
    with pytest.raises(ValueError):
        route("hi", s, force_model="nonexistent")


def test_context_too_large_raises():
    s = Settings()
    huge = "word " * 60000  # ~ >16k tokens, exceeds the small model window
    with pytest.raises(ContextTooLargeError):
        route(huge, s, task_type="simple", budget_usd=100.0)
