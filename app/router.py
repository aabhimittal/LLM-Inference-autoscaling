"""Model routing: pick the cheapest capable model, then apply the token/cost
budget check that may *switch* the model down.

This is where "token consumption check based on model switching for task
complexity" actually happens:

  1. Complexity gives us the minimum capable tier.
  2. We pick the cheapest model at or above that tier.
  3. We project token consumption and its cost.
  4. If the projected cost exceeds the request budget, we try to *switch* to a
     cheaper (lower-tier) model that still fits the budget, recording the
     downgrade. If nothing fits, we surface a budget error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

from .complexity import ComplexityResult, estimate_complexity
from .config import Complexity, ModelSpec, Settings
from .tokens import count_tokens, estimate_output_tokens


class BudgetExceededError(Exception):
    """Raised when no model can serve the request within the given budget."""

    def __init__(self, projected_cost: float, budget: float):
        self.projected_cost = projected_cost
        self.budget = budget
        super().__init__(
            f"cheapest capable model projects ${projected_cost:.4f} "
            f"but budget is ${budget:.4f}"
        )


class ContextTooLargeError(Exception):
    """Raised when the prompt exceeds every model's context window."""


class NoModelAvailableError(Exception):
    """Every candidate model was excluded — e.g. all their circuits are open."""


@dataclass
class RoutingDecision:
    model: ModelSpec
    complexity: ComplexityResult
    input_tokens: int
    projected_output_tokens: int
    projected_cost: float
    downgraded: bool  # True when the budget check forced a cheaper model
    notes: List[str]


def _capable_models(catalog: List[ModelSpec], tier: Complexity) -> List[ModelSpec]:
    """Models whose tier can handle ``tier``, cheapest first.

    "Cheapest" is ranked by the blended price of a nominal 1k in / 1k out call so
    the ordering is stable and independent of the specific request size.
    """
    capable = [m for m in catalog if m.tier >= tier]
    return sorted(capable, key=lambda m: m.cost(1000, 1000))


def _all_models_by_price(catalog: List[ModelSpec]) -> List[ModelSpec]:
    return sorted(catalog, key=lambda m: m.cost(1000, 1000))


def route(
    prompt: str,
    settings: Settings,
    *,
    task_type: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    budget_usd: Optional[float] = None,
    force_model: Optional[str] = None,
    exclude: Optional[Set[str]] = None,
) -> RoutingDecision:
    """Choose a model for ``prompt`` under the given budget.

    ``force_model`` bypasses complexity routing (but still runs the budget and
    context checks) so callers can pin a model when they need to.

    ``exclude`` removes model names from consideration. The service uses it for
    failover: when a backend errors or its circuit is open, it re-routes with
    that model excluded, which naturally yields the next-best affordable choice
    instead of duplicating the selection logic.
    """
    notes: List[str] = []
    budget = budget_usd if budget_usd is not None else settings.default_request_budget_usd

    complexity = estimate_complexity(
        prompt,
        task_type=task_type,
        max_output_tokens=max_output_tokens,
        chars_per_token=settings.chars_per_token,
    )

    input_tokens = count_tokens(prompt or "", settings.chars_per_token)
    projected_output = estimate_output_tokens(
        max_output_tokens, settings.default_max_output_tokens
    )

    catalog = [m for m in settings.catalog if m.name not in (exclude or set())]
    if not catalog:
        raise NoModelAvailableError(
            f"every model is excluded ({sorted(exclude or [])})"
        )

    # Forced model path: honor the caller's choice but still validate it.
    if force_model:
        by_name = {m.name: m for m in catalog}
        if force_model not in by_name:
            raise ValueError(f"unknown model '{force_model}'")
        chosen = by_name[force_model]
        _check_context(chosen, input_tokens, projected_output)
        cost = chosen.cost(input_tokens, projected_output)
        if cost > budget:
            raise BudgetExceededError(cost, budget)
        notes.append(f"forced model={force_model}")
        return RoutingDecision(
            model=chosen,
            complexity=complexity,
            input_tokens=input_tokens,
            projected_output_tokens=projected_output,
            projected_cost=cost,
            downgraded=False,
            notes=notes,
        )

    capable = _capable_models(catalog, complexity.tier)
    if not capable:
        raise ValueError(f"no model can serve tier {complexity.tier.name}")

    # Step 1: cheapest capable model.
    primary = capable[0]
    _check_context(primary, input_tokens, projected_output)
    primary_cost = primary.cost(input_tokens, projected_output)
    notes.append(
        f"tier={complexity.tier.name} -> primary={primary.name} "
        f"(${primary_cost:.4f})"
    )

    if primary_cost <= budget:
        return RoutingDecision(
            model=primary,
            complexity=complexity,
            input_tokens=input_tokens,
            projected_output_tokens=projected_output,
            projected_cost=primary_cost,
            downgraded=False,
            notes=notes,
        )

    # Step 2: budget check failed — attempt to switch DOWN to any cheaper model
    # that fits the budget, accepting a possible quality reduction. We only
    # consider models strictly cheaper than the primary.
    notes.append(
        f"primary ${primary_cost:.4f} > budget ${budget:.4f}; attempting downgrade"
    )
    cheaper_first = _all_models_by_price(catalog)
    for candidate in cheaper_first:
        if candidate.cost(1000, 1000) >= primary.cost(1000, 1000):
            continue  # not actually cheaper
        try:
            _check_context(candidate, input_tokens, projected_output)
        except ContextTooLargeError:
            continue
        cost = candidate.cost(input_tokens, projected_output)
        if cost <= budget:
            notes.append(
                f"downgraded to {candidate.name} (${cost:.4f}) to fit budget"
            )
            return RoutingDecision(
                model=candidate,
                complexity=complexity,
                input_tokens=input_tokens,
                projected_output_tokens=projected_output,
                projected_cost=cost,
                downgraded=True,
                notes=notes,
            )

    # Nothing fits.
    raise BudgetExceededError(primary_cost, budget)


def _check_context(model: ModelSpec, input_tokens: int, output_tokens: int) -> None:
    if input_tokens + output_tokens > model.max_context_tokens:
        raise ContextTooLargeError(
            f"{input_tokens}+{output_tokens} tokens exceed "
            f"{model.name} context window {model.max_context_tokens}"
        )
