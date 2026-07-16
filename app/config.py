"""Configuration: the model catalog, pricing, and runtime settings.

Everything downstream (routing, cost control, autoscaling) reads from the
objects defined here, so this is the single source of truth for "what models
exist, what they cost, and what the service limits are".
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class Complexity(IntEnum):
    """Task complexity tiers, ordered so we can compare with <, >, etc.

    The ordering matters: a model that can handle COMPLEX work can also handle
    SIMPLE work, so routing picks the *cheapest model whose tier >= the task's
    tier*.
    """

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a single deployable model.

    Prices are USD per 1,000 tokens, split between input (prompt) and output
    (completion) because real providers charge those differently.
    """

    name: str
    provider: str
    tier: Complexity  # the highest complexity this model is trusted to handle
    input_price_per_1k: float
    output_price_per_1k: float
    max_context_tokens: int
    # Relative throughput hint (tokens/sec-ish). Only used for ranking, not billing.
    speed: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of a call given input/output token counts."""
        return (
            input_tokens / 1000.0 * self.input_price_per_1k
            + output_tokens / 1000.0 * self.output_price_per_1k
        )


# A small, deliberately generic catalog. The names are provider-neutral so the
# demo runs without any API keys; the mock provider (providers.py) understands
# them directly, and a real adapter can map them to concrete model IDs.
DEFAULT_CATALOG: List[ModelSpec] = [
    ModelSpec(
        name="small",
        provider="mock",
        tier=Complexity.SIMPLE,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        max_context_tokens=16000,
        speed=3.0,
    ),
    ModelSpec(
        name="medium",
        provider="mock",
        tier=Complexity.MODERATE,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        max_context_tokens=200000,
        speed=2.0,
    ),
    ModelSpec(
        name="large",
        provider="mock",
        tier=Complexity.COMPLEX,
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        max_context_tokens=200000,
        speed=1.0,
    ),
]


@dataclass
class Settings:
    """Runtime knobs. Environment variables override the defaults so the same
    image can run in dev and prod without code changes."""

    # ---- Cost controls -------------------------------------------------
    # Default budget granted to a request when the caller does not pass one.
    default_request_budget_usd: float = 0.50
    # Hard ceiling on spend per user per rolling window (enforced by the ledger).
    default_user_daily_budget_usd: float = 25.0
    # If a request has no explicit budget and the user ledger is exhausted, we
    # reject rather than silently spend.
    reject_when_over_budget: bool = True

    # ---- Autoscaling ---------------------------------------------------
    min_replicas: int = 1
    max_replicas: int = 20
    # How many concurrent in-flight requests one replica is sized to handle.
    target_concurrency_per_replica: float = 4.0
    # Seconds a scale-down decision must persist before we act (prevents flapping).
    scale_down_cooldown_s: float = 60.0
    scale_up_cooldown_s: float = 10.0

    # ---- Token estimation ---------------------------------------------
    # Fallback chars-per-token ratio used when no real tokenizer is available.
    chars_per_token: float = 4.0
    default_max_output_tokens: int = 512

    catalog: List[ModelSpec] = field(default_factory=lambda: list(DEFAULT_CATALOG))

    @classmethod
    def from_env(cls) -> "Settings":
        def _f(key: str, default: float) -> float:
            return float(os.getenv(key, default))

        def _i(key: str, default: int) -> int:
            return int(os.getenv(key, default))

        return cls(
            default_request_budget_usd=_f("LLM_REQUEST_BUDGET_USD", 0.50),
            default_user_daily_budget_usd=_f("LLM_USER_DAILY_BUDGET_USD", 25.0),
            min_replicas=_i("LLM_MIN_REPLICAS", 1),
            max_replicas=_i("LLM_MAX_REPLICAS", 20),
            target_concurrency_per_replica=_f("LLM_TARGET_CONCURRENCY", 4.0),
            scale_down_cooldown_s=_f("LLM_SCALE_DOWN_COOLDOWN_S", 60.0),
            scale_up_cooldown_s=_f("LLM_SCALE_UP_COOLDOWN_S", 10.0),
        )

    def catalog_by_name(self) -> Dict[str, ModelSpec]:
        return {m.name: m for m in self.catalog}
