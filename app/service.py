"""The orchestrator that wires routing + cost control + provider + metrics into a
single ``handle`` call, plus a background autoscaling loop.

Flow of one request (see docs/IMPLEMENTATION.md for the full walk-through):

  1. route()        -> pick model via complexity + budget (may downgrade)
  2. ledger.reserve -> hold projected cost against the user's daily budget
  3. provider.gen   -> actually run the model (mock or real)
  4. ledger.settle  -> reconcile the hold with real token usage
  5. metrics        -> record tokens/cost so the autoscaler can react
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import List, Optional

from .autoscaler import Autoscaler, LoadSample, ScaleDecision
from .config import Settings
from .cost import Ledger, UserBudgetExceededError
from .metrics import Metrics
from .providers import MockProvider, Provider
from .router import (
    BudgetExceededError,
    ContextTooLargeError,
    RoutingDecision,
    route,
)


@dataclass
class InferenceResult:
    text: str
    model: str
    complexity: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    downgraded: bool
    routing_notes: List[str]


class InferenceService:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        provider: Optional[Provider] = None,
        ledger: Optional[Ledger] = None,
        metrics: Optional[Metrics] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.provider = provider or MockProvider()
        self.ledger = ledger or Ledger()
        self.metrics = metrics or Metrics()
        self.autoscaler = Autoscaler(self.settings)
        # Concurrency gate reflecting current replica capacity. Requests beyond
        # capacity wait here, which is exactly the "queue_depth" the scaler sees.
        self._capacity = self._capacity_for(self.autoscaler.current_replicas)
        self._sem = asyncio.Semaphore(self._capacity)
        self._waiting = 0
        self._waiting_lock = asyncio.Lock()

    def _capacity_for(self, replicas: int) -> int:
        per = self.settings.target_concurrency_per_replica
        return max(1, int(replicas * per))

    async def handle(
        self,
        prompt: str,
        *,
        user_id: str,
        task_type: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        budget_usd: Optional[float] = None,
        user_daily_budget_usd: Optional[float] = None,
        force_model: Optional[str] = None,
    ) -> InferenceResult:
        """Handle one inference request end to end.

        Raises ``BudgetExceededError``/``UserBudgetExceededError``/
        ``ContextTooLargeError`` for the caller (the API layer maps these to
        4xx responses).
        """
        # --- 1. Route (complexity + per-request budget, may switch model) ----
        try:
            decision: RoutingDecision = route(
                prompt,
                self.settings,
                task_type=task_type,
                max_output_tokens=max_output_tokens,
                budget_usd=budget_usd,
                force_model=force_model,
            )
        except (BudgetExceededError, ContextTooLargeError, ValueError):
            self.metrics.request_rejected()
            raise

        # --- 2. Reserve against the user's rolling daily budget --------------
        user_limit = (
            user_daily_budget_usd
            if user_daily_budget_usd is not None
            else self.settings.default_user_daily_budget_usd
        )
        try:
            reservation = self.ledger.reserve(
                user_id, decision.projected_cost, user_limit
            )
        except UserBudgetExceededError:
            self.metrics.request_rejected()
            raise

        # --- 3. Enter the capacity gate (this waiting == queue depth) --------
        async with self._track_waiting():
            await self._sem.acquire()
        self.metrics.request_started()
        try:
            # --- 4. Run the model ---------------------------------------
            completion = await self.provider.generate(
                decision.model,
                prompt,
                max_output_tokens=decision.projected_output_tokens,
            )
            actual_cost = decision.model.cost(
                completion.input_tokens, completion.output_tokens
            )
            # --- 5. Settle the reservation with real usage --------------
            self.ledger.settle(
                reservation,
                actual_cost,
                model=decision.model.name,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            self.metrics.request_finished(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost=actual_cost,
                downgraded=decision.downgraded,
            )
            return InferenceResult(
                text=completion.text,
                model=decision.model.name,
                complexity=decision.complexity.tier.name,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=round(actual_cost, 6),
                downgraded=decision.downgraded,
                routing_notes=decision.notes,
            )
        except Exception:
            # On failure, don't charge the user for the reserved amount.
            self.ledger.release(reservation)
            self.metrics.request_finished(
                input_tokens=0, output_tokens=0, cost=0.0, downgraded=False
            )
            raise
        finally:
            self._sem.release()

    def _track_waiting(self):
        service = self

        class _Ctx:
            async def __aenter__(self):
                async with service._waiting_lock:
                    service._waiting += 1
                    service.metrics.set_queue_depth(service._waiting)
                return self

            async def __aexit__(self, *exc):
                async with service._waiting_lock:
                    service._waiting = max(0, service._waiting - 1)
                    service.metrics.set_queue_depth(service._waiting)
                return False

        return _Ctx()

    # ---- Autoscaling -------------------------------------------------------

    def current_load(self) -> LoadSample:
        snap = self.metrics.snapshot()
        return LoadSample(in_flight=snap["in_flight"], queue_depth=snap["queue_depth"])

    def autoscale_tick(self) -> ScaleDecision:
        """Run one control-loop iteration and resize the capacity gate to match.

        Returns the decision so a caller (background loop or test) can log it.
        """
        decision = self.autoscaler.step(self.current_load())
        if decision.changed:
            self._resize(decision.desired_replicas)
        return decision

    def _resize(self, replicas: int) -> None:
        new_capacity = self._capacity_for(replicas)
        delta = new_capacity - self._capacity
        if delta > 0:
            for _ in range(delta):
                self._sem.release()
        # Shrinking is handled lazily: we simply acquire without releasing so the
        # effective capacity drops as in-flight requests complete. To keep this
        # simple and non-blocking we only grow the semaphore here; the scaler's
        # target still governs future decisions.
        self._capacity = new_capacity

    async def run_autoscaler(self, interval_s: float = 5.0, stop=None) -> None:
        """Background loop; call from FastAPI startup. ``stop`` is an asyncio.Event."""
        while stop is None or not stop.is_set():
            self.autoscale_tick()
            await asyncio.sleep(interval_s)
