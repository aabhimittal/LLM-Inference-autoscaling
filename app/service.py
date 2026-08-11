"""Orchestration: routing + cost control + provider + metrics, buffered and streamed.

Flow of one request (see docs/IMPLEMENTATION.md for the full walk-through):

  1. route()        -> pick model via complexity + budget (may downgrade)
  2. ledger.reserve -> hold projected cost against the user's daily budget
  3. gate.acquire   -> admission control; waiting here is the autoscaler's signal
  4. provider       -> run the model (buffered or streamed)
  5. ledger.settle  -> reconcile the hold with real token usage
  6. metrics        -> record tokens/cost so the autoscaler can react

The streaming path adds one hard requirement the buffered path does not have:
**a client that disconnects mid-generation must still be billed for the tokens
already produced.** Cleanup therefore runs in a detached background task rather
than inline in the generator's ``finally``, because a cancelled task cannot
reliably ``await`` a Redis round-trip. See :meth:`_finalize_later`.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional, Set

from .autoscaler import Autoscaler, LoadSample, ScaleDecision
from .config import Settings
from .cost import Ledger, LedgerUnavailableError, UserBudgetExceededError
from .gate import CapacityGate
from .metrics import Metrics
from .providers import MockProvider, Provider, Usage
from .router import (
    BudgetExceededError,
    ContextTooLargeError,
    RoutingDecision,
    route,
)
from .tokens import count_tokens

log = logging.getLogger(__name__)


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


@dataclass
class StreamEvent:
    """One event on the streaming path.

    ``type`` is ``start`` | ``delta`` | ``end`` | ``error``. ``start`` carries the
    routing decision so a client learns which model it got (and whether it was
    downgraded) before the first token arrives; ``end`` carries final billing.
    """

    type: str
    text: str = ""
    data: dict = field(default_factory=dict)


class InferenceService:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        provider: Optional[Provider] = None,
        ledger: Any = None,
        metrics: Optional[Metrics] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.provider = provider or MockProvider()
        self.ledger = ledger or Ledger()
        self.metrics = metrics or Metrics()
        self.autoscaler = Autoscaler(self.settings)
        self.gate = CapacityGate(self._capacity_for(self.autoscaler.current_replicas))
        # Detached cleanup tasks. Held in a set so they are not garbage-collected
        # mid-flight (asyncio only keeps weak references to running tasks).
        self._pending: Set[asyncio.Task] = set()

    def _capacity_for(self, replicas: int) -> int:
        return max(1, int(replicas * self.settings.target_concurrency_per_replica))

    # ---- shared request preamble ------------------------------------------

    async def _prepare(
        self,
        prompt: str,
        *,
        user_id: str,
        task_type: Optional[str],
        max_output_tokens: Optional[int],
        budget_usd: Optional[float],
        user_daily_budget_usd: Optional[float],
        force_model: Optional[str],
    ):
        """Route and reserve. Raises before any model call or capacity is taken."""
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

        user_limit = (
            user_daily_budget_usd
            if user_daily_budget_usd is not None
            else self.settings.default_user_daily_budget_usd
        )
        try:
            reservation = await self.ledger.reserve(
                user_id, decision.projected_cost, user_limit
            )
        except (UserBudgetExceededError, LedgerUnavailableError):
            self.metrics.request_rejected()
            raise
        return decision, reservation

    # ---- buffered ---------------------------------------------------------

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
        decision, reservation = await self._prepare(
            prompt,
            user_id=user_id,
            task_type=task_type,
            max_output_tokens=max_output_tokens,
            budget_usd=budget_usd,
            user_daily_budget_usd=user_daily_budget_usd,
            force_model=force_model,
        )

        await self.gate.acquire()
        self.metrics.request_started()
        try:
            completion = await self.provider.generate(
                decision.model,
                prompt,
                max_output_tokens=decision.projected_output_tokens,
            )
            actual_cost = decision.model.cost(
                completion.input_tokens, completion.output_tokens
            )
            await self.ledger.settle(
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
            # The call failed: refund the hold and record a finished request so
            # the in-flight gauge does not leak.
            await self.ledger.release(reservation)
            self.metrics.request_finished(
                input_tokens=0, output_tokens=0, cost=0.0, downgraded=False
            )
            raise
        finally:
            await self.gate.release()

    # ---- streaming --------------------------------------------------------

    async def handle_stream(
        self,
        prompt: str,
        *,
        user_id: str,
        task_type: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        budget_usd: Optional[float] = None,
        user_daily_budget_usd: Optional[float] = None,
        force_model: Optional[str] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion, billing for whatever is actually produced.

        Three termination paths, all of which must settle correctly:

        * normal completion — bill the provider-reported usage;
        * provider error mid-stream — bill the partial output, surface an error
          event, and let the exception propagate;
        * **client disconnect** — the generator is closed at a ``yield``; the
          detached finalizer still bills the partial output.
        """
        decision, reservation = await self._prepare(
            prompt,
            user_id=user_id,
            task_type=task_type,
            max_output_tokens=max_output_tokens,
            budget_usd=budget_usd,
            user_daily_budget_usd=user_daily_budget_usd,
            force_model=force_model,
        )

        await self.gate.acquire()
        self.metrics.request_started()

        pieces: List[str] = []
        usage: Optional[Usage] = None
        finalized = False

        def finalize() -> None:
            nonlocal finalized
            if finalized:
                return
            finalized = True
            final_usage = usage or Usage(
                input_tokens=decision.input_tokens,
                output_tokens=count_tokens(
                    "".join(pieces), self.settings.chars_per_token
                ),
            )
            self._finalize_later(decision, reservation, final_usage)

        try:
            yield StreamEvent(
                "start",
                data={
                    "model": decision.model.name,
                    "complexity": decision.complexity.tier.name,
                    "downgraded": decision.downgraded,
                    "routing_notes": decision.notes,
                    "projected_cost_usd": round(decision.projected_cost, 6),
                },
            )
            async for chunk in self.provider.generate_stream(
                decision.model,
                prompt,
                max_output_tokens=decision.projected_output_tokens,
            ):
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.text:
                    pieces.append(chunk.text)
                    yield StreamEvent("delta", text=chunk.text)

            final_usage = usage or Usage(
                input_tokens=decision.input_tokens,
                output_tokens=count_tokens(
                    "".join(pieces), self.settings.chars_per_token
                ),
            )
            cost = decision.model.cost(
                final_usage.input_tokens, final_usage.output_tokens
            )
            yield StreamEvent(
                "end",
                data={
                    "model": decision.model.name,
                    "input_tokens": final_usage.input_tokens,
                    "output_tokens": final_usage.output_tokens,
                    "cost_usd": round(cost, 6),
                    "downgraded": decision.downgraded,
                },
            )
        except Exception as e:
            # Partial output still costs money upstream, so we still bill it.
            yield StreamEvent(
                "error", data={"error": type(e).__name__, "detail": str(e)}
            )
            raise
        finally:
            # Runs on success, on provider error, and on client disconnect
            # (GeneratorExit). Must not await — see _finalize_later.
            finalize()

    def _finalize_later(self, decision, reservation, usage: Usage) -> None:
        """Settle billing, release capacity, and record metrics off the request path.

        This is deliberately a detached task: when a client disconnects, the
        request task is being cancelled, and any ``await`` inside the generator's
        ``finally`` would be cancelled too — losing the billing write. A separate
        task is not part of that cancellation scope, so settlement always
        completes.
        """

        async def _run() -> None:
            cost = decision.model.cost(usage.input_tokens, usage.output_tokens)
            try:
                await self.ledger.settle(
                    reservation,
                    cost,
                    model=decision.model.name,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            except Exception:
                log.exception("failed to settle reservation %s", reservation.id)
            finally:
                self.metrics.request_finished(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost=cost,
                    downgraded=decision.downgraded,
                )
                await self.gate.release()

        task = asyncio.create_task(_run())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def drain_settlements(self) -> None:
        """Wait for detached billing tasks to finish (shutdown and tests)."""
        while self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)

    # ---- autoscaling -------------------------------------------------------

    def current_load(self) -> LoadSample:
        return LoadSample(in_flight=self.gate.active, queue_depth=self.gate.waiting)

    async def autoscale_tick(self) -> ScaleDecision:
        self.metrics.set_queue_depth(self.gate.waiting)
        decision = self.autoscaler.step(self.current_load())
        if decision.changed:
            await self.gate.resize(self._capacity_for(decision.desired_replicas))
        return decision

    async def run_autoscaler(self, interval_s: float = 5.0, stop=None) -> None:
        while stop is None or not stop.is_set():
            try:
                await self.autoscale_tick()
            except Exception:  # a scaling hiccup must never kill the loop
                log.exception("autoscaler tick failed")
            await asyncio.sleep(interval_s)

    async def aclose(self) -> None:
        await self.drain_settlements()
        for closeable in (self.provider, self.ledger):
            close = getattr(closeable, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    log.exception("error closing %r", closeable)
