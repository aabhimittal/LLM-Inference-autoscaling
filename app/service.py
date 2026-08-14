"""Orchestration: every cross-cutting concern in one place, buffered and streamed.

Order of operations for a request, and why it is this order:

  1. rate limit      cheapest possible rejection — no routing, no money, no capacity
  2. route           complexity + budget decide the model (may downgrade)
  3. cache lookup    a hit generated no tokens, so it is free and skips 4–7
  4. reserve         hold projected cost against the user's rolling budget
  5. admit           enter the capacity gate, or shed if the queue is too deep
  6. generate        circuit-breaker guarded, with failover to another model
  7. settle + cache  reconcile real usage, store the result, record metrics

Two invariants hold across both paths:

* **Every exit refunds or charges exactly once.** A shed request, a failed
  generation, and a client disconnect must each leave the ledger consistent.
* **Only server-side faults trigger failover or trip a breaker.** A malformed
  request fails identically everywhere; retrying it multiplies the damage.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional, Set, Tuple

from .autoscaler import Autoscaler, LoadSample, ScaleDecision
from .breaker import BreakerRegistry, CircuitOpenError
from .cache import CacheEntry, InMemoryCache, make_cache_key
from .config import Settings
from .cost import Ledger, LedgerUnavailableError, UserBudgetExceededError
from .gate import CapacityGate
from .metrics import Metrics
from .providers import (
    MockProvider,
    Provider,
    ProviderTimeoutError,
    ProviderUnavailableError,
    Usage,
)
from .ratelimit import RateLimitExceededError, TokenBucketLimiter
from .router import (
    BudgetExceededError,
    ContextTooLargeError,
    NoModelAvailableError,
    RoutingDecision,
    route,
)
from .tokens import count_tokens

log = logging.getLogger(__name__)

# Failures meaning "this backend is sick" rather than "this request is bad".
SERVER_FAULTS = (ProviderUnavailableError, ProviderTimeoutError)

# Routing/admission failures that are the caller's problem, not the system's.
ADMISSION_FAULTS = (
    BudgetExceededError,
    ContextTooLargeError,
    NoModelAvailableError,
    ValueError,
)


class AdmissionTimeoutError(Exception):
    """The request waited too long for capacity and was shed.

    Shedding beats queueing forever: past a certain wait the client has already
    given up, and generating an answer nobody reads still costs GPU time.
    """

    def __init__(self, waited_s: float):
        self.waited_s = waited_s
        super().__init__(f"timed out after {waited_s:.1f}s waiting for capacity")


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
    # True when served from cache: no tokens generated, cost_usd is 0.
    cached: bool = False


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
        cache: Any = None,
        rate_limiter: Any = None,
        breakers: Optional[BreakerRegistry] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.provider = provider or MockProvider()
        self.ledger = ledger or Ledger()
        self.metrics = metrics or Metrics()
        self.autoscaler = Autoscaler(self.settings)
        self.gate = CapacityGate(self._capacity_for(self.autoscaler.current_replicas))

        if cache is None and self.settings.cache_backend != "none":
            cache = InMemoryCache(max_entries=self.settings.cache_max_entries)
        self.cache = cache

        if rate_limiter is None and self.settings.rate_limit_enabled:
            rate_limiter = TokenBucketLimiter(
                capacity=self.settings.rate_limit_burst,
                refill_per_second=self.settings.rate_limit_per_second,
            )
        self.rate_limiter = rate_limiter

        self.breakers = breakers or BreakerRegistry(
            failure_threshold=self.settings.breaker_failure_threshold,
            recovery_timeout_s=self.settings.breaker_recovery_timeout_s,
        )

        # Detached cleanup tasks. Held in a set so they are not garbage-collected
        # mid-flight (asyncio only keeps weak references to running tasks).
        self._pending: Set[asyncio.Task] = set()

    def _capacity_for(self, replicas: int) -> int:
        return max(1, int(replicas * self.settings.target_concurrency_per_replica))

    # ---- admission steps ---------------------------------------------------

    async def _check_rate_limit(self, user_id: str) -> None:
        """Step 1. Supports both the sync in-process limiter and the async
        Redis one, so either can be injected without the caller caring."""
        if self.rate_limiter is None:
            return
        try:
            result = self.rate_limiter.check(user_id)
            if asyncio.iscoroutine(result):
                await result
        except RateLimitExceededError:
            self.metrics.request_rate_limited()
            raise

    def _route(self, prompt, *, task_type, max_output_tokens, budget_usd, force_model):
        """Step 2."""
        try:
            return route(
                prompt,
                self.settings,
                task_type=task_type,
                max_output_tokens=max_output_tokens,
                budget_usd=budget_usd,
                force_model=force_model,
            )
        except ADMISSION_FAULTS:
            self.metrics.request_rejected()
            raise

    async def _lookup_cache(
        self, decision: RoutingDecision, prompt: str
    ) -> Optional[CacheEntry]:
        """Step 3. Runs *before* the budget reservation: a hit generated no
        tokens, so it costs $0 and is deliberately not charged to the user."""
        key = self._cache_key(decision, prompt)
        if key is None:
            return None
        hit = await self.cache.get(key)
        if hit is not None:
            self.metrics.cache_hit()
        else:
            self.metrics.cache_miss()
        return hit

    async def _reserve(self, decision, user_id, user_daily_budget_usd):
        """Step 4."""
        user_limit = (
            user_daily_budget_usd
            if user_daily_budget_usd is not None
            else self.settings.default_user_daily_budget_usd
        )
        try:
            return await self.ledger.reserve(
                user_id, decision.projected_cost, user_limit
            )
        except (UserBudgetExceededError, LedgerUnavailableError):
            self.metrics.request_rejected()
            raise

    async def _admit(self, reservation) -> None:
        """Step 5. A shed request must refund its hold — otherwise every
        timed-out request silently bills the user for nothing."""
        timeout = self.settings.admission_timeout_s
        if timeout <= 0:
            await self.gate.acquire()
            return
        try:
            await asyncio.wait_for(self.gate.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            self.metrics.request_shed()
            await self.ledger.release(reservation)
            raise AdmissionTimeoutError(timeout)

    # ---- cache helpers -----------------------------------------------------

    def _cache_key(self, decision: RoutingDecision, prompt: str) -> Optional[str]:
        if self.cache is None:
            return None
        return make_cache_key(
            model=decision.model.name,
            prompt=prompt,
            max_output_tokens=decision.projected_output_tokens,
        )

    def _cached_result(self, decision, entry: CacheEntry) -> InferenceResult:
        return InferenceResult(
            text=entry.text,
            model=entry.model,
            complexity=decision.complexity.tier.name,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cost_usd=0.0,
            downgraded=decision.downgraded,
            routing_notes=decision.notes + ["served from cache (cost $0)"],
            cached=True,
        )

    async def _cache_store(
        self, decision, prompt: str, text: str, usage: Usage
    ) -> None:
        """Only ever called after a *successful*, complete generation.

        Caching a partial or failed response would turn one transient upstream
        blip into a permanently wrong answer served at full speed.
        """
        key = self._cache_key(decision, prompt)
        if key is None:
            return
        await self.cache.set(
            key,
            CacheEntry(
                text=text,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                model=decision.model.name,
            ),
            ttl_s=self.settings.cache_ttl_s,
        )

    # ---- failover ----------------------------------------------------------

    def _next_model(
        self, excluded, prompt, *, task_type, max_output_tokens, budget_usd
    ) -> RoutingDecision:
        """Re-route with the failed models excluded.

        Reusing ``route()`` rather than hand-picking a replacement means the
        fallback still respects the complexity tier, the per-request budget, and
        the context window.
        """
        return route(
            prompt,
            self.settings,
            task_type=task_type,
            max_output_tokens=max_output_tokens,
            budget_usd=budget_usd,
            exclude=excluded,
        )

    async def _generate_with_failover(
        self,
        decision: RoutingDecision,
        prompt: str,
        *,
        task_type,
        max_output_tokens,
        budget_usd,
        force_model,
    ) -> Tuple[RoutingDecision, Any]:
        """Generate, moving to another model if this backend is sick.

        A pinned model (``force_model``) is never silently replaced — the caller
        asked for that model specifically.
        """
        excluded: Set[str] = set()
        current = decision
        while True:
            breaker = self.breakers.get(current.model.name)
            last_error: Exception

            if breaker.allows():
                try:
                    completion = await self.provider.generate(
                        current.model,
                        prompt,
                        max_output_tokens=current.projected_output_tokens,
                    )
                    breaker.record_success()
                    return current, completion
                except SERVER_FAULTS as e:
                    breaker.record_failure()
                    last_error = e
            else:
                self.metrics.request_circuit_rejected()
                last_error = CircuitOpenError(
                    current.model.name, breaker.retry_after_s()
                )

            if not self.settings.fallback_enabled or force_model:
                raise last_error

            excluded.add(current.model.name)
            try:
                current = self._next_model(
                    excluded,
                    prompt,
                    task_type=task_type,
                    max_output_tokens=max_output_tokens,
                    budget_usd=budget_usd,
                )
            except Exception:
                raise last_error  # nothing affordable left to fail over to
            self.metrics.request_failover()
            log.warning("failing over to %s after %s", current.model.name, last_error)

    # ---- buffered ----------------------------------------------------------

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
        await self._check_rate_limit(user_id)
        decision = self._route(
            prompt,
            task_type=task_type,
            max_output_tokens=max_output_tokens,
            budget_usd=budget_usd,
            force_model=force_model,
        )

        hit = await self._lookup_cache(decision, prompt)
        if hit is not None:
            return self._cached_result(decision, hit)

        reservation = await self._reserve(decision, user_id, user_daily_budget_usd)
        await self._admit(reservation)

        self.metrics.request_started()
        try:
            served, completion = await self._generate_with_failover(
                decision,
                prompt,
                task_type=task_type,
                max_output_tokens=max_output_tokens,
                budget_usd=budget_usd,
                force_model=force_model,
            )
            actual_cost = served.model.cost(
                completion.input_tokens, completion.output_tokens
            )
            await self.ledger.settle(
                reservation,
                actual_cost,
                model=served.model.name,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            self.metrics.request_finished(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost=actual_cost,
                downgraded=served.downgraded,
            )
            # Store under the key of the model that actually served, so a
            # failover response is never handed back as the primary's answer.
            await self._cache_store(
                served,
                prompt,
                completion.text,
                Usage(completion.input_tokens, completion.output_tokens),
            )
            return InferenceResult(
                text=completion.text,
                model=served.model.name,
                complexity=served.complexity.tier.name,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=round(actual_cost, 6),
                downgraded=served.downgraded,
                routing_notes=served.notes,
            )
        except Exception:
            await self.ledger.release(reservation)
            self.metrics.request_finished(
                input_tokens=0, output_tokens=0, cost=0.0, downgraded=False
            )
            raise
        finally:
            await self.gate.release()

    # ---- streaming ---------------------------------------------------------

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

        Four termination paths, all of which must settle correctly:

        * cache hit — replayed as a stream, billed $0, no capacity taken;
        * normal completion — bill provider-reported usage, then cache it;
        * provider error mid-stream — bill the partial output, emit an ``error``
          event, do **not** cache;
        * **client disconnect** — the generator is closed at a ``yield``; the
          detached finalizer still bills the partial output.
        """
        await self._check_rate_limit(user_id)
        decision = self._route(
            prompt,
            task_type=task_type,
            max_output_tokens=max_output_tokens,
            budget_usd=budget_usd,
            force_model=force_model,
        )

        hit = await self._lookup_cache(decision, prompt)
        if hit is not None:
            async for event in self._replay_cached(decision, hit):
                yield event
            return

        reservation = await self._reserve(decision, user_id, user_daily_budget_usd)
        await self._admit(reservation)
        self.metrics.request_started()

        pieces: List[str] = []
        usage: Optional[Usage] = None
        completed = False
        finalized = False

        def finalize() -> None:
            """Synchronous by necessity: this runs inside the generator's
            ``finally``, which may be executing during task cancellation."""
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
            self._finalize_later(
                decision,
                reservation,
                final_usage,
                prompt="".join(pieces) if completed else None,
                original_prompt=prompt,
            )

        try:
            yield StreamEvent(
                "start",
                data={
                    "model": decision.model.name,
                    "complexity": decision.complexity.tier.name,
                    "downgraded": decision.downgraded,
                    "cached": False,
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

            completed = True
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
                    "cached": False,
                },
            )
        except Exception as e:
            # Partial output still cost GPU time upstream, so it is still billed.
            yield StreamEvent(
                "error", data={"error": type(e).__name__, "detail": str(e)}
            )
            raise
        finally:
            finalize()

    async def _replay_cached(
        self, decision, entry: CacheEntry
    ) -> AsyncIterator[StreamEvent]:
        """Serve a cache hit through the streaming protocol.

        Clients should not need two code paths depending on whether their
        request happened to hit, so the event shape is identical — only
        ``cached`` and the $0 cost differ.
        """
        yield StreamEvent(
            "start",
            data={
                "model": entry.model,
                "complexity": decision.complexity.tier.name,
                "downgraded": decision.downgraded,
                "cached": True,
                "routing_notes": decision.notes + ["served from cache (cost $0)"],
                "projected_cost_usd": 0.0,
            },
        )
        if entry.text:
            yield StreamEvent("delta", text=entry.text)
        yield StreamEvent(
            "end",
            data={
                "model": entry.model,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "cost_usd": 0.0,
                "downgraded": decision.downgraded,
                "cached": True,
            },
        )

    def _finalize_later(
        self,
        decision,
        reservation,
        usage: Usage,
        *,
        prompt: Optional[str] = None,
        original_prompt: str = "",
    ) -> None:
        """Settle billing, release capacity, cache, and record metrics off the
        request path.

        Deliberately a detached task: when a client disconnects, the request
        task is being cancelled, and any ``await`` inside the generator's
        ``finally`` would be cancelled with it — losing the billing write. A
        separate task is not in that cancellation scope, so settlement always
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
            try:
                # `prompt` is the accumulated text, set only when the stream
                # finished cleanly. A truncated stream must never be cached.
                if prompt is not None:
                    await self._cache_store(decision, original_prompt, prompt, usage)
            except Exception:
                log.exception("failed to cache streamed response")
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
        for closeable in (self.provider, self.ledger, self.cache, self.rate_limiter):
            close = getattr(closeable, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    log.exception("error closing %r", closeable)
