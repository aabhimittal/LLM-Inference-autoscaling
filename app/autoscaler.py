"""Autoscaling controller.

Given a live load signal (queue depth + in-flight requests), compute the desired
number of replicas. The controller is deliberately simple and stateless apart
from cooldown bookkeeping, so it is easy to test and reason about:

  desired = ceil(load / target_concurrency_per_replica)  clamped to [min, max]

We scale *up* quickly and scale *down* slowly (cooldown) to avoid thrashing —
the classic asymmetry every real autoscaler uses. The output is a target replica
count; an orchestrator adapter (Kubernetes HPA, a process pool, etc.) turns that
number into actual workers.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .config import Settings


@dataclass
class LoadSample:
    """Instantaneous load. ``in_flight`` = requests being served right now;
    ``queue_depth`` = requests waiting for a worker."""

    in_flight: int
    queue_depth: int

    @property
    def total(self) -> int:
        return self.in_flight + self.queue_depth


@dataclass
class ScaleDecision:
    current_replicas: int
    desired_replicas: int
    reason: str

    @property
    def changed(self) -> bool:
        return self.desired_replicas != self.current_replicas


class Autoscaler:
    def __init__(self, settings: Settings, *, clock=time.monotonic):
        self.settings = settings
        self._clock = clock
        self._current = settings.min_replicas
        self._last_scale_up = 0.0
        self._last_scale_down = 0.0

    @property
    def current_replicas(self) -> int:
        return self._current

    def _raw_desired(self, load: LoadSample) -> int:
        target = self.settings.target_concurrency_per_replica
        needed = math.ceil(load.total / target) if load.total > 0 else 0
        # Never drop below the floor; never exceed the ceiling.
        return max(self.settings.min_replicas, min(needed, self.settings.max_replicas))

    def decide(self, load: LoadSample) -> ScaleDecision:
        """Compute (and, via :meth:`apply`, potentially commit) a target count."""
        now = self._clock()
        desired = self._raw_desired(load)
        current = self._current

        if desired > current:
            if now - self._last_scale_up >= self.settings.scale_up_cooldown_s:
                reason = (
                    f"scale up: load {load.total} needs {desired} replicas "
                    f"(target {self.settings.target_concurrency_per_replica}/replica)"
                )
                return ScaleDecision(current, desired, reason)
            return ScaleDecision(
                current, current, "scale up suppressed by cooldown"
            )

        if desired < current:
            if now - self._last_scale_down >= self.settings.scale_down_cooldown_s:
                reason = (
                    f"scale down: load {load.total} needs only {desired} replicas"
                )
                return ScaleDecision(current, desired, reason)
            return ScaleDecision(
                current, current, "scale down suppressed by cooldown"
            )

        return ScaleDecision(current, current, "no change")

    def apply(self, decision: ScaleDecision) -> None:
        """Commit a decision, stamping the appropriate cooldown clock."""
        if not decision.changed:
            return
        now = self._clock()
        if decision.desired_replicas > self._current:
            self._last_scale_up = now
        else:
            self._last_scale_down = now
        self._current = decision.desired_replicas

    def step(self, load: LoadSample) -> ScaleDecision:
        """Convenience: decide and apply in one call."""
        decision = self.decide(load)
        self.apply(decision)
        return decision
