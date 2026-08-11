"""Lightweight in-memory metrics.

Tracks the counters the autoscaler and the /metrics endpoint need: in-flight
requests, cumulative counts, tokens, and spend. Kept dependency-free (no
Prometheus client required) but structured so it can be exported in Prometheus
text format via :meth:`prometheus`.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    in_flight: int = 0
    queue_depth: int = 0
    total_requests: int = 0
    total_rejected: int = 0
    total_downgraded: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    # Cache hits are the cost-savings signal: hit_rate * avg_cost = money saved.
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    # Load-shedding and abuse-control counters. A rising shed count means the
    # autoscaler is not keeping up; a rising circuit count means a sick backend.
    total_rate_limited: int = 0
    total_shed: int = 0
    total_circuit_rejected: int = 0
    total_failovers: int = 0

    def request_started(self) -> None:
        with self._lock:
            self.in_flight += 1
            self.total_requests += 1

    def request_finished(
        self, *, input_tokens: int, output_tokens: int, cost: float, downgraded: bool
    ) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
            if downgraded:
                self.total_downgraded += 1

    def request_rejected(self) -> None:
        with self._lock:
            self.total_rejected += 1

    def set_queue_depth(self, depth: int) -> None:
        with self._lock:
            self.queue_depth = depth

    def cache_hit(self) -> None:
        with self._lock:
            self.total_cache_hits += 1
            self.total_requests += 1

    def cache_miss(self) -> None:
        with self._lock:
            self.total_cache_misses += 1

    def request_rate_limited(self) -> None:
        with self._lock:
            self.total_rate_limited += 1
            self.total_rejected += 1

    def request_shed(self) -> None:
        with self._lock:
            self.total_shed += 1
            self.total_rejected += 1

    def request_circuit_rejected(self) -> None:
        with self._lock:
            self.total_circuit_rejected += 1

    def request_failover(self) -> None:
        with self._lock:
            self.total_failovers += 1

    def cache_hit_rate(self) -> float:
        with self._lock:
            total = self.total_cache_hits + self.total_cache_misses
            return round(self.total_cache_hits / total, 4) if total else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            lookups = self.total_cache_hits + self.total_cache_misses
            return {
                "in_flight": self.in_flight,
                "queue_depth": self.queue_depth,
                "total_requests": self.total_requests,
                "total_rejected": self.total_rejected,
                "total_downgraded": self.total_downgraded,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_cache_hits": self.total_cache_hits,
                "total_cache_misses": self.total_cache_misses,
                "cache_hit_rate": (
                    round(self.total_cache_hits / lookups, 4) if lookups else 0.0
                ),
                "total_rate_limited": self.total_rate_limited,
                "total_shed": self.total_shed,
                "total_circuit_rejected": self.total_circuit_rejected,
                "total_failovers": self.total_failovers,
            }

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = []
        for key, value in snap.items():
            lines.append(f"# TYPE llm_{key} gauge")
            lines.append(f"llm_{key} {value}")
        return "\n".join(lines) + "\n"
