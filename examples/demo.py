"""End-to-end demo you can run without any API keys or a running server.

    python examples/demo.py

Exercises the whole pipeline in-process: complexity classification, model
switching for cost, per-user budget enforcement, streaming (including a
mid-stream client disconnect that must still be billed), and autoscaling.
"""
import asyncio

from app.autoscaler import LoadSample
from app.config import Settings
from app.cost import UserBudgetExceededError
from app.providers import MockProvider, ProviderUnavailableError
from app.ratelimit import RateLimitExceededError, TokenBucketLimiter
from app.service import AdmissionTimeoutError, InferenceService


class _DeadSmallModel(MockProvider):
    """Stands in for a vLLM replica serving 'small' that has fallen over."""

    async def generate(self, model, prompt, *, max_output_tokens):
        if model.name == "small":
            raise ProviderUnavailableError("small backend is down")
        return await super().generate(model, prompt, max_output_tokens=max_output_tokens)


async def main() -> None:
    svc = InferenceService(
        Settings(rate_limit_enabled=False), provider=MockProvider(chunk_chars=10)
    )

    prompts = [
        ("say hi", "u1", None, None),
        ("Translate 'good morning' to Spanish.", "u1", None, None),
        (
            "Analyze and design a distributed rate limiter, then prove its "
            "correctness step by step.",
            "u1",
            1.0,
            None,
        ),
        # Same complex task but a tiny budget -> forces a downgrade.
        (
            "Analyze and design a distributed rate limiter, then prove its "
            "correctness step by step.",
            "u2",
            0.001,
            100,
        ),
    ]

    print("=== Inference (complexity-based routing + cost controls) ===")
    for prompt, user, budget, max_out in prompts:
        res = await svc.handle(
            prompt, user_id=user, budget_usd=budget, max_output_tokens=max_out
        )
        flag = " (DOWNGRADED)" if res.downgraded else ""
        print(
            f"[{user}] {res.complexity:8} -> {res.model:6}{flag} "
            f"| in={res.input_tokens} out={res.output_tokens} "
            f"| ${res.cost_usd:.5f}"
        )

    print("\n=== Streaming ===")
    print("  ", end="")
    async for event in svc.handle_stream("stream me a fresh answer", user_id="u1"):
        if event.type == "delta":
            print(event.text, end="", flush=True)
        elif event.type == "end":
            print(f"\n   billed ${event.data['cost_usd']:.5f}")

    print("\n=== Streaming with a client disconnect (still billed) ===")
    # A distinct prompt, so this measures generation rather than a cache hit.
    disconnect_prompt = "stream something for the disconnect demo"
    stream = svc.handle_stream(disconnect_prompt, user_id="u4")
    n = 0
    async for event in stream:
        if event.type == "delta":
            n += 1
            if n >= 2:
                break  # client hangs up
    await stream.aclose()
    await svc.drain_settlements()
    billed = (await svc.ledger.summary()).get("u4", 0.0)
    print(f"   disconnected after {n} chunks -> billed ${billed:.5f}")

    print("\n=== Response cache (a hit costs $0) ===")
    miss = await svc.handle("what is the capital of France?", user_id="u5")
    hit = await svc.handle("what is the capital of France?", user_id="u5")
    print(f"   1st call: cached={miss.cached} cost=${miss.cost_usd:.5f}")
    print(f"   2nd call: cached={hit.cached}  cost=${hit.cost_usd:.5f}")
    print(f"   hit rate: {svc.metrics.cache_hit_rate():.0%}")

    print("\n=== Rate limiting (abuse control, separate from $ budgets) ===")
    limited = InferenceService(
        Settings(cache_backend="none"),
        rate_limiter=TokenBucketLimiter(capacity=2.0, refill_per_second=0.5),
    )
    for i in range(4):
        try:
            await limited.handle(f"request {i}", user_id="spammer")
            print(f"   request {i}: served")
        except RateLimitExceededError as e:
            print(f"   request {i}: throttled (retry in {e.retry_after_s:.1f}s)")

    print("\n=== Circuit breaker + failover (primary backend is down) ===")
    failing = InferenceService(
        Settings(cache_backend="none", rate_limit_enabled=False),
        provider=_DeadSmallModel(),
    )
    res = await failing.handle("say hi", user_id="u6")
    print(f"   primary 'small' failed -> served by '{res.model}'")
    print(f"   circuits: {failing.breakers.states()}")

    print("\n=== Load shedding (queue too deep) ===")
    shedding = InferenceService(
        Settings(
            cache_backend="none",
            rate_limit_enabled=False,
            admission_timeout_s=0.1,
            target_concurrency_per_replica=1,
        )
    )
    await shedding.gate.acquire()  # occupy the only slot
    try:
        await shedding.handle("say hi", user_id="u7")
    except AdmissionTimeoutError as e:
        print(f"   shed as expected: {e}")
    print(f"   refunded: ${await shedding.ledger.spent('u7'):.5f}")

    print("\n=== Cost control: user daily budget ===")
    try:
        await svc.handle("a brand new uncached prompt", user_id="u3",
                         user_daily_budget_usd=0.0)
    except UserBudgetExceededError as e:
        print(f"   rejected as expected: {e}")

    print("\n=== Spend ledger ===")
    for user, spend in sorted((await svc.ledger.summary()).items()):
        print(f"   {user}: ${spend:.5f}")

    print("\n=== Autoscaling control loop ===")
    for load in [0, 4, 20, 60, 8, 0]:
        decision = svc.autoscaler.step(LoadSample(in_flight=load, queue_depth=0))
        print(
            f"   load={load:3} -> replicas={decision.desired_replicas:2} "
            f"({decision.reason})"
        )

    await svc.aclose()


if __name__ == "__main__":
    asyncio.run(main())
