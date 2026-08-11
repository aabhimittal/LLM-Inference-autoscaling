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
from app.providers import MockProvider
from app.service import InferenceService


async def main() -> None:
    svc = InferenceService(Settings(), provider=MockProvider(chunk_chars=10))

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
    async for event in svc.handle_stream("say hi", user_id="u1"):
        if event.type == "delta":
            print(event.text, end="", flush=True)
        elif event.type == "end":
            print(f"\n   billed ${event.data['cost_usd']:.5f}")

    print("\n=== Streaming with a client disconnect (still billed) ===")
    before = (await svc.ledger.summary()).get("u4", 0.0)
    stream = svc.handle_stream("say hi", user_id="u4")
    n = 0
    async for event in stream:
        if event.type == "delta":
            n += 1
            if n >= 2:
                break  # client hangs up
    await stream.aclose()
    await svc.drain_settlements()
    after = (await svc.ledger.summary()).get("u4", 0.0)
    print(f"   disconnected after {n} chunks -> billed ${after - before:.5f}")

    print("\n=== Cost control: user daily budget ===")
    try:
        await svc.handle("say hi", user_id="u3", user_daily_budget_usd=0.0)
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
