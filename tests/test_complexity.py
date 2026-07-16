from app.complexity import estimate_complexity
from app.config import Complexity


def test_simple_prompt_is_simple():
    r = estimate_complexity("Translate 'hello' to French.")
    assert r.tier == Complexity.SIMPLE


def test_reasoning_prompt_is_complex():
    r = estimate_complexity(
        "Analyze and design an algorithm to optimize this system step by step, "
        "then prove its correctness."
    )
    assert r.tier == Complexity.COMPLEX


def test_code_block_bumps_complexity():
    simple = estimate_complexity("fix this")
    with_code = estimate_complexity("fix this\n```\ndef f(): return 1/0\n```")
    assert with_code.score > simple.score


def test_explicit_task_type_overrides_heuristic():
    r = estimate_complexity(
        "Analyze and prove this complex theorem", task_type="simple"
    )
    assert r.tier == Complexity.SIMPLE
    assert any("explicit" in reason for reason in r.reasons)


def test_long_output_raises_score():
    short = estimate_complexity("write something", max_output_tokens=50)
    long = estimate_complexity("write something", max_output_tokens=4000)
    assert long.score >= short.score
