"""Task-complexity estimation.

This is the heart of "model switching for task complexity": before we pick a
model we score how hard the request is likely to be. The score is a cheap,
explainable heuristic — no model call — so it adds negligible latency and cost.

Signals we combine:
  * prompt length (longer prompts tend to carry more context / harder tasks)
  * requested output length (asking for a long answer implies more work)
  * reasoning keywords (analyze, prove, design, debug, ...)
  * an explicit caller hint (``task_type``) which always wins if provided
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .config import Complexity
from .tokens import count_tokens

# Keywords that reliably signal genuine reasoning / generation effort. Kept
# lowercase; matched on word boundaries so "design" doesn't fire on "designated".
_REASONING_KEYWORDS = {
    "analyze",
    "analyse",
    "prove",
    "derive",
    "design",
    "architect",
    "debug",
    "optimize",
    "optimise",
    "refactor",
    "explain",
    "reason",
    "step by step",
    "step-by-step",
    "compare",
    "evaluate",
    "algorithm",
    "trade-off",
    "tradeoff",
    "strategy",
    "plan",
}

# Keywords that signal a trivial lookup / formatting task.
_SIMPLE_KEYWORDS = {
    "translate",
    "capitalize",
    "uppercase",
    "lowercase",
    "spell",
    "define",
    "what is",
    "hello",
    "hi ",
    "thanks",
    "summarize",
    "tl;dr",
    "list",
}

_CODE_FENCE = re.compile(r"```")


@dataclass
class ComplexityResult:
    """Outcome of scoring, kept explainable so operators can audit routing."""

    tier: Complexity
    score: float  # continuous 0..1 score before bucketing
    reasons: list[str]


def _keyword_hits(text: str, keywords: set[str]) -> list[str]:
    hits = []
    for kw in keywords:
        if " " in kw:
            if kw in text:
                hits.append(kw)
        elif re.search(rf"\b{re.escape(kw)}\b", text):
            hits.append(kw)
    return hits


def estimate_complexity(
    prompt: str,
    *,
    task_type: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    chars_per_token: float = 4.0,
) -> ComplexityResult:
    """Score a request and bucket it into a :class:`Complexity` tier.

    ``task_type`` is an explicit override: pass "simple"/"moderate"/"complex"
    (case-insensitive) to skip the heuristic entirely. This lets trusted callers
    who already know the task shape bypass the classifier.
    """
    reasons: list[str] = []

    # 1) Explicit override always wins.
    if task_type:
        key = task_type.strip().lower()
        override = {
            "simple": Complexity.SIMPLE,
            "moderate": Complexity.MODERATE,
            "complex": Complexity.COMPLEX,
        }.get(key)
        if override is not None:
            return ComplexityResult(
                tier=override, score=float(override) / 3.0,
                reasons=[f"explicit task_type={key}"],
            )

    text = (prompt or "").lower()
    prompt_tokens = count_tokens(prompt or "", chars_per_token)

    # 2) Length signal — normalized so ~2k prompt tokens saturates the signal.
    length_signal = min(prompt_tokens / 2000.0, 1.0)
    reasons.append(f"prompt_tokens={prompt_tokens} (length_signal={length_signal:.2f})")

    # 3) Output length signal.
    out = max_output_tokens or 0
    output_signal = min(out / 2000.0, 1.0)
    if out:
        reasons.append(f"max_output_tokens={out} (output_signal={output_signal:.2f})")

    # 4) Keyword signals. Reasoning saturates at 3 distinct hits.
    reasoning_hits = _keyword_hits(text, _REASONING_KEYWORDS)
    simple_hits = _keyword_hits(text, _SIMPLE_KEYWORDS)
    reasoning_signal = min(len(reasoning_hits) / 3.0, 1.0)
    if reasoning_hits:
        reasons.append(f"reasoning_keywords={reasoning_hits}")
    if simple_hits:
        reasons.append(f"simple_keywords={simple_hits}")

    # 5) Presence of a code block strongly implies a real engineering task.
    code_signal = 0.15 if _CODE_FENCE.search(prompt or "") else 0.0
    if code_signal:
        reasons.append("contains_code_block")

    # Weighted blend. Reasoning keywords dominate because they are the most
    # reliable indicator (weighted so that several strong hits alone push a
    # request into COMPLEX); length/output are weak secondary signals.
    score = (
        0.60 * reasoning_signal
        + 0.15 * length_signal
        + 0.15 * output_signal
        + code_signal
    )
    # A pure "simple" keyword hit with no reasoning signal pulls the score down.
    if simple_hits and not reasoning_hits:
        score = max(0.0, score - 0.15)

    score = min(score, 1.0)

    # 6) Bucket. Thresholds chosen so that a bare "translate this" lands SIMPLE
    # and anything with reasoning keywords or code lands at least MODERATE.
    if score >= 0.5:
        tier = Complexity.COMPLEX
    elif score >= 0.2:
        tier = Complexity.MODERATE
    else:
        tier = Complexity.SIMPLE

    reasons.append(f"score={score:.2f} -> tier={tier.name}")
    return ComplexityResult(tier=tier, score=score, reasons=reasons)
