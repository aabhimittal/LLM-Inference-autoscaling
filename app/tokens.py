"""Token estimation.

We keep token counting behind one function so the rest of the code never cares
whether we used a real tokenizer or a heuristic. If ``tiktoken`` is installed we
use it for accuracy; otherwise we fall back to a chars-per-token heuristic that
is good enough for routing and budget projection.
"""
from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - exercised only when the optional dep is present
    import tiktoken

    _HAVE_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAVE_TIKTOKEN = False


@lru_cache(maxsize=8)
def _encoder(name: str = "cl100k_base"):  # pragma: no cover - dep-specific
    return tiktoken.get_encoding(name)


def count_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Return an estimated token count for ``text``.

    The heuristic (len/chars_per_token) is intentionally conservative and never
    returns 0 for non-empty text, so a tiny prompt still costs at least 1 token.
    """
    if not text:
        return 0
    if _HAVE_TIKTOKEN:  # pragma: no cover - depends on optional install
        try:
            return len(_encoder().encode(text))
        except Exception:
            pass
    return max(1, round(len(text) / chars_per_token))


def estimate_output_tokens(requested_max: int | None, default_max: int) -> int:
    """Projected completion length used for pre-call cost estimation.

    We assume a request will use ~75% of its output budget on average; this is
    only used to *reserve* budget, real usage is billed after the call.
    """
    ceiling = requested_max if requested_max and requested_max > 0 else default_max
    return max(1, round(ceiling * 0.75))
