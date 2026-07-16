"""Provider abstraction.

The service talks to one interface, :class:`Provider`, so the routing / cost /
scaling machinery never depends on a specific vendor. A deterministic
:class:`MockProvider` ships by default so the whole system runs and tests offline
with no API keys. A real adapter (OpenAI/Anthropic/vLLM/etc.) just needs to
implement :meth:`generate`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from .config import ModelSpec
from .tokens import count_tokens


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class Provider(Protocol):
    async def generate(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> Completion:
        ...


class MockProvider:
    """Deterministic, offline provider.

    Latency is modeled as inversely proportional to the model's ``speed`` so the
    autoscaler and queue see realistic timing differences between small and large
    models, without calling anything external.
    """

    def __init__(self, *, latency_scale: float = 0.0, chars_per_token: float = 4.0):
        # latency_scale=0 keeps unit tests fast; set >0 for realistic demos.
        self.latency_scale = latency_scale
        self.chars_per_token = chars_per_token

    async def generate(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> Completion:
        if self.latency_scale:
            await asyncio.sleep(self.latency_scale / model.speed)
        input_tokens = count_tokens(prompt, self.chars_per_token)
        # Produce a short, deterministic "answer" whose length is bounded by the
        # output budget so billing/token accounting stays realistic.
        body = (
            f"[{model.name}] response to a {input_tokens}-token prompt. "
            f"This is a deterministic mock completion."
        )
        # Trim to the output budget (in tokens ~ chars/chars_per_token).
        max_chars = int(max_output_tokens * self.chars_per_token)
        text = body[:max_chars] if max_chars else body
        output_tokens = count_tokens(text, self.chars_per_token)
        return Completion(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model.name,
        )
