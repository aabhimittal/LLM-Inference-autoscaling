"""Deterministic offline provider used for tests and the no-API-key demo."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from ..config import ModelSpec
from ..tokens import count_tokens
from .base import Completion, ProviderError, StreamChunk, Usage


class MockProvider:
    """Reproducible provider that never touches the network.

    Latency is modeled as inversely proportional to the model's ``speed`` so the
    autoscaler sees realistic timing differences between small and large models.

    Failure injection (``fail_after_chunks`` / ``raise_error``) exists so the
    edge-case suite can exercise partial-stream billing without a real server.
    """

    def __init__(
        self,
        *,
        latency_scale: float = 0.0,
        chars_per_token: float = 4.0,
        fail_after_chunks: Optional[int] = None,
        raise_error: Optional[Exception] = None,
        chunk_chars: int = 12,
    ):
        self.latency_scale = latency_scale
        self.chars_per_token = chars_per_token
        self.fail_after_chunks = fail_after_chunks
        self.raise_error = raise_error
        self.chunk_chars = chunk_chars

    def _body(self, model: ModelSpec, prompt: str, max_output_tokens: int) -> str:
        input_tokens = count_tokens(prompt, self.chars_per_token)
        body = (
            f"[{model.name}] response to a {input_tokens}-token prompt. "
            f"This is a deterministic mock completion."
        )
        max_chars = int(max_output_tokens * self.chars_per_token)
        return body[:max_chars] if max_chars else body

    async def generate(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> Completion:
        if self.raise_error and self.fail_after_chunks is None:
            raise self.raise_error
        if self.latency_scale:
            await asyncio.sleep(self.latency_scale / model.speed)
        text = self._body(model, prompt, max_output_tokens)
        return Completion(
            text=text,
            input_tokens=count_tokens(prompt, self.chars_per_token),
            output_tokens=count_tokens(text, self.chars_per_token),
            model=model.name,
        )

    async def generate_stream(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        text = self._body(model, prompt, max_output_tokens)
        emitted = 0
        sent_chunks = 0
        for start in range(0, len(text), self.chunk_chars):
            if (
                self.fail_after_chunks is not None
                and sent_chunks >= self.fail_after_chunks
            ):
                raise self.raise_error or ProviderError("injected mid-stream failure")
            piece = text[start : start + self.chunk_chars]
            if self.latency_scale:
                await asyncio.sleep(self.latency_scale / model.speed)
            emitted += len(piece)
            sent_chunks += 1
            yield StreamChunk(text=piece)
        yield StreamChunk(
            finish_reason="stop",
            usage=Usage(
                input_tokens=count_tokens(prompt, self.chars_per_token),
                output_tokens=count_tokens(text[:emitted], self.chars_per_token),
            ),
        )

    async def aclose(self) -> None:
        return None
