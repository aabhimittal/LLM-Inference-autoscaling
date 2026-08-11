"""Provider interface shared by every backend (mock, vLLM, ...).

Two calling conventions are supported:

* :meth:`Provider.generate` — buffered; returns a single :class:`Completion`.
* :meth:`Provider.generate_stream` — incremental; yields :class:`StreamChunk`
  objects as tokens arrive, then a final chunk carrying ``usage``.

Streaming matters for cost control: a client can disconnect halfway through a
generation, and the service must still bill for the tokens that were actually
produced. Providers therefore report usage on the *final* chunk, and the service
also tracks accumulated text so it can bill a truncated stream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from ..config import ModelSpec


class ProviderError(Exception):
    """Base class for provider failures (transport, protocol, or upstream)."""


class ProviderTimeoutError(ProviderError):
    """The upstream model server did not respond within the deadline."""


class ProviderUnavailableError(ProviderError):
    """The upstream model server is unreachable or returned a 5xx/429 after retries."""


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class StreamChunk:
    """One incremental piece of a streamed completion.

    ``usage`` is populated only on the terminal chunk; intermediate chunks carry
    text only. ``finish_reason`` mirrors the upstream field ("stop", "length").
    """

    text: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None or self.usage is not None


@runtime_checkable
class Provider(Protocol):
    async def generate(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> Completion:
        ...

    def generate_stream(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        ...

    async def aclose(self) -> None:
        """Release transport resources. No-op for in-process providers."""
        ...
