"""vLLM provider.

Talks to a vLLM server through its OpenAI-compatible HTTP API
(``/v1/completions``), which is what ``vllm serve <model>`` exposes by default.

Industrial concerns handled here:

* **Timeouts** — a separate, shorter connect timeout from the read timeout, so a
  dead server fails fast while a slow generation is allowed to finish.
* **Retries with backoff + jitter** — only for *retryable* conditions
  (connection errors, 429, 5xx). A 400 is a client bug and is never retried.
* **Streaming** — SSE frames are parsed incrementally; malformed or unknown
  frames are skipped rather than killing the generation.
* **Usage reporting** — vLLM returns real prompt/completion token counts, which
  we prefer over our own estimates for billing accuracy.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

from ..config import ModelSpec
from ..tokens import count_tokens
from .base import (
    Completion,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamChunk,
    Usage,
)

try:  # httpx is required only when actually using this provider
    import httpx

    _HAVE_HTTPX = True
except Exception:  # pragma: no cover
    _HAVE_HTTPX = False

# Status codes worth retrying: rate limiting and transient server faults.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class VLLMProvider:
    """Adapter for a vLLM server exposing the OpenAI-compatible API.

    ``model_map`` translates our catalog names ("small"/"medium"/"large") into
    the concrete model IDs the server was launched with. Without a mapping the
    catalog name is sent through unchanged.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: Optional[str] = None,
        model_map: Optional[Dict[str, str]] = None,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        chars_per_token: float = 4.0,
        client: Any = None,
    ):
        if not _HAVE_HTTPX and client is None:
            raise RuntimeError(
                "VLLMProvider requires httpx: pip install httpx"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_map = model_map or {}
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.chars_per_token = chars_per_token
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
        )

    # ---- helpers -------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self, model: ModelSpec, prompt: str, max_output_tokens: int, stream: bool
    ) -> Dict[str, Any]:
        return {
            "model": self.model_map.get(model.name, model.name),
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "stream": stream,
            # Ask vLLM to include token accounting on the final SSE frame so we
            # bill on the server's counts rather than our estimate.
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }

    async def _sleep_backoff(self, attempt: int) -> None:
        # Exponential backoff with deterministic jitter derived from the attempt
        # number (no RNG, so tests stay reproducible).
        delay = self.backoff_base_s * (2**attempt) * (1.0 + 0.1 * (attempt % 3))
        await asyncio.sleep(delay)

    @staticmethod
    def _usage_from(data: Dict[str, Any]) -> Optional[Usage]:
        usage = data.get("usage")
        if not usage:
            return None
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if prompt_tokens is None and completion_tokens is None:
            return None
        return Usage(
            input_tokens=int(prompt_tokens or 0),
            output_tokens=int(completion_tokens or 0),
        )

    # ---- buffered ------------------------------------------------------

    async def generate(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> Completion:
        payload = self._payload(model, prompt, max_output_tokens, stream=False)
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                resp = await self._client.post(
                    "/v1/completions", json=payload, headers=self._headers()
                )
            except Exception as e:  # transport-level failure
                if _HAVE_HTTPX and isinstance(e, httpx.TimeoutException):
                    last_exc = ProviderTimeoutError(f"vLLM timed out: {e}")
                else:
                    last_exc = ProviderUnavailableError(f"vLLM unreachable: {e}")
                if attempt < self.max_retries - 1:
                    await self._sleep_backoff(attempt)
                    continue
                raise last_exc

            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = ProviderUnavailableError(
                    f"vLLM returned {resp.status_code}"
                )
                if attempt < self.max_retries - 1:
                    await self._sleep_backoff(attempt)
                    continue
                raise last_exc

            if resp.status_code >= 400:
                # Client error: retrying will not help.
                raise ProviderError(
                    f"vLLM rejected the request ({resp.status_code}): {resp.text[:200]}"
                )

            return self._parse_completion(resp.json(), model, prompt)

        raise last_exc or ProviderUnavailableError("vLLM request failed")

    def _parse_completion(
        self, data: Dict[str, Any], model: ModelSpec, prompt: str
    ) -> Completion:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("vLLM returned no choices")
        text = choices[0].get("text", "") or ""
        usage = self._usage_from(data)
        if usage is None:
            # Fall back to local estimation so billing still works if the server
            # omits usage (some vLLM builds/flags do).
            usage = Usage(
                input_tokens=count_tokens(prompt, self.chars_per_token),
                output_tokens=count_tokens(text, self.chars_per_token),
            )
        return Completion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=model.name,
        )

    # ---- streaming -----------------------------------------------------

    async def generate_stream(
        self, model: ModelSpec, prompt: str, *, max_output_tokens: int
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens as SSE frames arrive.

        Retries apply only to *establishing* the stream. Once bytes have been
        delivered to the caller we cannot transparently restart without
        duplicating output, so a mid-stream failure propagates (and the service
        bills for the partial output).
        """
        payload = self._payload(model, prompt, max_output_tokens, stream=True)

        for attempt in range(self.max_retries):
            started = False
            try:
                async with self._client.stream(
                    "POST", "/v1/completions", json=payload, headers=self._headers()
                ) as resp:
                    if resp.status_code in _RETRYABLE_STATUS:
                        if attempt < self.max_retries - 1:
                            await self._sleep_backoff(attempt)
                            continue
                        raise ProviderUnavailableError(
                            f"vLLM returned {resp.status_code}"
                        )
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        raise ProviderError(
                            f"vLLM rejected the stream ({resp.status_code}): "
                            f"{body[:200]!r}"
                        )

                    final_usage: Optional[Usage] = None
                    finish_reason: Optional[str] = None
                    accumulated = 0

                    async for line in resp.aiter_lines():
                        frame = self._parse_sse_line(line)
                        if frame is None:
                            continue
                        if frame == "[DONE]":
                            break
                        started = True
                        choices = frame.get("choices") or []
                        if choices:
                            piece = choices[0].get("text", "") or ""
                            finish_reason = (
                                choices[0].get("finish_reason") or finish_reason
                            )
                            if piece:
                                accumulated += len(piece)
                                yield StreamChunk(text=piece)
                        usage = self._usage_from(frame)
                        if usage is not None:
                            final_usage = usage

                    if final_usage is None:
                        final_usage = Usage(
                            input_tokens=count_tokens(prompt, self.chars_per_token),
                            output_tokens=max(
                                1, round(accumulated / self.chars_per_token)
                            )
                            if accumulated
                            else 0,
                        )
                    yield StreamChunk(
                        finish_reason=finish_reason or "stop", usage=final_usage
                    )
                    return

            except (ProviderError, ProviderTimeoutError, ProviderUnavailableError):
                raise
            except Exception as e:
                if started:
                    # Already emitted tokens — cannot safely retry.
                    raise ProviderError(f"vLLM stream failed mid-flight: {e}") from e
                if _HAVE_HTTPX and isinstance(e, httpx.TimeoutException):
                    err: Exception = ProviderTimeoutError(f"vLLM timed out: {e}")
                else:
                    err = ProviderUnavailableError(f"vLLM unreachable: {e}")
                if attempt < self.max_retries - 1:
                    await self._sleep_backoff(attempt)
                    continue
                raise err

        raise ProviderUnavailableError("vLLM stream could not be established")

    @staticmethod
    def _parse_sse_line(line: str) -> Any:
        """Return a parsed frame, the string ``"[DONE]"``, or None to skip.

        Unknown or malformed frames return None: a single bad frame must not
        abort an otherwise healthy generation.
        """
        if not line:
            return None
        line = line.strip()
        if not line or line.startswith(":"):
            return None  # comment / keep-alive
        if not line.startswith("data:"):
            return None
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return "[DONE]"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
