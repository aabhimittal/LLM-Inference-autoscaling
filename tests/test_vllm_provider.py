"""vLLM provider edge cases, driven by a fake httpx client.

No vLLM server is needed: we script the transport to reproduce the failures a
real deployment hits — timeouts, 429 storms, half-written SSE frames, servers
that forget to report usage.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.config import Complexity, ModelSpec
from app.providers import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    VLLMProvider,
)

MODEL = ModelSpec(
    name="small", provider="vllm", tier=Complexity.SIMPLE,
    input_price_per_1k=0.001, output_price_per_1k=0.002,
    max_context_tokens=8192, speed=1.0,
)


def _provider(handler, **kw) -> VLLMProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://vllm.test")
    return VLLMProvider(
        "http://vllm.test", client=client, backoff_base_s=0.0, **kw
    )


def _sse(*frames: str) -> bytes:
    return "".join(f"data: {f}\n\n" for f in frames).encode()


# ---- buffered -------------------------------------------------------------


async def test_generate_parses_completion_and_server_usage():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "hello world"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4},
            },
        )

    p = _provider(handler)
    c = await p.generate(MODEL, "hi", max_output_tokens=16)
    assert c.text == "hello world"
    # Server-reported counts win over local estimates — they are what we bill on.
    assert (c.input_tokens, c.output_tokens) == (11, 4)


async def test_missing_usage_falls_back_to_local_estimate():
    """Some vLLM builds omit usage. Billing must not silently become zero."""

    def handler(request):
        return httpx.Response(200, json={"choices": [{"text": "hello world"}]})

    p = _provider(handler)
    c = await p.generate(MODEL, "hi there", max_output_tokens=16)
    assert c.output_tokens >= 1
    assert c.input_tokens >= 1


async def test_empty_choices_raises_rather_than_returning_blank():
    def handler(request):
        return httpx.Response(200, json={"choices": []})

    p = _provider(handler)
    with pytest.raises(ProviderError):
        await p.generate(MODEL, "hi", max_output_tokens=16)


async def test_retries_on_429_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "ok"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    p = _provider(handler, max_retries=3)
    c = await p.generate(MODEL, "hi", max_output_tokens=8)
    assert c.text == "ok"
    assert calls["n"] == 3


async def test_gives_up_after_max_retries_on_503():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="unavailable")

    p = _provider(handler, max_retries=3)
    with pytest.raises(ProviderUnavailableError):
        await p.generate(MODEL, "hi", max_output_tokens=8)
    assert calls["n"] == 3, "must stop retrying, not hammer a down server forever"


async def test_client_error_is_not_retried():
    """A 400 is our bug. Retrying it wastes time and amplifies load."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    p = _provider(handler, max_retries=3)
    with pytest.raises(ProviderError):
        await p.generate(MODEL, "hi", max_output_tokens=8)
    assert calls["n"] == 1


async def test_timeout_surfaces_as_timeout_error():
    def handler(request):
        raise httpx.ReadTimeout("too slow", request=request)

    p = _provider(handler, max_retries=2)
    with pytest.raises(ProviderTimeoutError):
        await p.generate(MODEL, "hi", max_output_tokens=8)


async def test_connection_refused_surfaces_as_unavailable():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    p = _provider(handler, max_retries=2)
    with pytest.raises(ProviderUnavailableError):
        await p.generate(MODEL, "hi", max_output_tokens=8)


# ---- streaming ------------------------------------------------------------


async def test_stream_yields_text_then_usage():
    body = _sse(
        json.dumps({"choices": [{"text": "he"}]}),
        json.dumps({"choices": [{"text": "llo"}]}),
        json.dumps(
            {
                "choices": [{"text": "", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ),
        "[DONE]",
    )

    def handler(request):
        return httpx.Response(200, content=body)

    p = _provider(handler)
    chunks = [c async for c in p.generate_stream(MODEL, "hi", max_output_tokens=8)]
    assert "".join(c.text for c in chunks) == "hello"
    assert chunks[-1].usage.input_tokens == 3
    assert chunks[-1].usage.output_tokens == 2
    assert chunks[-1].finish_reason == "stop"


async def test_malformed_sse_frames_are_skipped_not_fatal():
    """One corrupt frame must not abort an otherwise healthy generation."""
    body = (
        b": keep-alive\n\n"
        b"data: {not json}\n\n"
        b"garbage line without prefix\n\n"
        + _sse(json.dumps({"choices": [{"text": "ok"}]}))
        + _sse("[DONE]")
    )

    def handler(request):
        return httpx.Response(200, content=body)

    p = _provider(handler)
    chunks = [c async for c in p.generate_stream(MODEL, "hi", max_output_tokens=8)]
    assert "".join(c.text for c in chunks) == "ok"


async def test_stream_without_usage_estimates_from_text():
    body = _sse(json.dumps({"choices": [{"text": "hello there"}]}), "[DONE]")

    def handler(request):
        return httpx.Response(200, content=body)

    p = _provider(handler)
    chunks = [c async for c in p.generate_stream(MODEL, "hi", max_output_tokens=8)]
    assert chunks[-1].usage.output_tokens >= 1


async def test_stream_retries_before_first_byte():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, content=b"")
        return httpx.Response(
            200, content=_sse(json.dumps({"choices": [{"text": "ok"}]}), "[DONE]")
        )

    p = _provider(handler, max_retries=3)
    chunks = [c async for c in p.generate_stream(MODEL, "hi", max_output_tokens=8)]
    assert "".join(c.text for c in chunks) == "ok"
    assert calls["n"] == 2


async def test_stream_4xx_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, content=b"bad")

    p = _provider(handler, max_retries=3)
    with pytest.raises(ProviderError):
        async for _ in p.generate_stream(MODEL, "hi", max_output_tokens=8):
            pass
    assert calls["n"] == 1


def test_sse_parser_ignores_comments_and_blank_lines():
    assert VLLMProvider._parse_sse_line("") is None
    assert VLLMProvider._parse_sse_line(": ping") is None
    assert VLLMProvider._parse_sse_line("event: message") is None
    assert VLLMProvider._parse_sse_line("data: [DONE]") == "[DONE]"
    assert VLLMProvider._parse_sse_line('data: {"a":1}') == {"a": 1}
    assert VLLMProvider._parse_sse_line("data: [1,2,3]") is None  # not an object
