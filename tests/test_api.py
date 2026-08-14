import json

import pytest
from fastapi.testclient import TestClient

from app.breaker import CircuitOpenError
from app.config import Settings
from app.cost import LedgerUnavailableError, UserBudgetExceededError
from app.main import _as_http, create_app
from app.providers import (
    MockProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.ratelimit import RateLimitExceededError, TokenBucketLimiter
from app.router import (
    BudgetExceededError,
    ContextTooLargeError,
    NoModelAvailableError,
)
from app.service import AdmissionTimeoutError, InferenceService


@pytest.fixture
def client():
    # Cache and rate limits off by default here so each test exercises the path
    # it is actually about; the suites below turn them on explicitly.
    svc = InferenceService(Settings(cache_backend="none", rate_limit_enabled=False))
    with TestClient(create_app(svc)) as c:
        yield c


def _events(response):
    """Parse an SSE body into a list of decoded events."""
    out = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            out.append({"type": "__done__"})
            continue
        out.append(json.loads(payload))
    return out


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["models"]}
    assert {"small", "medium", "large"} <= names


def test_infer_simple(client):
    r = client.post("/v1/infer", json={"prompt": "say hi", "user_id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "small"
    assert body["complexity"] == "SIMPLE"


def test_route_explain_is_dry_run(client):
    r = client.post(
        "/v1/route/explain",
        json={"prompt": "Analyze and design and prove step by step", "budget_usd": 1.0},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "large"
    assert client.get("/v1/usage").json()["spend_by_user"] == {}


def test_infer_budget_downgrade(client):
    r = client.post(
        "/v1/infer",
        json={
            "prompt": "Analyze and design and prove correctness step by step",
            "user_id": "u2",
            "budget_usd": 0.001,
            "max_output_tokens": 100,
        },
    )
    assert r.status_code == 200
    assert r.json()["downgraded"] is True


def test_infer_user_budget_429(client):
    r = client.post(
        "/v1/infer",
        json={"prompt": "say hi", "user_id": "u3", "user_daily_budget_usd": 0.0},
    )
    assert r.status_code == 429


def test_metrics_endpoint(client):
    client.post("/v1/infer", json={"prompt": "say hi", "user_id": "u1"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "llm_total_requests" in r.text


# ---- streaming ------------------------------------------------------------


def test_stream_emits_start_deltas_end_and_done(client):
    r = client.post("/v1/infer/stream", json={"prompt": "say hi", "user_id": "s1"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _events(r)
    assert events[0]["type"] == "start"
    assert events[0]["model"] == "small"
    assert any(e["type"] == "delta" for e in events)
    end = [e for e in events if e["type"] == "end"][0]
    assert end["output_tokens"] >= 1
    assert end["cost_usd"] > 0
    assert events[-1]["type"] == "__done__"

    text = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    assert text


def test_stream_rejects_over_budget_before_streaming_starts(client):
    """An admission failure must be a real 4xx, not a 200 with an error buried
    in the stream — clients key retry/alerting off the status code."""
    r = client.post(
        "/v1/infer/stream",
        json={"prompt": "say hi", "user_id": "s2", "user_daily_budget_usd": 0.0},
    )
    assert r.status_code == 429


def test_stream_rejects_impossible_request_budget(client):
    r = client.post(
        "/v1/infer/stream",
        json={"prompt": "say hi", "user_id": "s3", "budget_usd": 0.0},
    )
    assert r.status_code == 402


def test_stream_billing_lands_in_usage(client):
    client.post("/v1/infer/stream", json={"prompt": "say hi", "user_id": "s4"})
    spend = client.get("/v1/usage").json()["spend_by_user"]
    assert spend.get("s4", 0) > 0


def test_stream_reports_downgrade_in_start_event(client):
    r = client.post(
        "/v1/infer/stream",
        json={
            "prompt": "Analyze and design and prove correctness step by step",
            "user_id": "s5",
            "budget_usd": 0.001,
            "max_output_tokens": 100,
        },
    )
    start = _events(r)[0]
    assert start["type"] == "start"
    assert start["downgraded"] is True


def test_stream_midflight_provider_error_emits_error_event():
    svc = InferenceService(
        Settings(),
        provider=MockProvider(chunk_chars=4, fail_after_chunks=2),
    )
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/infer/stream", json={"prompt": "say hi", "user_id": "s6"})
        # Status is already committed to 200 once bytes flow, so the failure is
        # reported in-band.
        assert r.status_code == 200
        events = _events(r)
        assert any(e["type"] == "error" for e in events)


def test_provider_unavailable_maps_to_503():
    svc = InferenceService(
        Settings(cache_backend="none", rate_limit_enabled=False, fallback_enabled=False),
        provider=MockProvider(raise_error=ProviderUnavailableError("vllm down")),
    )
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/infer", json={"prompt": "say hi", "user_id": "s7"})
        assert r.status_code == 503


# ---- caching, shedding, throttling over HTTP -------------------------------


def test_cache_hit_is_reported_and_free():
    svc = InferenceService(Settings(cache_backend="memory", rate_limit_enabled=False))
    with TestClient(create_app(svc)) as c:
        first = c.post("/v1/infer", json={"prompt": "say hi", "user_id": "c1"}).json()
        second = c.post("/v1/infer", json={"prompt": "say hi", "user_id": "c1"}).json()
        assert first["cached"] is False
        assert second["cached"] is True
        assert second["cost_usd"] == 0.0


def test_stream_cache_hit_replays_with_cached_flag():
    svc = InferenceService(Settings(cache_backend="memory", rate_limit_enabled=False))
    with TestClient(create_app(svc)) as c:
        c.post("/v1/infer/stream", json={"prompt": "say hi", "user_id": "c2"})
        r = c.post("/v1/infer/stream", json={"prompt": "say hi", "user_id": "c2"})
        events = _events(r)
        assert events[0]["cached"] is True
        assert [e for e in events if e["type"] == "end"][0]["cost_usd"] == 0.0


def test_rate_limited_request_returns_429_with_retry_after():
    """A blind retry storm is what turns throttling into an outage. Retry-After
    turns it into coordinated backoff."""
    svc = InferenceService(
        Settings(cache_backend="none"),
        rate_limiter=TokenBucketLimiter(capacity=1.0, refill_per_second=1.0),
    )
    with TestClient(create_app(svc)) as c:
        assert c.post("/v1/infer", json={"prompt": "a", "user_id": "r1"}).status_code == 200
        r = c.post("/v1/infer", json={"prompt": "b", "user_id": "r1"})
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1


@pytest.mark.parametrize(
    "exc,expected",
    [
        (BudgetExceededError(1.0, 0.5), 402),
        (UserBudgetExceededError("u", 1.0, 1.0, 1.0), 429),
        (RateLimitExceededError("u", 2.0), 429),
        (ContextTooLargeError("too big"), 413),
        (LedgerUnavailableError("redis down"), 503),
        (AdmissionTimeoutError(30.0), 503),
        (CircuitOpenError("small", 12.0), 503),
        (NoModelAvailableError("all excluded"), 503),
        (ProviderTimeoutError("slow"), 504),
        (ProviderUnavailableError("down"), 503),
        (ProviderError("garbage"), 502),
        (ValueError("bad model"), 400),
    ],
)
def test_every_domain_error_maps_to_a_status(exc, expected):
    """The map is order-sensitive (subclasses must precede their base), so it is
    worth asserting exhaustively rather than trusting the ordering by eye."""
    assert _as_http(exc).status_code == expected


def test_retry_after_is_set_for_throttling_errors():
    assert _as_http(RateLimitExceededError("u", 2.0)).headers["retry-after"] == "3"
    assert _as_http(CircuitOpenError("small", 12.0)).headers["retry-after"] == "13"


def test_retry_after_is_omitted_when_unbounded():
    """An infinite retry delay must not render as a nonsense header value."""
    assert _as_http(RateLimitExceededError("u", float("inf"))).headers is None


def test_status_endpoint_reports_operational_state():
    svc = InferenceService(Settings(cache_backend="memory", rate_limit_enabled=False))
    with TestClient(create_app(svc)) as c:
        c.post("/v1/infer", json={"prompt": "say hi", "user_id": "st1"})
        c.post("/v1/infer", json={"prompt": "say hi", "user_id": "st1"})
        body = c.get("/v1/status").json()
        assert body["cache_hit_rate"] > 0
        assert body["replicas"] >= 1
        assert body["open_circuits"] == []


def test_metrics_include_cache_and_shedding_counters():
    svc = InferenceService(Settings(cache_backend="memory", rate_limit_enabled=False))
    with TestClient(create_app(svc)) as c:
        c.post("/v1/infer", json={"prompt": "say hi", "user_id": "m1"})
        text = c.get("/metrics").text
        for name in (
            "llm_total_cache_hits",
            "llm_total_cache_misses",
            "llm_total_shed",
            "llm_total_rate_limited",
            "llm_total_failovers",
        ):
            assert name in text
