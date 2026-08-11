import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers import MockProvider, ProviderUnavailableError
from app.service import InferenceService


@pytest.fixture
def client():
    svc = InferenceService(Settings())
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
        Settings(),
        provider=MockProvider(raise_error=ProviderUnavailableError("vllm down")),
    )
    with TestClient(create_app(svc)) as c:
        r = c.post("/v1/infer", json={"prompt": "say hi", "user_id": "s7"})
        assert r.status_code == 503
