import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.service import InferenceService


@pytest.fixture
def client():
    svc = InferenceService(Settings())
    app = create_app(svc)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


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
    # Dry-run must not have spent anything.
    usage = client.get("/v1/usage").json()
    assert usage["spend_by_user"] == {}


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
