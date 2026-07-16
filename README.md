# LLM Inference Service — Autoscaling + Cost Controls

An LLM inference service that **routes each request to the cheapest model that
can handle it**, enforces **per-request and per-user cost budgets**, and
**autoscales** with load. The headline feature is a *token-consumption check
based on model switching for task complexity*: simple prompts run on a small,
cheap model; only genuinely hard prompts pay for the large one — and if a
request's projected token cost would blow its budget, the router automatically
switches it down to a model that fits.

Runs end to end with **no API keys** thanks to a deterministic mock provider, so
you can try the whole pipeline — routing, budgeting, autoscaling — offline.

> New here? Read **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** for a
> step-by-step, module-by-module build of the whole system.

---

## Why this exists

Serving every request on your best model is simple and ruinously expensive. Most
production LLM traffic is easy (classification, short answers, formatting) and
does not need a frontier model. This service adds three things a raw model
endpoint lacks:

1. **Complexity-based model switching** — a cheap, explainable classifier scores
   each prompt and routes it to the smallest capable model.
2. **Cost controls** — projected token cost is checked against a per-request
   budget (switching models to fit) and against a rolling per-user daily budget
   (reserve-then-settle, so concurrent requests can't overspend).
3. **Autoscaling** — an in-process concurrency controller for bursts, plus a
   `/metrics` signal for a Kubernetes HPA to scale pods for sustained load.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
  POST /v1/infer  ───► │  complexity ─► router ─► ledger ─► provider  │ ─► response
                       │      (1)        (2)+(3)    (4)       (5)      │
                       └─────────────────────────────────────────────┘
                                     │ load metrics
                                     ▼
                             autoscaler (6) ──► replica target ──► /metrics ──► HPA

  (1) app/complexity.py   score task difficulty (no model call)
  (2) app/router.py       pick cheapest capable model
  (3) app/router.py       token/cost budget check -> switch model down if needed
  (4) app/cost.py         reserve against user daily budget, settle on real usage
  (5) app/providers.py    run the model (mock by default; pluggable)
  (6) app/autoscaler.py   ceil(load / target) with fast-up / slow-down cooldowns
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. Run the offline end-to-end demo (routing + budgets + autoscaling)
python examples/demo.py

# 2. Run the test suite
pytest

# 3. Serve the API
uvicorn app.main:app --reload
```

### Try the API

```bash
# Simple prompt -> routed to the small model
curl -s localhost:8000/v1/infer -H 'content-type: application/json' \
  -d '{"prompt":"Translate hello to French","user_id":"alice"}'

# Complex prompt -> routed to the large model
curl -s localhost:8000/v1/infer -H 'content-type: application/json' \
  -d '{"prompt":"Analyze and design a distributed rate limiter and prove correctness step by step","user_id":"alice","budget_usd":1.0}'

# Same complex prompt, tiny budget -> model is switched DOWN to fit (downgraded=true)
curl -s localhost:8000/v1/infer -H 'content-type: application/json' \
  -d '{"prompt":"Analyze and design a distributed rate limiter and prove correctness step by step","user_id":"bob","budget_usd":0.001,"max_output_tokens":100}'

# Dry-run: see the routing decision and projected cost WITHOUT spending
curl -s localhost:8000/v1/route/explain -H 'content-type: application/json' \
  -d '{"prompt":"summarize this","user_id":"alice"}'
```

## API

| Method & path | Description |
|---|---|
| `POST /v1/infer` | Run inference: classify → route → budget-check → generate. |
| `POST /v1/route/explain` | Dry-run the routing decision (no spend, no model call). |
| `GET /v1/models` | Model catalog with pricing and tiers. |
| `GET /v1/usage` | Per-user spend summary. |
| `GET /metrics` | Prometheus metrics (in-flight, queue depth, tokens, cost). |
| `GET /healthz` | Liveness + current replica count. |

**Request fields** (`POST /v1/infer`): `prompt` (required), `user_id`,
`task_type` (`simple|moderate|complex` — overrides the classifier),
`max_output_tokens`, `budget_usd` (per-request; triggers model switching),
`user_daily_budget_usd`, `force_model`.

**Error codes:** `402` request budget can't be met · `429` user daily budget
exhausted · `413` prompt exceeds every context window · `400` bad input.

## Model catalog

| Model | Tier | $/1k in | $/1k out | Context |
|---|---|---|---|---|
| `small` | SIMPLE | 0.00025 | 0.00125 | 16k |
| `medium` | MODERATE | 0.003 | 0.015 | 200k |
| `large` | COMPLEX | 0.015 | 0.075 | 200k |

Edit `DEFAULT_CATALOG` in `app/config.py` to change models/pricing. A real
provider adapter maps these names to concrete vendor model IDs.

## Configuration

All via environment variables (see `app/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_REQUEST_BUDGET_USD` | `0.50` | Default per-request budget. |
| `LLM_USER_DAILY_BUDGET_USD` | `25.0` | Rolling per-user daily cap. |
| `LLM_MIN_REPLICAS` / `LLM_MAX_REPLICAS` | `1` / `20` | Autoscaler bounds. |
| `LLM_TARGET_CONCURRENCY` | `4` | In-flight requests one replica handles. |
| `LLM_SCALE_UP_COOLDOWN_S` / `LLM_SCALE_DOWN_COOLDOWN_S` | `10` / `60` | Anti-flap cooldowns. |

## Deployment

```bash
docker compose up --build            # local container
kubectl apply -f deploy/hpa.yaml     # Deployment + HPA scaling on llm_queue_depth
```

The service scales in two layers: an **in-process concurrency gate** absorbs
bursts instantly, while the **HPA** adds pods for sustained load — both driven by
the same queue-depth signal so they never fight.

## Project layout

```
app/         complexity, router, cost, autoscaler, providers, metrics, service, main
tests/       one suite per component + service/api integration tests
examples/    demo.py — offline end-to-end run
deploy/      Kubernetes Deployment + HPA
docs/        IMPLEMENTATION.md — step-by-step build guide
```

## Testing

```bash
pytest -q      # 34 tests: complexity, routing, cost ledger, autoscaler, service, API
```

## License

See [LICENSE](LICENSE).
