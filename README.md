# LLM Inference Service — Autoscaling + Cost Controls

An LLM inference service that **routes each request to the cheapest model that
can handle it**, enforces **per-request and per-user cost budgets**, and
**autoscales** with load. The headline feature is a *token-consumption check
based on model switching for task complexity*: simple prompts run on a small,
cheap model; only genuinely hard prompts pay for the large one — and if a
request's projected token cost would blow its budget, the router automatically
switches it down to a model that fits.

Backed by **vLLM** for real inference, **Redis** for fleet-wide budget
enforcement, and **SSE streaming** that still bills correctly when a client
disconnects mid-generation. Runs end to end with **no API keys or GPU** thanks to
a deterministic mock provider, so you can try the whole pipeline offline.

> New here? Read **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** for a
> step-by-step, module-by-module build of the whole system.

---

## Why this exists

Serving every request on your best model is simple and ruinously expensive. Most
production LLM traffic is easy (classification, short answers, formatting) and
does not need a frontier model. This service adds what a raw model endpoint
lacks:

1. **Complexity-based model switching** — a cheap, explainable classifier scores
   each prompt and routes it to the smallest capable model.
2. **Cost controls** — projected token cost is checked against a per-request
   budget (switching models to fit) and against a rolling per-user daily budget.
   The Redis ledger makes that budget **global across replicas** — the in-memory
   version silently gives each user N× their limit once you scale out.
3. **Autoscaling** — a resizable in-process concurrency gate for bursts, plus a
   `/metrics` queue-depth signal for a Kubernetes HPA to scale pods.
4. **Streaming that bills honestly** — a client who disconnects halfway through
   is still charged for the tokens the GPU actually produced.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
  POST /v1/infer  ───► │  complexity ─► router ─► ledger ─► provider  │ ─► response
  POST /v1/infer/      │      (1)        (2)+(3)    (4)       (5)      │ ─► SSE stream
       stream          └─────────────────────────────────────────────┘
                                     │ load metrics
                                     ▼
                       gate (6) ─► autoscaler (7) ─► /metrics ─► HPA

  (1) app/complexity.py    score task difficulty (no model call)
  (2) app/router.py        pick cheapest capable model
  (3) app/router.py        token/cost budget check -> switch model down if needed
  (4) app/cost/            reserve -> settle; memory (1 node) or Redis (fleet)
  (5) app/providers/       mock (offline) or vLLM; buffered or streamed
  (6) app/gate.py          admission control; its wait queue is the load signal
  (7) app/autoscaler.py    ceil(load / target) with fast-up / slow-down cooldowns
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

# Stream tokens as they are generated (SSE)
curl -N localhost:8000/v1/infer/stream -H 'content-type: application/json' \
  -d '{"prompt":"say hi","user_id":"alice"}'
```

## API

| Method & path | Description |
|---|---|
| `POST /v1/infer` | Run inference: classify → route → budget-check → generate. |
| `POST /v1/infer/stream` | Same, streamed as Server-Sent Events. |
| `POST /v1/route/explain` | Dry-run the routing decision (no spend, no model call). |
| `GET /v1/models` | Model catalog with pricing and tiers. |
| `GET /v1/usage` | Per-user spend summary. |
| `GET /metrics` | Prometheus metrics (in-flight, queue depth, tokens, cost). |
| `GET /healthz` | Liveness + current replica count. |
| `GET /readyz` | Readiness — verifies the cost store is reachable. |

**Request fields**: `prompt` (required), `user_id`, `task_type`
(`simple|moderate|complex` — overrides the classifier), `max_output_tokens`,
`budget_usd` (per-request; triggers model switching), `user_daily_budget_usd`,
`force_model`.

**Stream events** are SSE frames: `start` (model, complexity, whether it was
downgraded), repeated `delta` (`text`), then `end` (final tokens + `cost_usd`),
terminated by `[DONE]`. Admission failures arrive as a normal HTTP status
*before* the stream opens, never buried inside a `200`.

**Error codes:** `402` request budget can't be met · `429` user daily budget
exhausted · `413` prompt exceeds every context window · `400` bad input ·
`502/503/504` upstream model server unusable/unreachable/too slow · `503` cost
store unreachable while fail-closed.

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
| `LLM_PROVIDER` | `mock` | `mock` (offline) or `vllm`. |
| `LLM_VLLM_BASE_URL` | `http://localhost:8000` | vLLM server address. |
| `LLM_VLLM_MODEL_MAP` | — | `small=org/model-a,large=org/model-b`. |
| `LLM_VLLM_MAX_RETRIES` | `3` | Retries on 429/5xx/connect errors. |
| `LLM_LEDGER` | `memory` | `memory` (per process) or `redis` (fleet-wide). |
| `LLM_REDIS_URL` | `redis://localhost:6379/0` | Ledger store. |
| `LLM_REDIS_FAIL_CLOSED` | `true` | On Redis outage: reject (`true`) or serve unmetered (`false`). |

### Using vLLM

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001   # your model server

LLM_PROVIDER=vllm \
LLM_VLLM_BASE_URL=http://localhost:8001 \
LLM_VLLM_MODEL_MAP="small=Qwen/Qwen2.5-1.5B-Instruct" \
LLM_LEDGER=redis \
uvicorn app.main:app
```

The adapter uses vLLM's OpenAI-compatible `/v1/completions`, prefers the
server's reported token counts for billing, retries 429/5xx with backoff, and
never retries a `4xx` (that's our bug, and retrying amplifies load).

## Deployment

```bash
docker compose up --build            # app + Redis
docker compose --profile gpu up      # ...plus a real vLLM server (needs a GPU)
kubectl apply -f deploy/hpa.yaml     # Deployment + HPA + PodDisruptionBudget
```

The service scales in two layers: the **in-process capacity gate** absorbs bursts
instantly, while the **HPA** adds pods for sustained load — both driven by the
same queue-depth signal so they never fight. Scale on queue depth, not CPU: a pod
blocked on a GPU isn't CPU-busy, so CPU-based scaling under-reacts exactly when
you need it most.

## Project layout

```
app/
  complexity.py router.py         routing + token/cost budget switch
  cost/                           memory + Redis ledgers (reserve/settle/release)
  providers/                      mock + vLLM (buffered and streaming)
  gate.py autoscaler.py metrics.py
  service.py factory.py main.py
tests/       per-component suites + edge-case, vLLM, and Redis suites
examples/    demo.py — offline end-to-end run
deploy/      Kubernetes Deployment + HPA + PDB
docs/        IMPLEMENTATION.md — step-by-step build guide
```

## Testing

```bash
pytest -q      # 103 tests
```

Beyond per-component coverage, `tests/test_edge_cases.py` targets the failure
modes that cost real money or wake people up:

- **Billing integrity** — mid-stream disconnect still bills; disconnect doesn't
  leak a capacity slot; provider crash bills only what was produced; failed calls
  refund; `settle` is idempotent; 50 concurrent reservations against a `$1.00`
  cap grant exactly 10.
- **Hostile input** — empty/whitespace/null-byte prompts; emoji and CJK never
  yield a zero token count (a zero count is a free request); oversized context;
  zero/negative budgets; single-model catalogs with nowhere to downgrade.
- **vLLM transport** — 429 retry, give-up after max retries, `4xx` never
  retried, timeouts vs refused connections, malformed SSE frames, missing
  `usage`.
- **Redis** — real Lua executed via `fakeredis`: atomicity under 50-way
  concurrency, shared budgets across instances, fail-closed vs fail-open,
  `NOSCRIPT` recovery after a restart, corrupt-member tolerance.
- **Capacity & scaling** — graceful shrink without interrupting in-flight work,
  waiters woken on grow, no flapping on spikes, `min > max` misconfiguration.

The two disconnect-billing tests are mutation-checked: deleting the finalizer
makes them fail.

## License

See [LICENSE](LICENSE).
