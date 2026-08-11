# End-to-End Implementation Guide

This document walks through how the service is built, one layer at a time, and
why each piece exists. Read it top to bottom to understand the whole system, or
jump to a step. Every step maps to one module under `app/`.

The product goal:

> An LLM inference service that **autoscales** with load, enforces **cost
> controls**, and **switches models based on task complexity** so simple
> requests run on cheap models and only hard requests pay for the expensive one
> — while continuously checking projected **token consumption** against a
> budget.

The request lifecycle these steps assemble:

```
            ┌──────────────────────────────────────────────────────────────┐
 prompt ──► │ 1. estimate_complexity ─► 2. route (+ budget switch) ─►        │
            │ 3. ledger.reserve ─► 4. capacity gate (queue) ─►               │
            │ 5. provider.generate ─► 6. ledger.settle ─► 7. metrics ─►      │
            │ 8. autoscaler reacts to load                                   │
            └──────────────────────────────────────────────────────────────┘
```

---

## Step 0 — Project layout

```
app/
  config.py            model catalog + pricing + settings  (source of truth)
  tokens.py            token estimation (tiktoken or heuristic)
  complexity.py        task-complexity classifier
  router.py            model selection + token/cost budget switch
  cost/
    base.py            ledger protocol: reserve / settle / release
    memory.py          in-process ledger (single node, tests)
    redis_ledger.py    distributed ledger, atomic via Lua
  providers/
    base.py            Provider protocol, Completion, StreamChunk
    mock.py            deterministic offline provider (+ failure injection)
    vllm.py            vLLM OpenAI-compatible adapter, buffered + streaming
  gate.py              resizable concurrency gate (admission control)
  autoscaler.py        replica-count control loop
  metrics.py           in-memory counters + Prometheus export
  service.py           orchestration: buffered + streaming paths
  factory.py           builds the service from Settings
  main.py              FastAPI HTTP surface
tests/                 one module per component + edge-case suites
examples/demo.py       runnable, no API keys
deploy/hpa.yaml        Deployment + HPA + PodDisruptionBudget
```

Design rule: each module does one thing and depends only on `config` and the
modules "below" it. That keeps every piece independently testable.

---

## Step 1 — The model catalog and pricing (`config.py`)

Everything starts from a catalog. A `ModelSpec` records the price (USD per 1k
input and output tokens), the **complexity tier** the model is trusted to
handle, and its context window.

```python
class Complexity(IntEnum):   # ordered so COMPLEX > MODERATE > SIMPLE
    SIMPLE = 1; MODERATE = 2; COMPLEX = 3

@dataclass(frozen=True)
class ModelSpec:
    name: str; tier: Complexity
    input_price_per_1k: float; output_price_per_1k: float
    max_context_tokens: int
    def cost(self, in_tok, out_tok): ...
```

Three tiers ship by default — `small`, `medium`, `large` — with realistic price
ratios (large ≈ 60× small). `Settings` holds the cost/scaling knobs and reads
overrides from environment variables, so the same image runs in dev and prod.

**Why an `IntEnum`?** Routing needs "any model whose tier ≥ the task's tier".
Ordered enums make that a one-line comparison.

---

## Step 2 — Token estimation (`tokens.py`)

Both routing and cost projection need a token count *before* the call. We hide
this behind `count_tokens(text)`:

- if `tiktoken` is installed, use it for accuracy;
- otherwise fall back to `len(text) / chars_per_token` (default 4).

`estimate_output_tokens()` projects completion length (≈75% of the requested
max) so we can reserve budget up front. Real usage is billed after the call.

**Why a heuristic fallback?** The service must run and test with zero external
dependencies. Accuracy from a real tokenizer is a drop-in upgrade, not a
requirement.

---

## Step 3 — Complexity classification (`complexity.py`)

This is the "task complexity" input to model switching. `estimate_complexity()`
scores a prompt **without calling a model** (so it adds no cost or latency) by
blending explainable signals:

| Signal | Weight | Rationale |
|---|---|---|
| reasoning keywords (`analyze`, `design`, `prove`, `debug`, …) | 0.60 | most reliable indicator of real effort |
| prompt length (tokens) | 0.15 | long context ⇒ harder task |
| requested output length | 0.15 | long answer ⇒ more work |
| contains a code block | +0.15 | engineering task |

The blended 0–1 score buckets into `SIMPLE` / `MODERATE` / `COMPLEX`. An
explicit `task_type` from a trusted caller **overrides** the heuristic. Every
decision returns `reasons`, so routing is auditable:

```python
estimate_complexity("Analyze and design an algorithm, prove correctness")
# tier=COMPLEX, score=0.60, reasons=[... reasoning_keywords=[analyze, design, prove] ...]
```

**Why heuristic, not a classifier model?** Using a model to route to a model is
circular and adds cost/latency to *every* request. A transparent heuristic is
cheap, fast, and debuggable; you can swap in a small classifier later behind the
same function signature.

---

## Step 4 — Routing + the token/cost budget switch (`router.py`)

This is the core feature. `route()` runs four steps:

1. **Classify** — get the minimum capable tier from Step 3.
2. **Pick the cheapest capable model** — among models with `tier ≥ task tier`,
   ranked by price.
3. **Project token consumption and its cost** — `input_tokens` from the prompt,
   `projected_output_tokens` from the request, then `model.cost(...)`.
4. **Budget check → model switch** — if the projected cost exceeds the
   per-request budget, **switch down** to the cheapest model that *does* fit,
   flagging the result as `downgraded`. If nothing fits, raise
   `BudgetExceededError`.

```python
d = route("say hi", settings)                       # -> small, not downgraded
d = route(complex_prompt, settings, budget_usd=1.0) # -> large
d = route(complex_prompt, settings, budget_usd=0.001, max_output_tokens=100)
#    -> downgraded=True, switched off "large" to fit the budget
```

There is also a `force_model` escape hatch (still validated for budget and
context window) and a `ContextTooLargeError` when the prompt exceeds every
model's window.

**Why switch *down* on budget, not reject?** For many workloads a cheaper answer
now beats no answer. The caller sees `downgraded=True` and can decide whether the
trade-off was acceptable; a strict caller can set a budget high enough to forbid
downgrades, or read the flag and retry.

---

## Step 5 — Cost controls: the spend ledger (`cost/memory.py`)

Routing decides what a request *would* cost; the ledger decides whether the user
is *allowed* to spend it. It enforces a **rolling daily budget per user** with a
reserve → settle protocol:

- `reserve(user, amount, limit)` — atomically check-and-hold the projected cost
  **before** the call. Holding up front stops concurrent requests from each
  passing the check and collectively blowing the budget.
- `settle(reservation, actual_cost, …)` — after the call, replace the hold with
  the real cost computed from actual token usage.
- `release(reservation)` — drop the hold without charging if the call failed.

Spend older than the window (24h default) is pruned, so the budget rolls. The
clock is injectable, which makes the window deterministically testable. The
implementation is in-memory and thread-safe; swapping in Redis/Postgres changes
the storage, not the interface.

**Why reserve-then-settle instead of "charge after"?** Charging only after the
call lets a burst of concurrent requests all pass a stale budget check and
overspend. Reserving first makes the check-and-charge atomic.

---

## Step 6 — Provider abstraction (`providers/base.py`)

The rest of the system talks to one `Provider` protocol with a single
`generate()` method, so no business logic depends on a specific vendor. A
deterministic `MockProvider` ships by default — it returns reproducible output
and models latency as inversely proportional to model speed, so the whole
service runs and tests offline. A real adapter (OpenAI / Anthropic / vLLM) just
implements `generate()`.

---

## Step 7 — Orchestration (`service.py`)

`InferenceService.handle()` is the conductor that runs steps 1–8 in order:

```
route()            # 1–4: complexity + budget switch
ledger.reserve()   # 5: hold projected cost against the daily budget
capacity gate      # enter the concurrency semaphore; waiting here == queue depth
provider.generate()# 6: run the model
ledger.settle()    # reconcile hold with real token usage
metrics.record()   # 7: expose load + spend
```

The **capacity gate** is an `asyncio.Semaphore` sized to
`replicas × target_concurrency_per_replica`. Requests that exceed capacity wait
on it — and that waiting count *is* the queue depth the autoscaler reads. On any
failure the reservation is released so the user is never charged for a failed
call.

---

## Step 8 — Autoscaling (`autoscaler.py`)

A small control loop turns load into a desired replica count:

```
desired = ceil((in_flight + queue_depth) / target_concurrency_per_replica)
desired = clamp(desired, min_replicas, max_replicas)
```

It **scales up fast and down slow** using asymmetric cooldowns (short up,
long down) to avoid thrashing — the standard autoscaler pattern. `service.py`
calls `autoscale_tick()` on a background loop and resizes the capacity gate to
match. The same load signal is exported at `/metrics`, so an external
Kubernetes HPA (see `deploy/hpa.yaml`) can add/remove *pods* using the same
number the in-process gate uses for *concurrency*.

**Two layers of scaling:** in-process concurrency handles bursts within a pod
instantly; the HPA handles sustained load by adding pods. They read the same
metric so they never fight.

---

## Step 9 — HTTP surface (`main.py`)

FastAPI exposes the service. Domain exceptions map to HTTP status codes so
clients get actionable errors:

| Endpoint | Purpose |
|---|---|
| `POST /v1/infer` | run inference (routing + cost control + generation) |
| `POST /v1/route/explain` | **dry-run** the routing decision — no spend, no model call |
| `GET /v1/models` | catalog + pricing |
| `GET /v1/usage` | per-user spend summary |
| `GET /metrics` | Prometheus text (drives external autoscalers) |
| `GET /healthz` | liveness + current replica count |

| Failure | Status |
|---|---|
| per-request budget can't be met | `402 Payment Required` |
| user daily budget exhausted | `429 Too Many Requests` |
| prompt exceeds every context window | `413 Payload Too Large` |
| bad model / task_type | `400 Bad Request` |

`/v1/route/explain` is worth calling out: it lets a client see *which model would
run and what it would cost* before committing to spend — useful for building UIs
and for testing routing policy.

---

## Step 10 — Real inference with vLLM (`providers/vllm.py`)

`VLLMProvider` talks to a vLLM server through its OpenAI-compatible API
(`/v1/completions`), which is what `vllm serve <model>` exposes. Switching to it
is configuration, not code: `LLM_PROVIDER=vllm`.

`model_map` translates catalog names into the model IDs the server was launched
with:

```bash
LLM_PROVIDER=vllm
LLM_VLLM_BASE_URL=http://vllm:8000
LLM_VLLM_MODEL_MAP="small=Qwen/Qwen2.5-1.5B-Instruct,large=Qwen/Qwen2.5-32B-Instruct"
```

Four production concerns are handled inside the adapter:

| Concern | Handling | Why |
|---|---|---|
| Slow vs dead server | separate connect (5s) and read (120s) timeouts | a dead server should fail fast; a long generation is legitimate |
| Transient faults | retry with exponential backoff on connect errors, 429, 5xx | vLLM returns 429 when its scheduler queue is full |
| Client errors | **never** retried (400 → immediate raise) | retrying our own bug amplifies load during an incident |
| Billing accuracy | prefer server-reported `usage` over local estimates | the server's tokenizer is authoritative; fall back to estimation only if `usage` is absent |

Mid-stream failures are deliberately *not* retried: bytes already reached the
client, so a restart would duplicate output. The failure propagates and the
service bills for the partial generation.

---

## Step 11 — Distributed budgets with Redis (`cost/redis_ledger.py`)

The in-memory ledger enforces budgets **per process**. Run four replicas and
each user effectively gets 4× their budget — the single most common way a cost
control silently stops working once you scale out.

`RedisLedger` makes the budget global. The critical detail is **atomicity**:

```
ZSET  llm:spend:{user}   member "{entry_id}:{amount}"   score = unix_ts
HASH  llm:usage:{user}   field  "total" -> lifetime settled spend
```

A read-then-write from Python has a race window where N replicas all read the
same balance and all conclude there is room. So prune → sum → compare → write
happens inside a **Lua script**, which Redis executes atomically:

```lua
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)  -- roll the window
-- sum live spend from members
if total + amount > limit then return {0, tostring(total)} end
redis.call('ZADD', key, now, member)                        -- take the hold
```

The test suite proves this: 50 concurrent `$0.10` reservations against a `$1.00`
cap grant **exactly 10**.

Three operational details worth knowing:

- **NOSCRIPT recovery** — the cached script SHA dies with a Redis restart, so
  `EVALSHA` failures transparently fall back to `EVAL` and reload.
- **`fail_closed` policy** — when Redis is unreachable, do we reject (protect
  spend) or allow (protect availability)? Default is **fail-closed**: a
  cost-control outage must not become an unmetered-spend incident. Flip it with
  `LLM_REDIS_FAIL_CLOSED=false` if availability genuinely outranks accuracy.
- **Corrupt members are skipped**, not fatal — one bad entry from an old schema
  must not break billing for an entire user.

`/readyz` checks the ledger, so a pod that cannot enforce budgets leaves the
load-balancer rotation rather than rejecting every request it receives.

---

## Step 12 — Streaming, and the money it can leak

Streaming is where cost control gets genuinely hard. The naive implementation
bills on clean completion — which means **any client can stream and then
disconnect to get the generation free**. The tokens were still produced; the GPU
time was still spent.

The provider interface therefore gains `generate_stream()`, yielding
`StreamChunk`s with `usage` on the terminal chunk. `InferenceService.handle_stream()`
has three termination paths, and all three must settle:

| Path | Billing |
|---|---|
| normal completion | provider-reported usage |
| provider error mid-stream | partial output, `error` event, exception propagates |
| **client disconnect** | partial output, reconstructed from accumulated text |

The subtle part is *how* cleanup runs. A disconnect closes the async generator
at its `yield`, and if the request task is being cancelled, any `await` in the
generator's `finally` is cancelled too — losing the billing write. So
finalization is dispatched to a **detached task**:

```python
finally:
    finalize()          # synchronous: schedules the work, never awaits

def _finalize_later(...):
    task = asyncio.create_task(_run())   # outside the request's cancel scope
    self._pending.add(task)              # keep a strong ref; asyncio does not
    task.add_done_callback(self._pending.discard)
```

`drain_settlements()` awaits those tasks — used at shutdown and in tests. Both
disconnect tests were mutation-checked: removing `finalize()` makes them fail.

On the HTTP side, `/v1/infer/stream` pulls the **first event eagerly** before
returning a `StreamingResponse`. That event is produced after routing and budget
admission, so a rejected request still gets a real `402`/`429` instead of a
`200` with an error buried in the body — clients key retries off status codes.

---

## Step 13 — A capacity gate that can shrink (`gate.py`)

`asyncio.Semaphore` can grow (release extra permits) but cannot shrink: taking
permits back means acquiring them, which blocks. The autoscaler needs both
directions, so `CapacityGate` tracks capacity explicitly behind a condition
variable.

Shrinking is **graceful** — in-flight generations are never interrupted; the gate
simply stops admitting new work until `active` drops below the new capacity. The
gate also owns the `waiting` counter, which *is* the queue-depth signal the
autoscaler scales on.

> Scale on queue depth, not CPU. A pod blocked on a GPU is not CPU-busy, so
> CPU-based scaling under-reacts exactly when you need it most.

---

## Step 14 — Verifying it

```bash
pip install -r requirements-dev.txt
pytest                        # 103 tests
python examples/demo.py       # end-to-end, no API keys
uvicorn app.main:app --reload
```

### The edge cases that matter

`tests/test_edge_cases.py` targets failure modes that never appear in a
happy-path demo. Each test names the incident it guards against:

**Billing integrity**
- client disconnects mid-stream → still billed for partial output
- disconnect releases its capacity slot (leaked slots silently erode throughput
  until the service deadlocks with an idle GPU)
- provider crash mid-stream → billed for what was produced, not the full
  reservation and not zero
- failed buffered call → reservation refunded, user charged `$0`
- `settle` is idempotent; `release` after `settle` does not erase a real charge
- actual cost may exceed the reservation (projections are estimates)
- 50 concurrent reservations against a `$1.00` cap grant exactly 10
- a request landing exactly on the limit is allowed (float epsilon)

**Rolling-window boundaries** — spend expires exactly at the edge; a capped-out
user can spend again after the window rolls.

**Hostile input** — empty/whitespace/null-byte prompts; emoji, CJK, accented and
mathematical-bold text never yield a zero token count (a zero count is a free
request); prompts exceeding every context window; `max_output_tokens` larger than
the window; zero and negative budgets; a single-model catalog with nowhere to
downgrade; an unknown `task_type` falling back to the heuristic rather than
silently routing everything to SIMPLE.

**vLLM transport** (`test_vllm_provider.py`, fake `httpx` transport — no server
needed) — 429 retry-then-succeed; give-up after max retries; 400 never retried;
timeouts vs connection refused; malformed SSE frames skipped; missing `usage`
falling back to estimation; empty `choices`.

**Redis** (`test_redis_ledger.py`, real Lua via `fakeredis`) — Lua atomicity
under 50-way concurrency; two ledger instances sharing one budget; fail-closed
vs fail-open on outage; `NOSCRIPT` fallback after a Redis restart; corrupt
members skipped.

**Capacity & scaling** — gate blocks past capacity and reports queue depth;
shrink does not interrupt in-flight work; grow wakes waiters; autoscaler does not
flap on a spike; survives `min > max` misconfiguration; counts queue depth rather
than in-flight alone (scaling on in-flight alone means a saturated service never
scales).

---

## Extending it

- **Other providers:** implement `generate()` + `generate_stream()`; everything
  above is provider-agnostic.
- **Postgres ledger:** implement the same reserve/settle/release protocol; the
  service does not care which backend it holds.
- **Smarter routing:** replace the heuristic in `complexity.py` with a small
  classifier; the `route()` contract doesn't change.
- **Prefix caching:** vLLM reuses KV cache across shared prefixes — routing
  cache-friendly requests to the same replica would cut cost further.
