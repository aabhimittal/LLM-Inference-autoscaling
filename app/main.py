"""FastAPI application exposing the inference service.

Endpoints:
  POST /v1/infer          run inference (routing + cost control + generation)
  POST /v1/infer/stream   same, streamed as Server-Sent Events
  POST /v1/route/explain  dry-run routing decision (no spend, no model call)
  GET  /v1/models         the model catalog with pricing
  GET  /v1/usage          per-user spend summary
  GET  /healthz           liveness
  GET  /readyz            readiness (backends reachable)
  GET  /metrics           Prometheus-format metrics (drives external autoscalers)
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from .breaker import CircuitOpenError
from .cost import LedgerUnavailableError, UserBudgetExceededError
from .factory import build_service
from .providers import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from .ratelimit import RateLimitExceededError
from .router import (
    BudgetExceededError,
    ContextTooLargeError,
    NoModelAvailableError,
    route,
)
from .schemas import InferenceRequest, InferenceResponse, RouteExplainResponse
from .service import AdmissionTimeoutError, InferenceService

# Domain exception -> HTTP status. Kept in one place so the buffered and
# streaming paths cannot drift apart. Order matters: the first match wins, so
# subclasses must precede their base (ProviderError is last for that reason).
_STATUS_MAP = [
    (BudgetExceededError, 402),       # request budget cannot be met
    (UserBudgetExceededError, 429),   # user daily budget exhausted
    (RateLimitExceededError, 429),    # too many requests
    (ContextTooLargeError, 413),      # prompt exceeds every context window
    (LedgerUnavailableError, 503),    # cost store down, policy is fail-closed
    (AdmissionTimeoutError, 503),     # queue too deep, request shed
    (CircuitOpenError, 503),          # backend circuit open, failing fast
    (NoModelAvailableError, 503),     # every candidate model excluded
    (ProviderTimeoutError, 504),      # upstream model server too slow
    (ProviderUnavailableError, 503),  # upstream model server unreachable
    (ProviderError, 502),             # upstream returned something unusable
    (ValueError, 400),                # bad model name / task type
]

# Errors that should tell the client when to come back. Sending Retry-After
# turns a blind retry storm into coordinated backoff.
_RETRY_AFTER_ATTR = {
    RateLimitExceededError: "retry_after_s",
    CircuitOpenError: "retry_after_s",
}


def _status_for(exc: Exception) -> int | None:
    for exc_type, status in _STATUS_MAP:
        if isinstance(exc, exc_type):
            return status
    return None


def _as_http(exc: Exception) -> HTTPException:
    status = _status_for(exc)
    if status is None:
        raise exc
    headers = None
    for exc_type, attr in _RETRY_AFTER_ATTR.items():
        if isinstance(exc, exc_type):
            seconds = getattr(exc, attr, None)
            if seconds is not None and seconds != float("inf"):
                headers = {"retry-after": str(max(1, int(seconds) + 1))}
            break
    return HTTPException(status_code=status, detail=str(exc), headers=headers)


def create_app(service: InferenceService | None = None) -> FastAPI:
    svc = service or build_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = asyncio.Event()
        task = asyncio.create_task(svc.run_autoscaler(interval_s=5.0, stop=stop))
        app.state.service = svc
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            # Flush any in-flight billing before the process exits.
            await svc.aclose()

    app = FastAPI(
        title="LLM Inference Service",
        description="Autoscaling LLM inference with complexity-based model "
        "switching and cost controls.",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "replicas": svc.autoscaler.current_replicas}

    @app.get("/readyz")
    async def readyz():
        """Readiness: can we actually reach the cost store?

        Liveness says the process is up; readiness says it can serve. With a
        fail-closed Redis ledger an unreachable store means every request would
        be rejected, so the pod should leave the load-balancer rotation.
        """
        try:
            await svc.ledger.spent("__readyz__")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ledger unavailable: {e}")
        return {"status": "ready", "replicas": svc.autoscaler.current_replicas}

    @app.get("/v1/models")
    async def models():
        return {
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "tier": m.tier.name,
                    "input_price_per_1k": m.input_price_per_1k,
                    "output_price_per_1k": m.output_price_per_1k,
                    "max_context_tokens": m.max_context_tokens,
                }
                for m in svc.settings.catalog
            ]
        }

    @app.post("/v1/route/explain", response_model=RouteExplainResponse)
    async def route_explain(req: InferenceRequest):
        try:
            decision = route(
                req.prompt,
                svc.settings,
                task_type=req.task_type,
                max_output_tokens=req.max_output_tokens,
                budget_usd=req.budget_usd,
                force_model=req.force_model,
            )
        except Exception as e:
            raise _as_http(e)
        return RouteExplainResponse(
            model=decision.model.name,
            complexity=decision.complexity.tier.name,
            complexity_score=round(decision.complexity.score, 4),
            complexity_reasons=decision.complexity.reasons,
            input_tokens=decision.input_tokens,
            projected_output_tokens=decision.projected_output_tokens,
            projected_cost_usd=round(decision.projected_cost, 6),
            downgraded=decision.downgraded,
            routing_notes=decision.notes,
        )

    @app.post("/v1/infer", response_model=InferenceResponse)
    async def infer(req: InferenceRequest):
        try:
            result = await svc.handle(
                req.prompt,
                user_id=req.user_id,
                task_type=req.task_type,
                max_output_tokens=req.max_output_tokens,
                budget_usd=req.budget_usd,
                user_daily_budget_usd=req.user_daily_budget_usd,
                force_model=req.force_model,
            )
        except Exception as e:
            raise _as_http(e)
        return InferenceResponse(**result.__dict__)

    @app.post("/v1/infer/stream")
    async def infer_stream(req: InferenceRequest):
        """Stream a completion as Server-Sent Events.

        Routing and budget admission happen *before* the response starts, so a
        rejected request still gets a proper 4xx status rather than a 200 with an
        error buried in the stream. Once tokens are flowing the status is already
        committed, so mid-stream failures are reported as an ``error`` event.
        """
        stream = svc.handle_stream(
            req.prompt,
            user_id=req.user_id,
            task_type=req.task_type,
            max_output_tokens=req.max_output_tokens,
            budget_usd=req.budget_usd,
            user_daily_budget_usd=req.user_daily_budget_usd,
            force_model=req.force_model,
        )

        # Pull the first event eagerly: it is produced after routing and the
        # budget reservation, so any admission error surfaces here as a status
        # code instead of a half-written 200 response.
        try:
            first = await stream.__anext__()
        except StopAsyncIteration:
            raise HTTPException(status_code=500, detail="empty stream")
        except Exception as e:
            await stream.aclose()
            raise _as_http(e)

        async def body() -> AsyncIterator[bytes]:
            try:
                yield _sse(first)
                async for event in stream:
                    yield _sse(event)
            except Exception:
                # handle_stream already emitted an "error" event before raising;
                # the connection simply ends here. Billing for partial output is
                # handled by the service's detached finalizer.
                pass
            finally:
                yield b"data: [DONE]\n\n"

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
                # Stop nginx from buffering the stream into one big response.
                "x-accel-buffering": "no",
            },
        )

    @app.get("/v1/usage")
    async def usage():
        try:
            return {"spend_by_user": await svc.ledger.summary()}
        except Exception as e:
            raise _as_http(e)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        svc.metrics.set_queue_depth(svc.gate.waiting)
        return svc.metrics.prometheus()

    @app.get("/v1/status")
    async def status():
        """Operational snapshot: what is scaled, cached, and broken right now."""
        snap = svc.metrics.snapshot()
        return {
            "replicas": svc.autoscaler.current_replicas,
            "capacity": svc.gate.capacity,
            "in_flight": svc.gate.active,
            "queue_depth": svc.gate.waiting,
            "cache_hit_rate": snap["cache_hit_rate"],
            "circuits": svc.breakers.states(),
            "open_circuits": svc.breakers.open_backends(),
            "shed": snap["total_shed"],
            "rate_limited": snap["total_rate_limited"],
            "failovers": snap["total_failovers"],
        }

    return app


def _sse(event) -> bytes:
    payload = {"type": event.type}
    if event.text:
        payload["text"] = event.text
    if event.data:
        payload.update(event.data)
    return f"data: {json.dumps(payload)}\n\n".encode()


app = create_app()
