"""FastAPI application exposing the inference service.

Endpoints:
  POST /v1/infer         run inference (routing + cost control + generation)
  POST /v1/route/explain dry-run routing decision (no spend, no model call)
  GET  /v1/models        the model catalog with pricing
  GET  /v1/usage         per-user spend summary
  GET  /healthz          liveness
  GET  /metrics          Prometheus-format metrics (drives external autoscalers)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from .config import Settings
from .cost import UserBudgetExceededError
from .router import BudgetExceededError, ContextTooLargeError, route
from .schemas import (
    InferenceRequest,
    InferenceResponse,
    RouteExplainResponse,
)
from .service import InferenceService


def create_app(service: InferenceService | None = None) -> FastAPI:
    svc = service or InferenceService(Settings.from_env())

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

    app = FastAPI(
        title="LLM Inference Service",
        description="Autoscaling LLM inference with complexity-based model "
        "switching and cost controls.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "replicas": svc.autoscaler.current_replicas}

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
        except BudgetExceededError as e:
            raise HTTPException(status_code=402, detail=str(e))
        except ContextTooLargeError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
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
        except BudgetExceededError as e:
            raise HTTPException(status_code=402, detail=str(e))
        except UserBudgetExceededError as e:
            raise HTTPException(status_code=429, detail=str(e))
        except ContextTooLargeError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return InferenceResponse(**result.__dict__)

    @app.get("/v1/usage")
    async def usage():
        return {"spend_by_user": svc.ledger.summary()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return svc.metrics.prometheus()

    return app


app = create_app()
