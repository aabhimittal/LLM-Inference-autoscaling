"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to run inference on.")
    user_id: str = Field("anonymous", description="Caller identity for budget tracking.")
    task_type: Optional[str] = Field(
        None,
        description="Optional explicit complexity: simple|moderate|complex. "
        "Overrides the automatic classifier.",
    )
    max_output_tokens: Optional[int] = Field(
        None, ge=1, description="Requested completion length."
    )
    budget_usd: Optional[float] = Field(
        None, ge=0, description="Per-request USD budget. Triggers model switching."
    )
    user_daily_budget_usd: Optional[float] = Field(
        None, ge=0, description="Override the user's rolling daily budget."
    )
    force_model: Optional[str] = Field(
        None, description="Pin a specific model, bypassing complexity routing."
    )


class InferenceResponse(BaseModel):
    text: str
    model: str
    complexity: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    downgraded: bool = Field(
        ..., description="True if the budget check switched to a cheaper model."
    )
    routing_notes: List[str]


class RouteExplainResponse(BaseModel):
    """Dry-run: what *would* happen, without spending budget or calling a model."""

    model: str
    complexity: str
    complexity_score: float
    complexity_reasons: List[str]
    input_tokens: int
    projected_output_tokens: int
    projected_cost_usd: float
    downgraded: bool
    routing_notes: List[str]
