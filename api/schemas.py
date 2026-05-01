"""
Pydantic request/response contracts for the surrogate inference HTTP API.

Clients send **Julia-style 1-based** edge endpoints; responses echo edges in the same convention.
Node feature matrices use global node order ``1 … N`` aligned with those endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class PredictStepRequest(BaseModel):
    """
    One-step prediction payload.

    ``edges`` uses inclusive **1-based** vertex IDs as in Julia-exported JSON.
    """

    num_nodes: int = Field(..., ge=1, description="Number of graph nodes N.")
    node_features: List[List[float]] = Field(
        ...,
        description="Shape [N, state_dim]; row i is node i+1 in 1-based Julia numbering.",
    )
    edges: List[List[int]] = Field(
        ...,
        description="Directed edges as pairs [src, dst], each in 1..N (Julia convention).",
    )
    dt_override: Optional[float] = Field(
        None,
        gt=0.0,
        description="Optional macro timestep; defaults to checkpoint meta dt.",
    )

    @field_validator("edges")
    @classmethod
    def _edges_pairs(cls, v: List[List[int]]) -> List[List[int]]:
        for e in v:
            if len(e) != 2:
                raise ValueError("each edge must be [src, dst] with length 2")
            if e[0] < 1 or e[1] < 1:
                raise ValueError("edge endpoints must be >= 1 (1-based)")
        return v

    @field_validator("node_features")
    @classmethod
    def _feat_nonempty(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("node_features must be non-empty")
        row_len = len(v[0])
        if row_len < 1:
            raise ValueError("state_dim must be >= 1")
        for row in v[1:]:
            if len(row) != row_len:
                raise ValueError("all rows of node_features must share state_dim")
        return v

    @model_validator(mode="after")
    def _edges_within_num_nodes(self) -> PredictStepRequest:
        for a, b in self.edges:
            if a > self.num_nodes or b > self.num_nodes:
                raise ValueError(
                    f"edge ({a},{b}) exceeds num_nodes={self.num_nodes} (1-based range)"
                )
        return self


class PredictStepResponse(BaseModel):
    """One-step output: next state and edges echoed in **1-based** form."""

    node_features_next: List[List[float]] = Field(
        ...,
        description="h^{t+1} with same layout as request node_features [N, state_dim].",
    )
    edges_julia: List[List[int]] = Field(
        ...,
        description="Edge list [src, dst] round-tripped through 0-based PyTorch and back (1-based).",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostics: architecture key, device, dt used.",
    )


class HealthResponse(BaseModel):
    """Service liveness and model readiness."""

    status: str
    model_loaded: bool
    checkpoint: Optional[str] = None
    detail: Optional[str] = None
