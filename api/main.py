"""
FastAPI application: health check and single-step surrogate inference.

Run from repository root::

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Environment:

- ``SURROGATE_CHECKPOINT``: path to ``PhysicsGNNSurrogate`` ``.pth`` (default: data/interim wave Step 1).
- ``SURROGATE_DEVICE``: ``cuda`` | ``cpu`` (optional).
"""

from __future__ import annotations

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictStepRequest, PredictStepResponse

try:
    from api import inference
except ImportError:  # pragma: no cover
    import inference  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load weights once at startup."""
    try:
        inference.initialize_runtime()
        app.state.model_error = None
    except Exception as exc:  # noqa: BLE001 - surface to /health
        app.state.model_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    yield


app = FastAPI(
    title="Physics GNN Surrogate · Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and checkpoint status."""
    ok, detail = inference.runtime_health_detail()
    err = getattr(app.state, "model_error", None)
    if err:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            checkpoint=inference.configured_checkpoint_display(),
            detail=err[:2000],
        )
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=ok,
        checkpoint=inference.configured_checkpoint_display(),
        detail=detail,
    )


@app.post("/predict_step", response_model=PredictStepResponse)
def predict_step(body: PredictStepRequest) -> PredictStepResponse:
    """
    One surrogate step with **1-based** edges in JSON; internally converted to 0-based PyG indices.
    """
    err = getattr(app.state, "model_error", None)
    if err:
        raise HTTPException(status_code=503, detail="Model failed to load at startup; see /health.")

    if len(body.node_features) != body.num_nodes:
        raise HTTPException(
            status_code=422,
            detail="num_nodes must match len(node_features)",
        )

    try:
        feats_next, edges_j, meta = inference.predict_step(
            num_nodes=body.num_nodes,
            node_features=body.node_features,
            edges_julia=body.edges,
            dt_override=body.dt_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return PredictStepResponse(
        node_features_next=feats_next,
        edges_julia=edges_j,
        meta=meta,
    )
