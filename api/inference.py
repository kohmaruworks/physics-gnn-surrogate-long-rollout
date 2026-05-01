"""
Model loading and single-step inference with **index round-trip**.

Pipeline::

    JSON edges (1-based) → ``convert_julia_to_python_indices`` → model.forward (Heun)
    → tensor → JSON features; edges reconstructed via ``+1`` for Julia clients.

Only **single-graph PhysicsGNNSurrogate** checkpoints are supported (see README).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

_API_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _API_ROOT.parent
_SURROGATE = _REPO_ROOT / "surrogate_model"
if str(_SURROGATE) not in sys.path:
    sys.path.insert(0, str(_SURROGATE))

from model import PhysicsGNNSurrogate  # noqa: E402
from utils.index_converter import (  # noqa: E402
    assert_valid_python_edge_index,
    convert_julia_to_python_indices,
)


class InferenceRuntimeError(RuntimeError):
    """Raised when the checkpoint cannot be used with this API surface."""


_lock = Lock()
_model: Optional[PhysicsGNNSurrogate] = None
_device: torch.device = torch.device("cpu")
_checkpoint_path: Optional[str] = None
_model_meta: Dict[str, Any] = {}
_dt_default: float = 0.05


def _resolve_checkpoint_path() -> Path:
    raw = os.environ.get(
        "SURROGATE_CHECKPOINT",
        str(_REPO_ROOT / "data" / "interim" / "wave_rollout_step1_model.pth"),
    )
    return Path(raw).expanduser().resolve()


def configured_checkpoint_display() -> str:
    """Configured checkpoint path (environment or default), for logging and ``/health``."""
    return str(_resolve_checkpoint_path())


def _pick_device() -> torch.device:
    pref = os.environ.get("SURROGATE_DEVICE", "").lower().strip()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_runtime(*, force_reload: bool = False) -> None:
    """
    Load ``PhysicsGNNSurrogate`` weights once (singleton).

    Raises
    ------
    InferenceRuntimeError
        If the checkpoint denotes hierarchical / DDM-only weights (``bond`` in meta).
    FileNotFoundError
        If the checkpoint file is missing.
    """
    global _model, _device, _checkpoint_path, _model_meta, _dt_default

    with _lock:
        if _model is not None and not force_reload:
            return

        ckpt_path = _resolve_checkpoint_path()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta = payload.get("meta") or {}
        if "bond" in meta:
            raise InferenceRuntimeError(
                "This API serves single-graph Heun checkpoints only "
                "(hierarchical / multigrid weights require extra IR). "
                "Train or export a ``PhysicsGNNSurrogate`` .pth."
            )

        _device = _pick_device()
        dt = float(meta.get("dt", _dt_default))
        layers = int(meta.get("layers", meta.get("num_layers", 3)))
        hidden = int(meta["hidden"])

        model = PhysicsGNNSurrogate(
            state_dim=2,
            hidden_dim=hidden,
            num_message_layers=layers,
            dt=dt,
        ).to(_device)
        model.load_state_dict(payload["model"])
        model.eval()

        _model = model
        _checkpoint_path = str(ckpt_path)
        _model_meta = dict(meta)
        _dt_default = dt


def runtime_health_detail() -> Tuple[bool, Optional[str]]:
    if _model is None:
        return False, "model not loaded"
    return True, None


def python_edge_index_to_julia_pairs(edge_index: Tensor) -> List[List[int]]:
    """
    Convert PyG ``edge_index`` ``[2, E]`` (**0-based**) to Julia edge rows (**1-based**).
    """
    edge_index = edge_index.detach().cpu()
    out: List[List[int]] = []
    for i in range(edge_index.shape[1]):
        out.append([int(edge_index[0, i]) + 1, int(edge_index[1, i]) + 1])
    return out


def build_edge_index_from_julia(
    edges: List[List[int]],
    *,
    num_nodes: int,
    device: torch.device,
) -> Tensor:
    """Validate and convert **1-based** ``edges`` → ``[2, E]`` long tensor on ``device``."""
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ei_julia = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ei = convert_julia_to_python_indices(ei_julia, num_nodes=num_nodes)
    assert_valid_python_edge_index(ei, num_nodes=num_nodes)
    return ei.to(device)


@torch.no_grad()
def predict_step(
    *,
    num_nodes: int,
    node_features: List[List[float]],
    edges_julia: List[List[int]],
    dt_override: Optional[float],
) -> Tuple[List[List[float]], List[List[int]], Dict[str, Any]]:
    """
    Run one Heun step :math:`h^{t} \\mapsto h^{t+1}`.

    Indices are converted Julia→Python before ``forward`` and Python→Julia for the
    edge echo in the response.
    """
    if _model is None:
        raise RuntimeError("InferenceRuntime not initialized; call initialize_runtime() first.")

    if len(node_features) != num_nodes:
        raise ValueError(f"node_features rows ({len(node_features)}) != num_nodes ({num_nodes})")

    device = _device
    h = torch.tensor(node_features, dtype=torch.float32, device=device)
    if h.dim() != 2:
        raise ValueError("node_features must be 2-D after wrapping")

    try:
        ei = build_edge_index_from_julia(edges_julia, num_nodes=num_nodes, device=device)
    except JuliaIndexError as exc:
        raise ValueError(str(exc)) from exc

    edges_roundtrip = python_edge_index_to_julia_pairs(ei)

    dt_use = float(dt_override) if dt_override is not None else float(_model.dt)
    prev_dt = float(_model.dt)
    try:
        _model.dt = dt_use
        h_next = _model(h, ei)
    finally:
        _model.dt = prev_dt

    meta: Dict[str, Any] = {
        "architecture": "physics_gnn_surrogate_heun",
        "device": str(device),
        "dt": dt_use,
        "checkpoint": _checkpoint_path,
    }
    return h_next.detach().cpu().tolist(), edges_roundtrip, meta
