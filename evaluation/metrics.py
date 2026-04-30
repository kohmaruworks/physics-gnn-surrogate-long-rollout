"""
Evaluation metrics independent of training pipelines (reusable / functor-friendly).

Discrete Hamiltonian matches Step 1 surrogate convention::
    H = KE + PE = 1/2 Σ v_i² + (λ/2) Σ_edges (u_i - u_j)²
with λ ≈ (c/dx)²/2 for bidirected grid edges.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def discrete_hamiltonian(
    state: Tensor,
    edge_index: Tensor,
    *,
    lambda_edges: float,
    u_channel: int = 0,
    v_channel: int = 1,
) -> Tensor:
    """
    Scalar ``H`` for a single time slice ``state``: ``[N, F]``.
    """
    u = state[:, u_channel].reshape(-1)
    v = state[:, v_channel].reshape(-1)
    ke = 0.5 * (v * v).sum()
    if edge_index.numel() == 0:
        pe = ke.new_zeros(())
    else:
        row, col = edge_index[0], edge_index[1]
        du = u[row] - u[col]
        lam = float(lambda_edges)
        pe = 0.5 * lam * (du * du).sum()
    return ke + pe


def compute_rollout_rmse(
    pred: Tensor,
    target: Tensor,
    *,
    exclude_t0: bool = True,
) -> float:
    """
    Cumulative rollout RMSE::

        sqrt( 1 / (T * |V| * F) * Σ_{t,i,f} (pred - target)² )

    Parameters
    ----------
    pred, target:
        ``[T_roll, N, F]`` aligned shapes (e.g. ``F=2`` for ``u,v``).
    exclude_t0:
        If ``True``, drops index 0 (initial condition copy) so error counts only
        predicted steps ``t = 1 … T-1`` (matches common autoregressive protocols).
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
    if pred.dim() != 3:
        raise ValueError("pred/target must be [T, N, F]")
    p = pred[1:] if exclude_t0 else pred
    g = target[1:] if exclude_t0 else target
    if p.numel() == 0:
        raise ValueError("no timesteps to score")
    mse = torch.mean((p - g) ** 2)
    return float(torch.sqrt(mse).item())


def compute_energy_drift(
    pred_rollout: Tensor,
    edge_index: Tensor,
    *,
    reference_state_t0: Tensor,
    lambda_edges: float,
    eps: float = 1e-12,
) -> float:
    """
    Maximum relative deviation of predicted Hamiltonian from initial **reference** energy::

        Drift_max = max_t | (H(pred_t) - H(ref_0)) / H(ref_0) |

    Parameters
    ----------
    pred_rollout:
        ``[T, N, F]`` autoregressive predictions (may include ``t=0`` matching GT IC).
    reference_state_t0:
        ``[N, F]`` ground-truth state at ``t=0`` used for normalization denominator.
    """
    if pred_rollout.dim() != 3:
        raise ValueError("pred_rollout must be [T, N, F]")
    h0 = discrete_hamiltonian(reference_state_t0, edge_index, lambda_edges=lambda_edges)
    denom = float(abs(h0.item()))
    if denom < eps:
        denom = eps
    max_d = pred_rollout.new_zeros(())
    for t in range(pred_rollout.size(0)):
        ht = discrete_hamiltonian(pred_rollout[t], edge_index, lambda_edges=lambda_edges)
        rel = torch.abs((ht - h0) / denom)
        max_d = torch.maximum(max_d, rel)
    return float(max_d.item())


def effective_lambda_edges(*, c: float, dx: float) -> float:
    """Bidirected grid stiffness scaling (same as training helpers)."""
    return 0.5 * (float(c) / float(dx)) ** 2
