"""
Physics-informed losses for long-rollout stability (symplectic / energy drift).

Energy model (discrete Hamiltonian surrogate):
    H = KE + PE = (1/2) Σ_i v_i² + (λ_edges/2) Σ_{(i,j)∈ℰ} (u_i - u_j)²

where ``λ_edges`` ties discrete stiffness to the simulation metadata (``c²/dx²``).
Loss penalizes squared drift ``(H_{t+1} - H_t)²`` summed over time.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class SymplecticLoss(nn.Module):
    """
    Penalize temporal change in scalar energy ``H`` reconstructed from predicted
    displacement ``u`` and velocity ``v`` (channels are explicit for morphism clarity).
    """

    def __init__(self, *, lambda_edges: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.lambda_edges = float(lambda_edges)
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def hamiltonian(
        self,
        u: Tensor,
        v: Tensor,
        edge_index: Tensor,
        *,
        lambda_edges: Optional[float] = None,
    ) -> Tensor:
        """
        Compute per-graph Hamiltonian scalar ``H`` for states ``u``, ``v``.

        Parameters
        ----------
        u, v:
            ``[N]`` or ``[N, 1]`` or broadcastable to node dimension.
        edge_index:
            ``[2, E]`` PyG indices (0-based).
        lambda_edges:
            Optional override for stiffness weight on edge differences.

        Returns
        -------
        torch.Tensor
            Scalar ``H`` (0-dim).
        """
        lam = float(lambda_edges) if lambda_edges is not None else self.lambda_edges
        u_n = u.reshape(-1)
        v_n = v.reshape(-1)
        ke = 0.5 * (v_n * v_n).sum()
        if edge_index.numel() == 0:
            pe = u_n.new_zeros(())
        else:
            row, col = edge_index[0], edge_index[1]
            du = u_n[row] - u_n[col]
            pe = 0.5 * lam * (du * du).sum()
        return ke + pe

    def forward(
        self,
        states: Tensor,
        edge_index: Tensor,
        *,
        u_channels: tuple[int, ...] = (0,),
        v_channels: tuple[int, ...] = (1,),
        lambda_edges: Optional[float] = None,
    ) -> Tensor:
        """
        Sum or mean over time of ``(H_{k+1} - H_k)²`` along batch dimension 0.

        Parameters
        ----------
        states:
            ``[T, N, C]`` trajectory (e.g. predicted rollout).
        edge_index:
            ``[2, E]`` fixed topology (0-based).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if states.dim() != 3:
            raise ValueError(f"states must be [T, N, C], got {tuple(states.shape)}")
        T = states.size(0)
        if T < 2:
            return states.new_zeros(())
        lam = float(lambda_edges) if lambda_edges is not None else self.lambda_edges

        H_list = []
        for t in range(T):
            st = states[t]
            u = st[:, list(u_channels)].sum(dim=-1)
            v = st[:, list(v_channels)].sum(dim=-1)
            H_list.append(self.hamiltonian(u, v, edge_index, lambda_edges=lam))
        H = torch.stack(H_list, dim=0)
        dH = H[1:] - H[:-1]
        loss = (dH * dH).sum()
        if self.reduction == "mean":
            loss = loss / float(T - 1)
        return loss
