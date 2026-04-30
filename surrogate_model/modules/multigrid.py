"""
Multigrid functors: Restriction ``R`` and Prolongation ``P`` as fixed sparse linear maps.

Morphism signatures:
    ``R: ℝ^{N_f × F} → ℝ^{N_c × F}``, ``P: ℝ^{N_c × F} → ℝ^{N_f × F}`` via batched ``torch.sparse.mm``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Restriction(nn.Module):
    """
    Apply a sparse restriction matrix ``R ∈ ℝ^{N_c × N_f}`` (e.g. full-weight averaging).
    """

    def __init__(self, sparse_r: Tensor) -> None:
        super().__init__()
        if not sparse_r.is_sparse:
            raise TypeError("Restriction expects a sparse_coo_tensor R")
        self.register_buffer("mat", sparse_r.coalesce())

    def forward(self, x_fine: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_fine:
            ``[N_f, F]`` nodal features on the fine graph.

        Returns
        -------
        torch.Tensor
            ``[N_c, F]`` coarse features ``h_coarse ≈ R @ h_fine``.
        """
        if x_fine.dim() != 2:
            raise ValueError(f"x_fine must be [N_f, F], got {tuple(x_fine.shape)}")
        return torch.sparse.mm(self.mat, x_fine)


class Prolongation(nn.Module):
    """
    Apply a sparse prolongation matrix ``P ∈ ℝ^{N_f × N_c}`` (e.g. piecewise constant).
    """

    def __init__(self, sparse_p: Tensor) -> None:
        super().__init__()
        if not sparse_p.is_sparse:
            raise TypeError("Prolongation expects a sparse_coo_tensor P")
        self.register_buffer("mat", sparse_p.coalesce())

    def forward(self, x_coarse: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_coarse:
            ``[N_c, F]``.

        Returns
        -------
        torch.Tensor
            ``[N_f, F]`` upsampled features ``P @ x_coarse``.
        """
        if x_coarse.dim() != 2:
            raise ValueError(f"x_coarse must be [N_c, F], got {tuple(x_coarse.shape)}")
        return torch.sparse.mm(self.mat, x_coarse)
