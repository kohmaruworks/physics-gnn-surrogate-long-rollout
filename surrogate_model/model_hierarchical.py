"""
Hierarchical (two-level) surrogate: fine local tensor MP ⊗ restriction ⊗ coarse global ⊗ prolongation.

Composition pipeline::

    embed → fine MPs → R → coarse MPs → P (additive correction) → decode

Independent from ``HeunIntegrator`` / DDM — compose externally when needed.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from modules.multigrid import Prolongation, Restriction
from modules.tensor_mp import TensorMessagePassing


class HierarchicalPhysicsGNN(nn.Module):
    """
    Implements::

        h ← σ(F_fine(h))
        h ← h + P( σ(F_coarse(R(h))) )

    where ``F_*`` are stacks of ``TensorMessagePassing``.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        bond_dim: int,
        sparse_r: Tensor,
        sparse_p: Tensor,
        num_fine_layers: int = 2,
        num_coarse_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        self.embed = nn.Linear(self.state_dim, self.hidden_dim)
        self.fine_layers = nn.ModuleList(
            [
                TensorMessagePassing(self.hidden_dim, self.hidden_dim, bond_dim)
                for _ in range(num_fine_layers)
            ]
        )
        self.coarse_layers = nn.ModuleList(
            [
                TensorMessagePassing(self.hidden_dim, self.hidden_dim, bond_dim)
                for _ in range(num_coarse_layers)
            ]
        )
        self.restrict = Restriction(sparse_r)
        self.prolong = Prolongation(sparse_p)
        self.decode = nn.Linear(self.hidden_dim, self.state_dim)

    def forward(self, x_fine: Tensor, edge_index_fine: Tensor, edge_index_coarse: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x_fine:
            ``[N_f, state_dim]`` nodal state (e.g. displacement + velocity).
        edge_index_fine:
            ``[2, E_f]`` fine graph (0-based).
        edge_index_coarse:
            ``[2, E_c]`` coarse graph (0-based).

        Returns
        -------
        torch.Tensor
            ``[N_f, state_dim]`` updated fine prediction.
        """
        h = F.relu(self.embed(x_fine))
        for layer in self.fine_layers:
            h = F.relu(layer(h, edge_index_fine))

        h_coarse = self.restrict(h)
        for layer in self.coarse_layers:
            h_coarse = F.relu(layer(h_coarse, edge_index_coarse))

        h = h + self.prolong(h_coarse)
        return self.decode(h)
