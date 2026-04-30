"""
Tensor-product style message passing inspired by MPS / bond contraction.

Implements a contracted edge message::

    m_{j→i} = Σ_{αβ} A_i^α W_{αβ}(e_{ij}) B_j^β

with ``torch.einsum``, exchangeable with ``SpatialMessagePassingLayer`` in deeper stacks.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class TensorMessagePassing(MessagePassing):
    """
    Single-head bond contraction message passing (aggregation: sum).

    Default edge embedding uses ``concat(x_i, x_j)`` when ``edge_attr`` is omitted.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bond_dim: int,
        *,
        edge_dim: Optional[int] = None,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr, flow="source_to_target")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bond_dim = int(bond_dim)
        self.edge_dim = int(edge_dim) if edge_dim is not None else 2 * self.in_channels

        self.lin_a = nn.Linear(self.in_channels, self.bond_dim, bias=True)
        self.lin_b = nn.Linear(self.in_channels, self.bond_dim, bias=True)
        self.lin_edge = nn.Linear(self.edge_dim, self.bond_dim * self.bond_dim, bias=True)
        self.lin_out = nn.Linear(self.in_channels + 1, self.out_channels, bias=True)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x:
            ``[N, F_in]``.
        edge_index:
            ``[2, E]`` PyG indices (0-based).
        edge_attr:
            Reserved for future use. Self-loops added internally require consistent
            ``edge_attr`` padding; currently embeddings always use ``[x_i ‖ x_j]``.
        """
        if edge_attr is not None:
            raise ValueError(
                "TensorMessagePassing currently builds edge features from node pairs only; "
                "omit edge_attr."
            )
        if x.dim() != 2 or x.size(-1) != self.in_channels:
            raise ValueError(f"x must be [N,{self.in_channels}], got {tuple(x.shape)}")
        edge_index2, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        agg = self.propagate(edge_index2, x=x)
        return self.lin_out(torch.cat([x, agg], dim=-1))

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        edge_feat = torch.cat([x_i, x_j], dim=-1)
        a_i = self.lin_a(x_i)
        b_j = self.lin_b(x_j)
        w = self.lin_edge(edge_feat).view(-1, self.bond_dim, self.bond_dim)
        m = torch.einsum("ea,eab,eb->e", a_i, w, b_j)
        return m.unsqueeze(-1)
