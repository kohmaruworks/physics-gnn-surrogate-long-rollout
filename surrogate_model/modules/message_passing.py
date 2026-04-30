"""
Spatial GNN layer using ``torch_geometric.nn.MessagePassing``.

Morphism signature (compositionality):
    ``(x: ℝ^{N×F_in}, edge_index: ℤ^{2×E}) → ℝ^{N×F_out}``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SpatialMessagePassingLayer(MessagePassing):
    """
    Single relational convolution-style layer: concatenate sender/receiver
    features, linear map, aggregate with normalization.

    This is a intentionally small, exchangeable spatial primitive that can be
    stacked or swapped in larger multi-physics surrogates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        aggr: str = "add",
        bias: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__(aggr=aggr, flow="source_to_target")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.normalize = normalize
        self.lin_msg = nn.Linear(2 * self.in_channels, self.out_channels, bias=bias)
        self.lin_root = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.act: Callable[[Tensor], Tensor] = F.relu

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Node features ``[N, F_in]``.
        edge_index:
            PyG COO ``[2, E]`` with **0-based** indices.

        Returns
        -------
        torch.Tensor
            Updated node embeddings ``[N, F_out]``.
        """
        if x.dim() != 2 or x.size(-1) != self.in_channels:
            raise ValueError(
                f"x must be [N, {self.in_channels}], got {tuple(x.shape)}"
            )
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, size=None)
        out = out + self.lin_root(x)
        return self.act(out)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        msg = torch.cat([x_i, x_j], dim=-1)
        return self.lin_msg(msg)

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        out = super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)
        if self.normalize and out.numel() > 0:
            deg = degree(index, dim_size, dtype=out.dtype).clamp(min=1.0)
            out = out / deg.view(-1, 1)
        return out
