"""
Halo exchange: pullback / pushforward of nodal features across subdomain patches.

Keeps ``message_passing`` / ``integrator`` ignorant of multi-patch layout; compose
this functor **between** global time substeps.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor


def sync_halo_features(
    subdomain_features: Sequence[Tensor],
    global_ids_python: Sequence[Tensor],
    is_ghost: Sequence[Tensor],
    *,
    num_global: int,
) -> List[Tensor]:
    """
    Overwrite ghost rows using core-owned values keyed by global vertex id.

    For each global id ``g``, exactly one subdomain contributes a **non-ghost** row
    (the Metis core owner). Those rows define a buffer ``buf[g]``. Every ghost row
    with the same ``g`` is replaced by ``buf[g]``.

    Parameters
    ----------
    subdomain_features:
        Length ``K`` list of ``[N'_i, F]`` tensors (one extended patch per subdomain).
    global_ids_python:
        Length ``K`` list of ``[N'_i]`` int64 global vertex ids (**0-based**).
    is_ghost:
        Length ``K`` list of ``[N'_i]`` bool; ``True`` for halo / ghost nodes.
    num_global:
        Global vertex count ``N`` (number of mesh nodes).

    Returns
    -------
    list[torch.Tensor]
        Patches with ghost features synchronized (cores unchanged).
    """
    if not (
        len(subdomain_features)
        == len(global_ids_python)
        == len(is_ghost)
    ):
        raise ValueError("subdomain_features, global_ids_python, is_ghost length mismatch")
    if len(subdomain_features) == 0:
        return []

    device = subdomain_features[0].device
    dtype = subdomain_features[0].dtype
    dim_f = int(subdomain_features[0].size(-1))

    buf = torch.zeros(num_global, dim_f, device=device, dtype=dtype)

    for h, gid, ghost in zip(subdomain_features, global_ids_python, is_ghost):
        if h.dim() != 2 or h.size(-1) != dim_f:
            raise ValueError("all subdomain features must share shape [N', F]")
        if gid.shape != ghost.shape or gid.dim() != 1:
            raise ValueError("global_ids and is_ghost must be 1D and matching length")
        core = ~ghost.to(device=device)
        if core.any():
            buf[gid[core].to(torch.long)] = h[core]

    out: List[Tensor] = []
    for h, gid, ghost in zip(subdomain_features, global_ids_python, is_ghost):
        h2 = h.clone()
        gmask = ghost.to(device=device)
        if gmask.any():
            h2[gmask] = buf[gid[gmask].to(torch.long)]
        out.append(h2)
    return out
