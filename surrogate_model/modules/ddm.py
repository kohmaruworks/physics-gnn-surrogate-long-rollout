"""
DDM batch functor: applies an underlying ``PhysicsGNNSurrogate`` per subdomain
without embedding halo logic inside individual GNN layers.
"""

from __future__ import annotations

from typing import Callable, List, Protocol, Sequence

import torch
from torch import Tensor

from utils.halo_sync import sync_halo_features


class SteppableSurrogate(Protocol):
    """Minimal protocol for ``forward(h, edge_index) -> h_next``."""

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor: ...


def forward_subdomain_batch(
    model: SteppableSurrogate,
    hs: Sequence[Tensor],
    edge_indices: Sequence[Tensor],
    *,
    sync_halos_first: bool,
    global_ids_python: Sequence[Tensor] | None = None,
    is_ghost: Sequence[Tensor] | None = None,
    num_global: int | None = None,
    sync_halos_after: bool = False,
) -> List[Tensor]:
    """
    Apply ``model.forward`` on each patch; optionally halo-sync before/after.

    Composition order when both flags are ``True``: sync → forward → sync.
    """
    xs = list(hs)
    if sync_halos_first:
        if global_ids_python is None or is_ghost is None or num_global is None:
            raise ValueError("halo sync requires global_ids_python, is_ghost, num_global")
        xs = sync_halo_features(xs, global_ids_python, is_ghost, num_global=num_global)
    ys = [model.forward(xs[i], edge_indices[i]) for i in range(len(xs))]
    if sync_halos_after:
        if global_ids_python is None or is_ghost is None or num_global is None:
            raise ValueError("halo sync requires global_ids_python, is_ghost, num_global")
        ys = sync_halo_features(ys, global_ids_python, is_ghost, num_global=num_global)
    return ys


def rollout_subdomains_with_closure(
    h0_list: Sequence[Tensor],
    edge_indices: Sequence[Tensor],
    global_ids_python: Sequence[Tensor],
    is_ghost: Sequence[Tensor],
    *,
    num_global: int,
    steps: int,
    step_fn: Callable[[List[Tensor]], List[Tensor]],
) -> List[Tensor]:
    """
    Generic rollout: ``step_fn`` maps synchronized patch list → next synchronized patches.

    Initial state is halo-synchronized before recording / stepping.
    """
    hs = sync_halo_features(
        list(h0_list), global_ids_python, is_ghost, num_global=num_global
    )
    traj = [[hs[i].clone()] for i in range(len(hs))]
    for _ in range(int(steps)):
        hs = step_fn(hs)
        for sd, h in enumerate(hs):
            traj[sd].append(h.clone())
    return [torch.stack(t, dim=0) for t in traj]
