"""
End-to-end composable surrogate: spatial message passing ⊗ Heun integrator.

State ``h`` stacks physical channels (default: displacement ``u``, velocity ``v``)
plus optional latent padding; the network predicts ``dh/dt`` which is integrated
with ``HeunIntegrator``.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from modules.integrator import HeunIntegrator
from modules.message_passing import SpatialMessagePassingLayer
from utils.halo_sync import sync_halo_features


class PhysicsGNNSurrogate(nn.Module):
    """
    morphism: ``(h^{t}, edge_index) → h^{t+1}`` using learned ``f_θ`` + Heun.

    Internal pipeline:
        h ↦ encoder ↦ stacked SpatialMessagePassingLayer ↦ decoder ↦ f(h)
        h^{t+1} = Heun(h^{t}, f, Δt)
    """

    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int,
        num_message_layers: int = 2,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.encoder = nn.Linear(self.state_dim, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                SpatialMessagePassingLayer(self.hidden_dim, self.hidden_dim)
                for _ in range(num_message_layers)
            ]
        )
        self.decoder = nn.Linear(self.hidden_dim, self.state_dim)
        self.integrator = HeunIntegrator()

    def derivative_field(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """
        Predict ``dh/dt`` given current state ``h`` (before time integration).

        Parameters
        ----------
        h:
            ``[N, state_dim]``.
        edge_index:
            ``[2, E]`` **0-based**.

        Returns
        -------
        torch.Tensor
            Same shape as ``h``.
        """
        x = F.relu(self.encoder(h))
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.decoder(x)

    def forward(self, h: Tensor, edge_index: Tensor) -> Tensor:
        """One Heun step: ``h^{t} → h^{t+1}``."""

        def f(z: Tensor) -> Tensor:
            return self.derivative_field(z, edge_index)

        return self.integrator(h, f, self.dt)

    def rollout(
        self,
        h0: Tensor,
        edge_index: Tensor,
        steps: int,
    ) -> Tensor:
        """
        Autoregressive rollout (inference-style).

        Returns
        -------
        torch.Tensor
            ``[steps + 1, N, state_dim]`` including the initial state at index 0.
        """
        traj: List[Tensor] = [h0]
        h = h0
        for _ in range(int(steps)):
            h = self.forward(h, edge_index)
            traj.append(h)
        return torch.stack(traj, dim=0)


class PhysicsGNNSurrogateDDM(PhysicsGNNSurrogate):
    """
    Domain-decomposed surrogate: **composition** of single-patch Heun steps with an
    outer ``sync_halo_features`` functor between global time levels.

    Internal GNN / ``HeunIntegrator`` are unchanged from ``PhysicsGNNSurrogate``.
    """

    def forward_subdomains(
        self,
        hs: List[Tensor],
        edge_indices: List[Tensor],
        global_ids_python: List[Tensor],
        is_ghost: List[Tensor],
        *,
        num_global: int,
        sync_before: bool = False,
        sync_after: bool = True,
    ) -> List[Tensor]:
        """
        One physical time step on all patches: optional pre-sync → Heun → optional post-sync.

        Default: **post-sync only** (after Heun), matching the Step 2 spec; set
        ``sync_before=True`` if ghosts must be refreshed before evaluating ``f_θ``.
        """
        xs = hs
        if sync_before:
            xs = sync_halo_features(
                xs, global_ids_python, is_ghost, num_global=num_global
            )
        outs = [super().forward(xs[i], edge_indices[i]) for i in range(len(xs))]
        if sync_after:
            outs = sync_halo_features(
                outs, global_ids_python, is_ghost, num_global=num_global
            )
        return outs

    def rollout_ddm(
        self,
        h0_list: List[Tensor],
        edge_indices: List[Tensor],
        global_ids_python: List[Tensor],
        is_ghost: List[Tensor],
        *,
        num_global: int,
        steps: int,
        sync_before_rollout: bool = True,
        sync_before_each_heun: bool = False,
    ) -> List[Tensor]:
        """
        Autoregressive multi-patch rollout.

        Each global time level: optionally sync inputs → Heun per patch → sync outputs.
        ``steps`` counts Heun integrations after the initial condition at ``t=0``.
        """
        hs = list(h0_list)
        if sync_before_rollout:
            hs = sync_halo_features(
                hs, global_ids_python, is_ghost, num_global=num_global
            )
        traj: List[List[Tensor]] = [[h.clone()] for h in hs]
        for _ in range(int(steps)):
            if sync_before_each_heun:
                hs = sync_halo_features(
                    hs, global_ids_python, is_ghost, num_global=num_global
                )
            outs = [
                super(PhysicsGNNSurrogateDDM, self).forward(hs[i], edge_indices[i])
                for i in range(len(hs))
            ]
            hs = sync_halo_features(
                outs, global_ids_python, is_ghost, num_global=num_global
            )
            for sd, h in enumerate(hs):
                traj[sd].append(h.clone())
        return [torch.stack(t, dim=0) for t in traj]
