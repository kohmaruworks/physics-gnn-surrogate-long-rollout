"""
Time integration schemes decoupled from spatial backbones (Heun / RK2-type).

Signature:
    ``HeunIntegrator: (h, f_θ, Δt) ↦ h'`` where ``f_θ(h) ≈ dh/dt``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn


TensorFn = Callable[[Tensor], Tensor]


class HeunIntegrator(nn.Module):
    """
    Explicit Heun (2nd-order Runge–Kutta) step::

        h̃ = h + Δt f(h)
        h' = h + (Δt/2)( f(h) + f(h̃) )

    ``forward`` accepts an arbitrary callable ``f`` (e.g. a neural velocity field).
    """

    def forward(self, h: Tensor, f: TensorFn, dt: float | Tensor) -> Tensor:
        """
        Parameters
        ----------
        h:
            State ``[N, D]`` (or batched ``[B, N, D]`` — ``f`` must match).
        f:
            Callable predicting ``dh/dt`` with the same shape as ``h``.
        dt:
            Scalar timestep (float or 0-dim tensor).

        Returns
        -------
        torch.Tensor
            ``h^{t+1}`` after one Heun step, same shape as ``h``.
        """
        if isinstance(dt, float):
            dt_t = h.new_tensor(dt)
        else:
            dt_t = dt
        k1 = f(h)
        h_tilde = h + dt_t * k1
        k2 = f(h_tilde)
        return h + 0.5 * dt_t * (k1 + k2)
