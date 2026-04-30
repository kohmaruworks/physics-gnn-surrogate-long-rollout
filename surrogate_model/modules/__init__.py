"""Composable spatial message-passing blocks for graph surrogates."""

from .message_passing import SpatialMessagePassingLayer
from .integrator import HeunIntegrator
from .physics_loss import SymplecticLoss

__all__ = [
    "SpatialMessagePassingLayer",
    "HeunIntegrator",
    "SymplecticLoss",
]
