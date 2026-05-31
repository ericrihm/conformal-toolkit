"""Discrete conformal invariants for geometric deep learning."""

__version__ = "0.1.1"

from conformal_features.features.pipeline import mesh_conformal_features
from conformal_features.discrete.spectral import (
    lbo_eigenvectors,
    heat_kernel_signature,
    wave_kernel_signature,
)

__all__ = [
    "mesh_conformal_features",
    "lbo_eigenvectors",
    "heat_kernel_signature",
    "wave_kernel_signature",
]
