"""Discrete Willmore energy density."""
from __future__ import annotations
import torch
from conformal_features.discrete.curvature import discrete_mean_curvature


def discrete_willmore_density(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Per-vertex Willmore energy density: W_i = H_i^2.

    Total Willmore energy: sum(W_i * A_i).
    For a round sphere of radius R: W = 4*pi (independent of R).

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        W: (V,) Tensor
    """
    H = discrete_mean_curvature(vertices, faces)
    return H ** 2
