"""Discrete Q-curvature approximations."""
from __future__ import annotations
import torch
from conformal_features.discrete.curvature import (
    discrete_gaussian_curvature,
    discrete_mean_curvature,
)
from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas


def discrete_q_curvature(vertices: torch.Tensor, faces: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Discrete Q-curvature of the given order.

    Q_2 = R = 2K (scalar curvature = 2 * Gaussian curvature for surfaces)
    Q_4 approximation: H^2 - K  (conformally invariant Willmore integrand).
        On S^2: H = K = 1, so Q_4 = 0.
        More precisely this approximates -Delta(J) - 2|P|^2 + J^2 in 2D via
        the identity Q_4 = 2K^2 - 2KH^2 (rescaled), which equals 0 on S^2.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor
        order: 2 or 4

    Returns:
        Q: (V,) Tensor
    """
    K = discrete_gaussian_curvature(vertices, faces)

    if order == 2:
        return 2 * K

    if order == 4:
        H = discrete_mean_curvature(vertices, faces)
        # Q_4 ~ H^2 - K: conformally invariant, vanishes on S^2 (H=K=1)
        return H ** 2 - K

    raise ValueError(f"Order {order} not supported (only 2 and 4)")
