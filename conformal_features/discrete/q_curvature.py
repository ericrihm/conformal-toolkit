"""Discrete Q-curvature approximations."""
from __future__ import annotations
import torch
from conformal_features.discrete.curvature import (
    discrete_gaussian_curvature,
    discrete_mean_curvature,
)
from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas


def discrete_q_curvature(vertices: torch.Tensor, faces: torch.Tensor, order: int = 2) -> torch.Tensor:
    """Discrete curvature surface features (orders 2 and 4).

    order=2:  Q_2 = R = 2K  (scalar curvature = 2 * Gaussian curvature
              for a surface).

    order=4:  returns the *Willmore integrand* H^2 - K, NOT the 4D GJMS /
              Branson Q_4. Two honesty caveats (see ERRATA M14/M15/M16):
        * Naming. The intrinsic 4th-order GJMS Q_4 is a 4-manifold object
          with Q_4 = (n-1)! = 6 on the round S^4. A 2-surface quantity such
          as H^2 - K cannot reproduce that value; it is a Willmore-type
          surface feature, not the GJMS Q_4.
        * Invariance. H^2 - K is conformally invariant only *under the
          integral*: ∫(H^2 - K) dA is controlled by Gauss-Bonnet
          (∫K dA = 2*pi*chi) plus Moebius-invariance of ∫H^2 dA. It is NOT
          pointwise conformally invariant.
        * Convergence. On a round S^2 of ANY radius R, H^2 = 1/R^2 = K, so
          H^2 - K = 0; thus this feature -> 0 under mesh refinement on a
          sphere (it does NOT converge to 6 -- an earlier README claim).
          (The earlier "identity" Q_4 = 2K^2 - 2KH^2 was false: that equals
          -2K (H^2 - K), not a rescaling of H^2 - K.)

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
        # Willmore integrand H^2 - K; -> 0 on a round sphere of any radius.
        return H ** 2 - K

    raise ValueError(f"Order {order} not supported (only 2 and 4)")
