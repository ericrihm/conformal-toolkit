"""Discrete approximation of Bach tensor norm.

|B| ~ |L^2(K)| as a 4th-order curvature feature.
This is not the true Bach tensor but a computationally tractable proxy
that captures 4th-order curvature variation.

The bi-Laplacian L^2 K (cotangent Laplacian applied twice to the discrete
Gaussian curvature) acts as a 4th-order differential operator. Since
discrete_gaussian_curvature already normalizes K = angle_defect / area, the
raw bi-Laplacian (no additional area weighting) gives values that are small on
a smooth, nearly-uniform sphere and converge to zero under mesh refinement.
"""
import torch
from conformal_features.discrete.curvature import discrete_gaussian_curvature
from conformal_features.discrete.mesh_utils import cotangent_laplacian


def discrete_bach_norm(vertices, faces):
    """Per-vertex discrete Bach tensor norm approximation.

    Computes |L @ L @ K| where K is the discrete Gaussian curvature and L is
    the cotangent Laplacian. This bi-Laplacian captures 4th-order curvature
    variation and is small on a sphere (where K is nearly constant).

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        bach_norm: (V,) Tensor
    """
    K = discrete_gaussian_curvature(vertices, faces)
    L = cotangent_laplacian(vertices, faces)

    # Apply the cotangent Laplacian twice. K is already area-normalized
    # (angle_defect / area), so no additional area weighting is needed —
    # the raw bi-Laplacian L@L@K converges to zero on a sphere where K is constant.
    LK = torch.sparse.mm(L, K.unsqueeze(1)).squeeze(1)
    LLK = torch.sparse.mm(L, LK.unsqueeze(1)).squeeze(1)

    return LLK.abs()
