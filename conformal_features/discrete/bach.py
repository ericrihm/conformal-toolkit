"""Discrete approximation of Bach tensor norm.

This is NOT the true Bach tensor (the 4D Fefferman-Graham obstruction) but a
computationally tractable scalar PROXY that captures 4th-order curvature
variation on a surface, used as one channel of the per-vertex feature vector.

Discretization honesty note (see ERRATA M17). The cotangent matrix L is the
FEM *stiffness* matrix, i.e. an INTEGRATED operator: (L f)_i ~ A_i (Delta f)_i,
not (Delta f)_i. So the quantity computed here, |L L K|, is a *stiffness-
weighted* (area-integrated) bi-Laplacian, NOT the pointwise bi-Laplacian. The
true pointwise operator is

    Delta^2 = (M^{-1} L)(M^{-1} L) = M^{-1} L M^{-1} L,   M = diag(vertex areas).

We deliberately keep the integrated form |L L K| as the feature, because it is
numerically well-behaved across mesh scales; but note two caveats:
  * its *small absolute magnitude on a sphere is partly a scaling artifact*
    (the A_i^2 weighting suppresses it), not proof of correctness -- the
    earlier docstring's claim "K is already area-normalized, so no extra
    weighting is needed" was the wrong reason for the right-ish behavior;
  * the mass-lumped pointwise Delta^2 above is the mathematically correct
    operator but amplifies coarse-mesh curvature noise (e.g. the 12 pentagonal
    defects of an icosphere), so it is offered as `pointwise=True` rather than
    the default. Always validate a discretization on a NON-constant field such
    as f = x^2, where (M^{-1} L)(x^2) = -2 recovers the pointwise Laplacian
    while (L)(x^2) = -2 A_i does not.
"""
from __future__ import annotations
import torch
from conformal_features.discrete.curvature import discrete_gaussian_curvature
from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas


def discrete_bach_norm(
    vertices: torch.Tensor, faces: torch.Tensor, pointwise: bool = False
) -> torch.Tensor:
    """Per-vertex discrete Bach tensor norm proxy.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor
        pointwise: if False (default), return the stiffness-weighted
            (area-integrated) bi-Laplacian |L L K| -- scale-stable, used as the
            feature channel. If True, return the mathematically-correct
            pointwise bi-Laplacian |M^{-1} L M^{-1} L K| (noisier on coarse
            meshes). See ERRATA M17.

    Returns:
        bach_norm: (V,) Tensor
    """
    K = discrete_gaussian_curvature(vertices, faces)
    L = cotangent_laplacian(vertices, faces)

    def stiffness(field: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(L, field.unsqueeze(1)).squeeze(1)

    if pointwise:
        inv_area = 1.0 / vertex_areas(vertices, faces).clamp_min(1e-12)
        lap = lambda f: inv_area * stiffness(f)   # pointwise Delta = M^{-1} L
        return lap(lap(K)).abs()

    # Integrated proxy |L L K| (the default feature channel).
    return stiffness(stiffness(K)).abs()
