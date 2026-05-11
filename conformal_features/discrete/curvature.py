"""Discrete Gaussian and mean curvature on triangle meshes."""
import torch
import math
from conformal_features.discrete.mesh_utils import vertex_areas, cotangent_laplacian, face_angles


def discrete_gaussian_curvature(vertices, faces):
    """Discrete Gaussian curvature via angle defect.

    K_i = (2*pi - sum of angles at vertex i) / A_i

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        K: (V,) Tensor
    """
    V = vertices.shape[0]
    angles = face_angles(vertices, faces)

    angle_sum = torch.zeros(V, dtype=vertices.dtype, device=vertices.device)
    for i in range(3):
        angle_sum.scatter_add_(0, faces[:, i], angles[:, i])

    areas = vertex_areas(vertices, faces)
    areas = areas.clamp(min=1e-12)

    K = (2 * math.pi - angle_sum) / areas
    return K


def discrete_mean_curvature(vertices, faces):
    """Discrete mean curvature from cotangent Laplacian.

    H_i = |L @ v_i| / (2 * A_i)

    where L is the cotangent Laplacian and A_i is the vertex area.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        H: (V,) Tensor (non-negative mean curvature magnitude)
    """
    L = cotangent_laplacian(vertices, faces)
    Lv = torch.sparse.mm(L, vertices)
    H_vec_norm = Lv.norm(dim=-1)
    areas = vertex_areas(vertices, faces).clamp(min=1e-12)
    H = H_vec_norm / (2 * areas)
    return H
