"""Discrete conformal factor via discrete Yamabe flow.

Finds u_i such that the conformally equivalent metric has constant curvature.
Uses iterative curvature flow: u_i += lr * (K_target - K_i(u)).
"""
import torch
import math
from conformal_features.discrete.mesh_utils import get_edges, vertex_areas, face_angles


def discrete_conformal_factor(vertices, faces, max_iter=500, lr=0.001, tol=1e-4):
    """Compute per-vertex log conformal factor via discrete Yamabe flow.

    Finds u such that the conformally equivalent discrete metric
    (with edge lengths l'_ij = e^{(u_i+u_j)/2} l_ij) has constant
    Gaussian curvature.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor
        max_iter: maximum iterations
        lr: learning rate for curvature flow
        tol: convergence tolerance on max curvature deviation

    Returns:
        u: (V,) Tensor — log conformal factor per vertex
    """
    V = vertices.shape[0]
    F_count = faces.shape[0]

    # Compute initial edge lengths
    edges = get_edges(faces)
    E = edges.shape[0]
    edge_lengths = (vertices[edges[:, 0]] - vertices[edges[:, 1]]).norm(dim=-1)

    # Euler characteristic and target curvature
    chi = V - E + F_count
    K_target = 2 * math.pi * chi / vertex_areas(vertices, faces).sum().item()

    u = torch.zeros(V, dtype=vertices.dtype)

    for iteration in range(max_iter):
        # Compute angle defect with current u
        K = _curvature_with_conformal_factor(vertices, faces, edges, edge_lengths, u)
        residual = K_target - K
        if residual.abs().max() < tol:
            break
        u = u + lr * residual

    # Normalize: subtract mean so that sum(u) = 0
    u = u - u.mean()
    return u


def _curvature_with_conformal_factor(vertices, faces, edges, base_lengths, u):
    """Compute discrete Gaussian curvature with conformally scaled edge lengths.

    Scaled length: l'_ij = exp((u_i + u_j)/2) * l_ij
    Then compute angle defect with the new lengths.
    """
    V = vertices.shape[0]

    # Build edge length lookup
    edge_key_to_length = {}
    for idx in range(edges.shape[0]):
        i, j = edges[idx, 0].item(), edges[idx, 1].item()
        scale = torch.exp((u[i] + u[j]) / 2)
        new_len = base_lengths[idx] * scale
        edge_key_to_length[(min(i, j), max(i, j))] = new_len

    # Compute angles from edge lengths using law of cosines
    angle_sum = torch.zeros(V, dtype=vertices.dtype)
    area_sum = torch.zeros(V, dtype=vertices.dtype)

    for f_idx in range(faces.shape[0]):
        i, j, k = faces[f_idx, 0].item(), faces[f_idx, 1].item(), faces[f_idx, 2].item()
        a = edge_key_to_length.get((min(j, k), max(j, k)), torch.tensor(1.0))  # opposite i
        b = edge_key_to_length.get((min(i, k), max(i, k)), torch.tensor(1.0))  # opposite j
        c = edge_key_to_length.get((min(i, j), max(i, j)), torch.tensor(1.0))  # opposite k

        # Angle at i: cos(A) = (b^2 + c^2 - a^2) / (2bc)
        cos_i = (b**2 + c**2 - a**2) / (2 * b * c).clamp(min=1e-12)
        cos_j = (a**2 + c**2 - b**2) / (2 * a * c).clamp(min=1e-12)
        cos_k = (a**2 + b**2 - c**2) / (2 * a * b).clamp(min=1e-12)

        angle_i = torch.acos(cos_i.clamp(-1 + 1e-7, 1 - 1e-7))
        angle_j = torch.acos(cos_j.clamp(-1 + 1e-7, 1 - 1e-7))
        angle_k = torch.acos(cos_k.clamp(-1 + 1e-7, 1 - 1e-7))

        # Heron's formula for area
        s = (a + b + c) / 2
        face_area = torch.sqrt((s * (s-a) * (s-b) * (s-c)).clamp(min=1e-20))

        angle_sum[i] += angle_i
        angle_sum[j] += angle_j
        angle_sum[k] += angle_k
        area_sum[i] += face_area / 3
        area_sum[j] += face_area / 3
        area_sum[k] += face_area / 3

    K = (2 * math.pi - angle_sum) / area_sum.clamp(min=1e-12)
    return K
