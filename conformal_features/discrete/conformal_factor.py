"""Discrete conformal factor via discrete Yamabe flow.

Finds u_i such that the conformally equivalent metric has constant curvature.
Uses iterative curvature flow: u_i += lr * (K_target - K_i(u)).
"""
from __future__ import annotations
import torch
import math
from conformal_features.discrete.mesh_utils import get_edges, vertex_areas, face_angles


def discrete_conformal_factor(vertices: torch.Tensor, faces: torch.Tensor, max_iter: int = 500, lr: float = 0.001, tol: float = 1e-4) -> torch.Tensor:
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


def _curvature_with_conformal_factor(vertices: torch.Tensor, faces: torch.Tensor, edges: torch.Tensor, base_lengths: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Compute discrete Gaussian curvature with conformally scaled edge lengths.

    Scaled length: l'_ij = exp((u_i + u_j)/2) * l_ij
    Then compute angle defect with the new lengths.

    Fully vectorized — no Python loops over edges or faces.
    """
    V = vertices.shape[0]

    # --- Step 1: Vectorized edge scaling ---
    scale = torch.exp((u[edges[:, 0]] + u[edges[:, 1]]) / 2)
    scaled_lengths = base_lengths * scale

    # --- Step 2: Build edge hash for O(1) lookup via searchsorted ---
    # edges are sorted so edges[:,0] < edges[:,1], and torch.unique returns
    # them in sorted order, so edge_hash is already sorted.
    V_total = vertices.shape[0]
    edge_hash = edges[:, 0].long() * V_total + edges[:, 1].long()

    # --- Step 3: For each face, extract its 3 edge indices ---
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Edge opposite vertex 0: (v1, v2), opposite vertex 1: (v0, v2), opposite vertex 2: (v0, v1)
    # Sort each edge pair so smaller index comes first
    e_a0 = torch.minimum(v1, v2)
    e_a1 = torch.maximum(v1, v2)
    e_b0 = torch.minimum(v0, v2)
    e_b1 = torch.maximum(v0, v2)
    e_c0 = torch.minimum(v0, v1)
    e_c1 = torch.maximum(v0, v1)

    hash_a = e_a0.long() * V_total + e_a1.long()  # edge opposite v0
    hash_b = e_b0.long() * V_total + e_b1.long()  # edge opposite v1
    hash_c = e_c0.long() * V_total + e_c1.long()  # edge opposite v2

    idx_a = torch.searchsorted(edge_hash, hash_a)
    idx_b = torch.searchsorted(edge_hash, hash_b)
    idx_c = torch.searchsorted(edge_hash, hash_c)

    a = scaled_lengths[idx_a]  # edge opposite v0
    b = scaled_lengths[idx_b]  # edge opposite v1
    c = scaled_lengths[idx_c]  # edge opposite v2

    # --- Step 4: Vectorized law of cosines for all faces ---
    a2, b2, c2 = a ** 2, b ** 2, c ** 2

    cos_0 = (b2 + c2 - a2) / (2 * b * c).clamp(min=1e-12)
    cos_1 = (a2 + c2 - b2) / (2 * a * c).clamp(min=1e-12)
    cos_2 = (a2 + b2 - c2) / (2 * a * b).clamp(min=1e-12)

    angle_0 = torch.acos(cos_0.clamp(-1 + 1e-7, 1 - 1e-7))
    angle_1 = torch.acos(cos_1.clamp(-1 + 1e-7, 1 - 1e-7))
    angle_2 = torch.acos(cos_2.clamp(-1 + 1e-7, 1 - 1e-7))

    # --- Step 5: Heron's formula for face areas (vectorized) ---
    s = (a + b + c) / 2
    face_area = torch.sqrt((s * (s - a) * (s - b) * (s - c)).clamp(min=1e-20))

    # --- Step 6: Accumulate via scatter_add_ ---
    angle_sum = torch.zeros(V, dtype=vertices.dtype)
    angle_sum.scatter_add_(0, v0, angle_0)
    angle_sum.scatter_add_(0, v1, angle_1)
    angle_sum.scatter_add_(0, v2, angle_2)

    area_third = face_area / 3
    area_sum = torch.zeros(V, dtype=vertices.dtype)
    area_sum.scatter_add_(0, v0, area_third)
    area_sum.scatter_add_(0, v1, area_third)
    area_sum.scatter_add_(0, v2, area_third)

    K = (2 * math.pi - angle_sum) / area_sum.clamp(min=1e-12)
    return K
