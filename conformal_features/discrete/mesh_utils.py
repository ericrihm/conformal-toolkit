"""Core mesh utilities: cotangent Laplacian, vertex areas, edge extraction."""
from __future__ import annotations
import torch


def get_edges(faces: torch.Tensor) -> torch.Tensor:
    """Extract unique undirected edges from face list.

    Args:
        faces: (F, 3) LongTensor

    Returns:
        edges: (E, 2) LongTensor, sorted so edges[:,0] < edges[:,1]
    """
    edge_pairs = torch.cat([
        faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]
    ], dim=0)
    sorted_edges = torch.sort(edge_pairs, dim=1)[0]
    unique_edges = torch.unique(sorted_edges, dim=0)
    return unique_edges


def vertex_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Barycentric vertex areas (face area / 3 distributed to each vertex).

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        areas: (V,) Tensor
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_areas = 0.5 * torch.linalg.norm(torch.linalg.cross(v1 - v0, v2 - v0), dim=-1)

    areas = torch.zeros(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
    for i in range(3):
        areas.scatter_add_(0, faces[:, i], face_areas / 3)
    return areas


def cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Build the cotangent Laplacian as a sparse matrix.

    For edge (i,j) shared by triangles with opposite vertices k and l:
        w_ij = (cot(angle_k) + cot(angle_l)) / 2

    L_ij = -w_ij  (off-diagonal)
    L_ii = sum_j w_ij  (diagonal, positive)

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        L: (V, V) sparse Tensor (positive semi-definite)
    """
    V = vertices.shape[0]
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    def cot_angle(a, b):
        cos_val = (a * b).sum(dim=-1)
        cross = torch.linalg.cross(a, b)
        sin_val = torch.linalg.norm(cross, dim=-1).clamp(min=1e-12)
        return cos_val / sin_val

    # Cotangent of angle at each vertex of each face
    cot0 = cot_angle(v1 - v0, v2 - v0)  # opposite edge 1-2
    cot1 = cot_angle(v0 - v1, v2 - v1)  # opposite edge 0-2
    cot2 = cot_angle(v0 - v2, v1 - v2)  # opposite edge 0-1

    # Each cotangent weight goes to the OPPOSITE edge
    ii = torch.cat([
        faces[:, 1], faces[:, 2],  # edge 1-2 (from cot0)
        faces[:, 0], faces[:, 2],  # edge 0-2 (from cot1)
        faces[:, 0], faces[:, 1],  # edge 0-1 (from cot2)
    ])
    jj = torch.cat([
        faces[:, 2], faces[:, 1],
        faces[:, 2], faces[:, 0],
        faces[:, 1], faces[:, 0],
    ])
    vals = torch.cat([cot0, cot0, cot1, cot1, cot2, cot2]) / 2

    # Off-diagonal (negative)
    L = torch.sparse_coo_tensor(
        torch.stack([ii, jj]), -vals, (V, V)
    ).coalesce()

    # Diagonal (positive: sum of weights)
    diag_vals = torch.zeros(V, dtype=vertices.dtype, device=vertices.device)
    diag_vals.scatter_add_(0, ii, vals)
    diag_idx = torch.arange(V, device=vertices.device)
    L = L + torch.sparse_coo_tensor(
        torch.stack([diag_idx, diag_idx]), diag_vals, (V, V)
    )

    return L.coalesce()


def face_angles(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute interior angles at each vertex of each face.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        angles: (F, 3) Tensor — angle at vertex 0, 1, 2 of each face
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    def angle_at(a, b, c):
        ab = b - a
        ac = c - a
        cos_val = (ab * ac).sum(-1) / (ab.norm(dim=-1) * ac.norm(dim=-1)).clamp(min=1e-12)
        return torch.acos(cos_val.clamp(-1 + 1e-7, 1 - 1e-7))

    a0 = angle_at(v0, v1, v2)
    a1 = angle_at(v1, v0, v2)
    a2 = angle_at(v2, v0, v1)
    return torch.stack([a0, a1, a2], dim=1)
