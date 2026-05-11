"""Discrete Mobius-invariant cross-ratios.

For an interior edge (i,j) shared by triangles (i,j,k) and (i,j,l):
    cr_{ij} = |v_i - v_k| * |v_j - v_l| / (|v_i - v_l| * |v_j - v_k|)

Cross-ratios characterize the discrete conformal structure
(Bobenko-Pinkall-Springborn, 2015).
"""
from __future__ import annotations
import torch


def discrete_cross_ratios(vertices: torch.Tensor, faces: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute Mobius-invariant cross-ratios and aggregate per vertex.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor

    Returns:
        dict with:
            'per_vertex_mean': (V,) Tensor — mean cross-ratio of incident edges
            'per_vertex_var': (V,) Tensor — variance of cross-ratios
            'per_edge': (E_interior,) Tensor — raw cross-ratio per interior edge
    """
    V = vertices.shape[0]
    F = faces.shape[0]
    device = vertices.device

    # --- Step 1: Build half-edge-to-opposite mapping (vectorized) ---
    # For each face (a,b,c), create 3 half-edges with their opposite vertex:
    #   edge (b,c) -> opp a,  edge (c,a) -> opp b,  edge (a,b) -> opp c
    e0 = torch.stack([faces[:, 1], faces[:, 2]], dim=1)  # (F, 2)
    e1 = torch.stack([faces[:, 2], faces[:, 0]], dim=1)  # (F, 2)
    e2 = torch.stack([faces[:, 0], faces[:, 1]], dim=1)  # (F, 2)
    all_edges = torch.cat([e0, e1, e2], dim=0)           # (3F, 2)
    all_opp = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]])  # (3F,)

    # Sort each edge so smaller vertex comes first
    sorted_edges, _ = torch.sort(all_edges, dim=1)       # (3F, 2)

    # Hash edges for grouping: unique key = v_lo * V + v_hi
    edge_hash = sorted_edges[:, 0] * V + sorted_edges[:, 1]  # (3F,)

    # Find unique edges and inverse mapping
    unique_hash, inverse = torch.unique(edge_hash, return_inverse=True)
    num_unique = unique_hash.shape[0]

    # Count how many faces share each edge
    ones = torch.ones(3 * F, dtype=torch.long, device=device)
    counts = torch.zeros(num_unique, dtype=torch.long, device=device)
    counts.scatter_add_(0, inverse, ones)

    # Interior edges have exactly 2 incident faces (count == 2)
    interior_mask = counts == 2  # (num_unique,)

    # --- Step 2: For interior edges, find the two opposite vertices ---
    # Sort by edge hash to group, then take consecutive pairs.
    sort_order = torch.argsort(inverse)
    sorted_inverse = inverse[sort_order]
    sorted_opp = all_opp[sort_order]
    sorted_all_edges = sorted_edges[sort_order]  # (3F, 2) sorted by group

    # Build group start positions via prefix-sum of counts
    group_starts = torch.zeros(num_unique, dtype=torch.long, device=device)
    group_starts[1:] = counts[:-1].cumsum(0)

    # Extract interior edge indices
    interior_idx = torch.where(interior_mask)[0]  # indices into unique_hash
    num_interior = interior_idx.shape[0]

    # For each interior edge, get the two positions in the sorted array
    pos0 = group_starts[interior_idx]      # first occurrence
    pos1 = pos0 + 1                        # second occurrence

    # Edge endpoint vertices (from sorted_edges, both entries have same i,j)
    edge_i = sorted_all_edges[pos0, 0]  # (num_interior,)
    edge_j = sorted_all_edges[pos0, 1]  # (num_interior,)

    # Opposite vertices
    opp_k = sorted_opp[pos0]  # (num_interior,)
    opp_l = sorted_opp[pos1]  # (num_interior,)

    # --- Step 3: Compute cross-ratios in batch ---
    vi = vertices[edge_i]  # (num_interior, 3)
    vj = vertices[edge_j]  # (num_interior, 3)
    vk = vertices[opp_k]   # (num_interior, 3)
    vl = vertices[opp_l]   # (num_interior, 3)

    num = (vi - vk).norm(dim=1) * (vj - vl).norm(dim=1)
    den = ((vi - vl).norm(dim=1) * (vj - vk).norm(dim=1)).clamp(min=1e-12)
    cr_tensor = num / den   # (num_interior,)

    # --- Step 4: Per-vertex aggregation via scatter_add_ ---
    # Each interior edge contributes its cross-ratio to both endpoints
    all_vertex_idx = torch.cat([edge_i, edge_j])      # (2 * num_interior,)
    cr_doubled = cr_tensor.repeat(2)                    # (2 * num_interior,)

    per_vertex_sum = torch.zeros(V, dtype=vertices.dtype, device=device)
    per_vertex_sum.scatter_add_(0, all_vertex_idx, cr_doubled)

    per_vertex_sq_sum = torch.zeros(V, dtype=vertices.dtype, device=device)
    per_vertex_sq_sum.scatter_add_(0, all_vertex_idx, cr_doubled ** 2)

    per_vertex_count = torch.zeros(V, dtype=vertices.dtype, device=device)
    per_vertex_count.scatter_add_(
        0, all_vertex_idx, torch.ones_like(cr_doubled)
    )

    count = per_vertex_count.clamp(min=1)
    per_vertex_mean = per_vertex_sum / count
    per_vertex_var = per_vertex_sq_sum / count - per_vertex_mean ** 2
    per_vertex_var = per_vertex_var.clamp(min=0)

    return {
        'per_vertex_mean': per_vertex_mean,
        'per_vertex_var': per_vertex_var,
        'per_edge': cr_tensor,
    }
