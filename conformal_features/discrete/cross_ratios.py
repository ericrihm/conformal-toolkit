"""Discrete Mobius-invariant cross-ratios.

For an interior edge (i,j) shared by triangles (i,j,k) and (i,j,l):
    cr_{ij} = |v_i - v_k| * |v_j - v_l| / (|v_i - v_l| * |v_j - v_k|)

Cross-ratios characterize the discrete conformal structure
(Bobenko-Pinkall-Springborn, 2015).
"""
import torch


def discrete_cross_ratios(vertices, faces):
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

    # Build edge-to-opposite-vertex mapping
    # For each half-edge (i,j) in face (i,j,k), the opposite vertex is k
    half_edge_to_opp = {}
    for f_idx in range(F):
        for local in range(3):
            i = faces[f_idx, local].item()
            j = faces[f_idx, (local + 1) % 3].item()
            k = faces[f_idx, (local + 2) % 3].item()
            key = (min(i, j), max(i, j))
            if key not in half_edge_to_opp:
                half_edge_to_opp[key] = []
            half_edge_to_opp[key].append(k)

    # Interior edges have exactly 2 opposite vertices
    edge_list = []
    cr_list = []
    for (i, j), opps in half_edge_to_opp.items():
        if len(opps) != 2:
            continue
        k, l = opps
        vi, vj, vk, vl = vertices[i], vertices[j], vertices[k], vertices[l]
        num = (vi - vk).norm() * (vj - vl).norm()
        den = ((vi - vl).norm() * (vj - vk).norm()).clamp(min=1e-12)
        cr = num / den
        edge_list.append((i, j))
        cr_list.append(cr.item())

    cr_tensor = torch.tensor(cr_list, dtype=vertices.dtype)

    # Aggregate per vertex
    per_vertex_sum = torch.zeros(V, dtype=vertices.dtype)
    per_vertex_sq_sum = torch.zeros(V, dtype=vertices.dtype)
    per_vertex_count = torch.zeros(V, dtype=vertices.dtype)

    for idx, (i, j) in enumerate(edge_list):
        cr_val = cr_tensor[idx]
        per_vertex_sum[i] += cr_val
        per_vertex_sum[j] += cr_val
        per_vertex_sq_sum[i] += cr_val ** 2
        per_vertex_sq_sum[j] += cr_val ** 2
        per_vertex_count[i] += 1
        per_vertex_count[j] += 1

    count = per_vertex_count.clamp(min=1)
    per_vertex_mean = per_vertex_sum / count
    per_vertex_var = per_vertex_sq_sum / count - per_vertex_mean ** 2
    per_vertex_var = per_vertex_var.clamp(min=0)

    return {
        'per_vertex_mean': per_vertex_mean,
        'per_vertex_var': per_vertex_var,
        'per_edge': cr_tensor,
    }
