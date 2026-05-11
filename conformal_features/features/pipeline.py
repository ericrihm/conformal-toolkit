"""Feature extraction pipeline: mesh -> per-vertex conformal invariants."""
import torch
from conformal_features.discrete.conformal_factor import discrete_conformal_factor
from conformal_features.discrete.willmore import discrete_willmore_density
from conformal_features.discrete.q_curvature import discrete_q_curvature
from conformal_features.discrete.cross_ratios import discrete_cross_ratios
from conformal_features.discrete.bach import discrete_bach_norm
from conformal_features.discrete.curvature import discrete_gaussian_curvature, discrete_mean_curvature


def mesh_conformal_features(vertices, faces, include_isometry=True):
    """Extract per-vertex conformal invariant features.

    Conformal features (7D, always included):
        0: conformal_factor — from discrete Yamabe flow
        1: willmore_density — H^2
        2: q_curvature_2 — R = 2K
        3: q_curvature_4 — discrete Q_4
        4: bach_norm — |L(L(K))|
        5: cross_ratio_mean — mean cross-ratio of incident edges
        6: cross_ratio_var — variance of cross-ratios

    Isometry features (additional, if include_isometry=True):
        7: gaussian_curvature
        8: mean_curvature
        9: H^2 - K (alternative Willmore density)

    Args:
        vertices: (V, 3) float Tensor
        faces: (F, 3) long Tensor
        include_isometry: include isometry-invariant features for ablation

    Returns:
        features: (V, D) Tensor where D=10 (or 7 if include_isometry=False)

    Raises:
        TypeError: if inputs are not torch Tensors
        ValueError: if tensor shapes are wrong
    """
    if not isinstance(vertices, torch.Tensor):
        raise TypeError(f"vertices must be a torch.Tensor, got {type(vertices).__name__}")
    if not isinstance(faces, torch.Tensor):
        raise TypeError(f"faces must be a torch.Tensor, got {type(faces).__name__}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {tuple(vertices.shape)}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {tuple(faces.shape)}")
    if not faces.dtype == torch.long:
        faces = faces.long()

    cf = discrete_conformal_factor(vertices, faces)
    wd = discrete_willmore_density(vertices, faces)
    q2 = discrete_q_curvature(vertices, faces, order=2)
    q4 = discrete_q_curvature(vertices, faces, order=4)
    bn = discrete_bach_norm(vertices, faces)
    cr = discrete_cross_ratios(vertices, faces)

    conformal_feats = torch.stack([
        cf,
        wd,
        q2,
        q4,
        bn,
        cr['per_vertex_mean'],
        cr['per_vertex_var'],
    ], dim=1)

    if not include_isometry:
        return conformal_feats

    K = discrete_gaussian_curvature(vertices, faces)
    H = discrete_mean_curvature(vertices, faces)
    willmore_alt = H ** 2 - K

    iso_feats = torch.stack([K, H, willmore_alt], dim=1)
    return torch.cat([conformal_feats, iso_feats], dim=1)
