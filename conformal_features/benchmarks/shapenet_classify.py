"""ShapeNet classification benchmark comparing conformal vs isometry features.

Usage:
    python -m conformal_features.benchmarks.shapenet_classify \
        --data_dir /path/to/shapenet \
        --features conformal \
        --epochs 50

Feature sets:
    A: xyz + normals (6D)
    B: HKS (16D) — isometry baseline
    C: conformal only (7D)
    D: conformal + isometry (10D)
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path


def extract_features(vertices, faces, feature_set):
    """Extract features based on the chosen set."""
    if feature_set == 'xyz':
        normals = _estimate_normals(vertices, faces)
        return torch.cat([vertices, normals], dim=1)

    if feature_set == 'conformal':
        from conformal_features.features.pipeline import mesh_conformal_features
        return mesh_conformal_features(vertices, faces, include_isometry=False)

    if feature_set == 'conformal+iso':
        from conformal_features.features.pipeline import mesh_conformal_features
        return mesh_conformal_features(vertices, faces, include_isometry=True)

    if feature_set == 'hks':
        return _compute_hks(vertices, faces)

    raise ValueError(f"Unknown feature set: {feature_set}")


def _estimate_normals(vertices, faces):
    """Estimate per-vertex normals via face area weighting."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = torch.linalg.cross(v1 - v0, v2 - v0)

    normals = torch.zeros_like(vertices)
    for i in range(3):
        normals.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), face_normals)
    return nn.functional.normalize(normals, dim=1)


def _compute_hks(vertices, faces, scales=None):
    """Heat kernel signature at multiple scales."""
    from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas
    import math
    if scales is None:
        scales = [math.exp(i * 0.5) for i in range(16)]  # 16 log-spaced scales

    L = cotangent_laplacian(vertices, faces)
    A = vertex_areas(vertices, faces)
    L_dense = L.to_dense()
    A_diag = torch.diag(A)
    A_inv = torch.diag(1.0 / A.clamp(min=1e-12))

    # Generalized eigenvalue problem: L v = lambda A v
    # Equivalent to: A^{-1} L v = lambda v
    M = A_inv @ L_dense
    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    # HKS(x, t) = sum_k exp(-lambda_k * t) * phi_k(x)^2
    k = min(50, vertices.shape[0])
    evals = eigenvalues[:k].clamp(min=0)
    evecs = eigenvectors[:, :k]

    hks_features = []
    for t in scales:
        heat = torch.exp(-evals * t)
        hks_t = (evecs ** 2) @ heat
        hks_features.append(hks_t)

    return torch.stack(hks_features, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--features', type=str, default='conformal',
                        choices=['xyz', 'hks', 'conformal', 'conformal+iso'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    print(f"Running ShapeNet classification with features={args.features}")
    print(f"Data dir: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    # Full training loop integrates with DiffusionNet
    # See: https://github.com/nmwsharp/diffusion-net
    print("TODO: integrate DiffusionNet training loop with ShapeNet dataloader")


if __name__ == '__main__':
    main()
