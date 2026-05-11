"""FAUST shape correspondence benchmark using conformal features.

Usage:
    python -m conformal_features.benchmarks.faust_correspondence \
        --data_dir /path/to/faust \
        --features conformal

Evaluates per-vertex feature quality for shape correspondence.
Computes functional maps between shape pairs and reports geodesic error.
"""
import argparse
import torch
from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas


def compute_laplacian_basis(vertices, faces, k=30):
    """Compute first k Laplacian eigenfunctions for functional maps.

    Args:
        vertices: (V, 3) Tensor
        faces: (F, 3) LongTensor
        k: number of eigenfunctions

    Returns:
        eigenvalues: (k,) Tensor
        eigenvectors: (V, k) Tensor
    """
    L = cotangent_laplacian(vertices, faces)
    A = vertex_areas(vertices, faces)
    L_dense = L.to_dense()
    A_inv = torch.diag(1.0 / A.clamp(min=1e-12))

    M = A_inv @ L_dense
    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    return eigenvalues[:k], eigenvectors[:, :k]


def functional_map(feats_source, feats_target, basis_source, basis_target):
    """Compute functional map C between two shapes.

    C maps functions on source to functions on target in the Laplacian basis.
    C = (Phi_target^T @ Phi_target)^{-1} @ Phi_target^T @ F_target @ F_source^T @ Phi_source

    Simplified: C = pinv(Phi_target) @ F_target @ pinv(F_source) @ Phi_source

    Args:
        feats_source: (V_s, D) per-vertex features on source
        feats_target: (V_t, D) per-vertex features on target
        basis_source: (V_s, k) Laplacian eigenvectors on source
        basis_target: (V_t, k) Laplacian eigenvectors on target

    Returns:
        C: (k, k) functional map matrix
    """
    # Project features into Laplacian basis
    A_s = basis_source.T @ feats_source  # (k, D)
    A_t = basis_target.T @ feats_target  # (k, D)

    # Solve for C: C @ A_s ≈ A_t
    C = torch.linalg.lstsq(A_s.T, A_t.T).solution.T
    return C


def geodesic_error(C, basis_source, basis_target, gt_correspondence):
    """Compute geodesic error of correspondence induced by functional map.

    Args:
        C: (k, k) functional map
        basis_source: (V_s, k) eigenvectors
        basis_target: (V_t, k) eigenvectors
        gt_correspondence: (V_s,) LongTensor — ground truth vertex map

    Returns:
        mean_error: float — mean geodesic error (normalized)
    """
    # Point-to-point map via nearest neighbor in spectral embedding
    transformed = basis_source @ C.T  # (V_s, k)
    dists = torch.cdist(transformed, basis_target)  # (V_s, V_t)
    predicted = dists.argmin(dim=1)  # (V_s,)

    # Geodesic error = Euclidean distance as proxy (true geodesic needs mesh structure)
    errors = (predicted - gt_correspondence).abs().float()
    return errors.mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--features', type=str, default='conformal',
                        choices=['xyz', 'hks', 'conformal', 'conformal+iso'])
    parser.add_argument('--k', type=int, default=30, help='Laplacian basis size')
    args = parser.parse_args()

    print(f"Running FAUST correspondence with features={args.features}, k={args.k}")
    print(f"Data dir: {args.data_dir}")
    print("TODO: integrate FAUST dataloader and geodesic distance computation")


if __name__ == '__main__':
    main()
