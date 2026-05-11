"""Spectral features from the Laplace-Beltrami operator.

Computes the first k eigenvectors of the generalized eigenvalue problem:
    L @ phi = lambda * M @ phi

where L is the cotangent Laplacian and M is the mass matrix (diagonal
vertex areas). The eigenvectors form the Heat Kernel Signature (HKS)
and Wave Kernel Signature (WKS) basis.
"""
from __future__ import annotations
import torch
from conformal_features.discrete.mesh_utils import cotangent_laplacian, vertex_areas


def lbo_eigenvectors(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k: int = 16,
) -> dict[str, torch.Tensor]:
    """Compute the first k eigenvectors of the Laplace-Beltrami operator.

    Solves the generalized eigenvalue problem  L phi = lambda M phi
    by transforming to the standard problem  M^{-1/2} L M^{-1/2} psi = lambda psi,
    then recovering phi = M^{-1/2} psi.

    Args:
        vertices: (V, 3) Tensor — vertex positions.
        faces: (F, 3) LongTensor — triangle indices.
        k: Number of eigenpairs to return (smallest eigenvalues).

    Returns:
        dict with:
            'eigenvalues':  (k,) Tensor — sorted eigenvalues (ascending).
            'eigenvectors': (V, k) Tensor — corresponding eigenvectors.
    """
    V = vertices.shape[0]
    k = min(k, V)

    L_sparse = cotangent_laplacian(vertices, faces)
    L_dense = L_sparse.to_dense()

    areas = vertex_areas(vertices, faces).clamp(min=1e-12)

    # M^{-1/2} as a diagonal vector
    M_inv_sqrt = 1.0 / torch.sqrt(areas)

    # Transform:  A = M^{-1/2} L M^{-1/2}
    # A_ij = M_inv_sqrt[i] * L_ij * M_inv_sqrt[j]
    A = L_dense * M_inv_sqrt.unsqueeze(0) * M_inv_sqrt.unsqueeze(1)

    # Solve standard eigenvalue problem
    eigenvalues_all, psi = torch.linalg.eigh(A)

    # Recover eigenvectors in original space: phi = M^{-1/2} psi
    phi = M_inv_sqrt.unsqueeze(1) * psi

    # Take the first k
    eigenvalues = eigenvalues_all[:k]
    eigenvectors = phi[:, :k]

    return {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def heat_kernel_signature(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k: int = 16,
    t_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heat Kernel Signature (HKS).

    HKS at vertex i for time t:
        h(i, t) = sum_j exp(-lambda_j * t) * phi_j(i)^2

    Args:
        vertices: (V, 3) Tensor — vertex positions.
        faces: (F, 3) LongTensor — triangle indices.
        k: Number of eigenpairs to use.
        t_values: (T,) Tensor of time scales.  If None, uses
            ``torch.logspace(-2, 2, 16)``.

    Returns:
        hks: (V, T) Tensor — HKS values per vertex per time scale.
    """
    eig = lbo_eigenvectors(vertices, faces, k=k)
    eigenvalues = eig["eigenvalues"]   # (k,)
    eigenvectors = eig["eigenvectors"]  # (V, k)

    if t_values is None:
        t_values = torch.logspace(-2, 2, 16, dtype=vertices.dtype, device=vertices.device)

    # exp(-lambda_j * t):  (k,) x (T,) -> (k, T)
    exponents = torch.exp(-eigenvalues.unsqueeze(1) * t_values.unsqueeze(0))

    # phi_j(i)^2: (V, k)
    phi_sq = eigenvectors ** 2

    # h(i, t) = sum_j phi_j(i)^2 * exp(-lambda_j * t)  ->  (V, T)
    hks = phi_sq @ exponents

    return hks


def wave_kernel_signature(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k: int = 16,
    e_values: torch.Tensor | None = None,
    sigma: float | None = None,
) -> torch.Tensor:
    """Wave Kernel Signature (WKS).

    WKS at vertex i for energy e:
        w(i, e) = sum_j exp(-(e - log(lambda_j))^2 / (2 sigma^2)) * phi_j(i)^2

    Args:
        vertices: (V, 3) Tensor — vertex positions.
        faces: (F, 3) LongTensor — triangle indices.
        k: Number of eigenpairs to use.
        e_values: (E,) Tensor of energy levels.  If None, uses
            ``torch.linspace(log(lambda_1), log(lambda_k), 16)``.
        sigma: Bandwidth.  If None, uses
            ``(log(lambda_k) - log(lambda_1)) / k``.

    Returns:
        wks: (V, E) Tensor — WKS values per vertex per energy level.
    """
    eig = lbo_eigenvectors(vertices, faces, k=k)
    eigenvalues = eig["eigenvalues"]   # (k,)
    eigenvectors = eig["eigenvectors"]  # (V, k)

    # Skip near-zero eigenvalues for log computation
    # Use eigenvalues starting from the first positive one
    pos_mask = eigenvalues > 1e-10
    if pos_mask.sum() < 2:
        # Fallback: not enough positive eigenvalues
        V = vertices.shape[0]
        E = 16 if e_values is None else e_values.shape[0]
        return torch.zeros(V, E, dtype=vertices.dtype, device=vertices.device)

    pos_eigenvalues = eigenvalues[pos_mask]
    pos_eigenvectors = eigenvectors[:, pos_mask]
    log_eigenvalues = torch.log(pos_eigenvalues)

    if sigma is None:
        sigma = (log_eigenvalues[-1] - log_eigenvalues[0]).item() / pos_eigenvalues.shape[0]

    if e_values is None:
        e_values = torch.linspace(
            log_eigenvalues[0].item(),
            log_eigenvalues[-1].item(),
            16,
            dtype=vertices.dtype,
            device=vertices.device,
        )

    # exp(-(e - log(lambda_j))^2 / (2 sigma^2)):  (k_pos,) x (E,) -> (k_pos, E)
    diff = log_eigenvalues.unsqueeze(1) - e_values.unsqueeze(0)
    weights = torch.exp(-diff ** 2 / (2 * sigma ** 2))

    # phi_j(i)^2: (V, k_pos)
    phi_sq = pos_eigenvectors ** 2

    # w(i, e) = sum_j phi_j(i)^2 * weight_j(e)  ->  (V, E)
    wks = phi_sq @ weights

    return wks
