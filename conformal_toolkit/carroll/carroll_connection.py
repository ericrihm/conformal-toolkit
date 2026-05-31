"""Carroll connection: Christoffel-like symbols adapted to a Carroll structure.

For a Carroll manifold with adapted coordinates (t, x^i) where v = ∂_t, this
module uses the block structure:

    Γ^t_{μν} = 0   (all)
    Γ^i_{tt} = 0
    Γ^i_{tj} = 0
    Γ^i_{jk} = spatial Christoffel symbols from h_{jk}

SCOPE CAVEAT (see ERRATA M9). Carrollian connections are NOT unique (h is
degenerate), and the choice Γ^i_{tj} = 0 is correct ONLY when the Carroll
"electric field" E_{ij} = (1/2)(£_v h)_{ij} vanishes -- i.e. when the spatial
metric is time-independent (∂_t h_{ij} = 0). When ∂_t h_{ij} ≠ 0, metric
compatibility ∇h = 0 forces Γ^k_{t(i} h_{j)k} = (1/2) ∂_t h_{ij} ≠ 0, so the
correct symmetric part is Γ^i_{(tj)} = -(1/2) h^{ik}(£_v h)_{kj}. This module
otherwise computes a generically NONZERO carroll_electric_field, so presenting
Γ^i_{tj} = 0 as THE Carroll connection is internally inconsistent in the
time-dependent case. Treat the symbols below as the preserved-h
(£_v h = 0) representative.
"""

from sage.all import SR


def spatial_christoffel(carroll_struct):
    """Compute spatial Christoffel symbols Γ^i_{jk} from the spatial metric h.

    The symbols are computed via the standard Levi-Civita formula restricted
    to spatial coordinate directions (all indices excluding the time direction
    identified with the temporal vector v = ∂_t).

    Assumptions
    -----------
    The manifold has an adapted chart in which v = ∂_{x^0} (the first
    coordinate direction).  The spatial metric h has h_{0μ} = 0.

    Parameters
    ----------
    carroll_struct : CarrollStructure

    Returns
    -------
    dict
        Mapping ``(i, j, k)`` → symbolic expression for Γ^i_{jk}, where
        indices run over **spatial** directions (1-based in the full manifold,
        0-based in the returned dict keys).  Keys use 0-based *spatial*
        indexing: index ``s`` corresponds to manifold direction ``s+1``.
    """
    manifold = carroll_struct.manifold
    h = carroll_struct.spatial_metric
    dim = carroll_struct.dimension  # n+1

    chart = manifold.default_chart()
    frame = chart.frame()

    n = dim - 1  # number of spatial dimensions

    # Extract the n×n spatial metric matrix h_{ij} (i,j = 1..n in manifold coords).
    # We collect components and build the inverse symbolically.
    from sage.all import matrix, SR

    h_mat = matrix(SR, n, n)
    for i in range(n):
        for j in range(n):
            comp = h[frame, i + 1, j + 1]
            h_mat[i, j] = comp.expr() if hasattr(comp, 'expr') else SR(comp)

    # Compute the inverse spatial metric (needed for raising the first index).
    h_inv = h_mat.inverse()

    # Spatial coordinates (manifold indices 1..n).
    coords = chart[:]

    # Christoffel: Γ^i_{jk} = (1/2) h^{il} (∂_j h_{lk} + ∂_k h_{lj} - ∂_l h_{jk})
    # where all indices are spatial (0-based in h_mat / h_inv).
    christoffel = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                val = SR(0)
                for l in range(n):
                    h_lk = h_mat[l, k]
                    h_lj = h_mat[l, j]
                    h_jk = h_mat[j, k]
                    # Derivatives w.r.t. spatial coordinates (manifold index l+1)
                    x_j = coords[j + 1]
                    x_k = coords[k + 1]
                    x_l = coords[l + 1]
                    dj_hlk = h_lk.diff(x_j)
                    dk_hlj = h_lj.diff(x_k)
                    dl_hjk = h_jk.diff(x_l)
                    val += h_inv[i, l] * (dj_hlk + dk_hlj - dl_hjk) / 2
                christoffel[(i, j, k)] = val.simplify_full()

    return christoffel


def carroll_christoffel_full(carroll_struct):
    """Return the full (n+1)^3 Carroll Christoffel symbols as a dict.

    The time components (any index = 0) are all zero.  Spatial components are
    delegated to :func:`spatial_christoffel`.

    Parameters
    ----------
    carroll_struct : CarrollStructure

    Returns
    -------
    dict
        Mapping ``(μ, ν, ρ)`` → symbolic expression (manifold index convention).
    """
    dim = carroll_struct.dimension
    spatial = spatial_christoffel(carroll_struct)

    result = {}
    for mu in range(dim):
        for nu in range(dim):
            for rho in range(dim):
                if mu == 0 or nu == 0 or rho == 0:
                    result[(mu, nu, rho)] = SR(0)
                else:
                    # Translate to 0-based spatial indices
                    result[(mu, nu, rho)] = spatial.get(
                        (mu - 1, nu - 1, rho - 1), SR(0)
                    )
    return result
