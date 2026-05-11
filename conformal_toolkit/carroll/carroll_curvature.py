"""Carroll curvature tensors.

Provides:
- :func:`carroll_spatial_riemann` — spatial Riemann tensor R^i_{jkl}
- :func:`carroll_spatial_ricci`   — spatial Ricci tensor R_{jl}
- :func:`carroll_electric_field`  — Carroll electric field E_{ij} = (1/2) £_v h
"""

from sage.all import SR


# ---------------------------------------------------------------------------
# Spatial Riemann tensor
# ---------------------------------------------------------------------------

def carroll_spatial_riemann(carroll_struct):
    """Compute the spatial Riemann tensor R^i_{jkl} from the spatial metric h.

    Uses the standard Riemann formula from the spatial Christoffel symbols:

        R^i_{jkl} = ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj}
                    + Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj}

    All indices are spatial (0-based spatial indexing matching
    :func:`~conformal_toolkit.carroll.carroll_connection.spatial_christoffel`).

    Parameters
    ----------
    carroll_struct : CarrollStructure

    Returns
    -------
    dict
        Mapping ``(i, j, k, l)`` → symbolic expression.
    """
    from conformal_toolkit.carroll.carroll_connection import spatial_christoffel

    manifold = carroll_struct.manifold
    dim = carroll_struct.dimension
    n = dim - 1  # spatial dimensions

    chart = manifold.default_chart()
    coords = chart[:]

    Gamma = spatial_christoffel(carroll_struct)

    riemann = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    x_k = coords[k + 1]
                    x_l = coords[l + 1]

                    # ∂_k Γ^i_{lj} - ∂_l Γ^i_{kj}
                    val = (Gamma[(i, l, j)].diff(x_k)
                           - Gamma[(i, k, j)].diff(x_l))

                    # + Γ^i_{km} Γ^m_{lj} - Γ^i_{lm} Γ^m_{kj}
                    for m in range(n):
                        val += (Gamma[(i, k, m)] * Gamma[(m, l, j)]
                                - Gamma[(i, l, m)] * Gamma[(m, k, j)])

                    riemann[(i, j, k, l)] = val.simplify_full()

    return riemann


# ---------------------------------------------------------------------------
# Spatial Ricci tensor
# ---------------------------------------------------------------------------

def carroll_spatial_ricci(carroll_struct):
    """Compute the spatial Ricci tensor R_{jl} = R^i_{jil}.

    Parameters
    ----------
    carroll_struct : CarrollStructure

    Returns
    -------
    dict
        Mapping ``(j, l)`` → symbolic expression (spatial 0-based indices).
    """
    dim = carroll_struct.dimension
    n = dim - 1

    riemann = carroll_spatial_riemann(carroll_struct)

    ricci = {}
    for j in range(n):
        for l in range(n):
            val = SR(0)
            for i in range(n):
                val += riemann.get((i, j, i, l), SR(0))
            ricci[(j, l)] = val.simplify_full()

    return ricci


# ---------------------------------------------------------------------------
# Carroll electric field
# ---------------------------------------------------------------------------

def carroll_electric_field(carroll_struct, extrinsic_curvature=None):
    """Compute the Carroll electric field E_{ij} = (1/2) £_v h_{ij}.

    The Lie derivative of h along v is computed component-wise:

        (£_v h)_{ij} = v^k ∂_k h_{ij} + h_{kj} ∂_i v^k + h_{ik} ∂_j v^k

    This is analogous to the extrinsic curvature tensor in ADM formalism and
    measures the failure of h to be preserved along the Carroll time direction.

    Parameters
    ----------
    carroll_struct : CarrollStructure
    extrinsic_curvature : ignored (reserved for future extensions)

    Returns
    -------
    dict
        Mapping ``(i, j)`` → symbolic expression for E_{ij} using spatial
        0-based indices.
    """
    manifold = carroll_struct.manifold
    h = carroll_struct.spatial_metric
    v = carroll_struct.temporal_vector
    dim = carroll_struct.dimension
    n = dim - 1  # spatial dimensions

    chart = manifold.default_chart()
    frame = chart.frame()
    coords = chart[:]

    # Extract h components (full manifold indices 0..dim-1).
    h_comps = {}
    for mu in range(dim):
        for nu in range(dim):
            comp = h[frame, mu, nu]
            h_comps[(mu, nu)] = comp.expr() if hasattr(comp, 'expr') else SR(comp)

    # Extract v components (contravariant, full manifold indices).
    v_comps = {}
    for mu in range(dim):
        comp = v[frame, mu]
        v_comps[mu] = comp.expr() if hasattr(comp, 'expr') else SR(comp)

    # E_{ij} = (1/2) (£_v h)_{ij} for spatial i,j (manifold indices i+1, j+1).
    electric = {}
    for i in range(n):
        for j in range(n):
            mi = i + 1  # manifold index
            mj = j + 1

            # v^k ∂_k h_{ij}
            term1 = SR(0)
            for k in range(dim):
                term1 += v_comps[k] * h_comps[(mi, mj)].diff(coords[k])

            # h_{kj} ∂_i v^k
            term2 = SR(0)
            for k in range(dim):
                term2 += h_comps[(k, mj)] * v_comps[k].diff(coords[mi])

            # h_{ik} ∂_j v^k
            term3 = SR(0)
            for k in range(dim):
                term3 += h_comps[(mi, k)] * v_comps[k].diff(coords[mj])

            electric[(i, j)] = ((term1 + term2 + term3) / 2).simplify_full()

    return electric
