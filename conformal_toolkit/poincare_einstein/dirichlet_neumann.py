"""Conformal Dirichlet-to-Neumann operator and holographic stress tensor.

For a Poincaré-Einstein metric the DN operator maps boundary data to its
normal derivative.  At leading order: DN(g_0) = g_2 = -P(g_0).
"""
from conformal_toolkit.core.schouten import compute_schouten, schouten_trace


def dirichlet_to_neumann(g0):
    """Compute the conformal Dirichlet-to-Neumann operator.

    For a PE metric, this relates boundary data to its normal derivative.
    At leading order: DN(g_0) = g_2 = -P(g_0).

    Returns
    -------
    (0,2) tensor field on the boundary manifold.
    """
    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_coefficient_g2
    return fg_coefficient_g2(g0)


def holographic_stress_tensor(g0, n):
    """Compute holographic stress tensor T_{ab}.

    T = n * g_n + (lower order trace terms).
    At leading non-trivial order: T = n * g_2 = -n * P(g_0).

    For n=3 (AdS4/CFT3) the trace anomaly is absent, so any *local*
    placeholder must be trace-free. The trace-free combination is

        T_{ab} = -3 * P_{ab} + J * (g_0)_{ab}      (tr T = -3J + J*3 = 0)

    where J = trace(P) = R/(2*(n-1)). The earlier coefficient (3/2)*J gave
    tr T = (3/2)J != 0, contradicting the stated absence of an anomaly (see
    ERRATA M13). NOTE: the genuine n=3 holographic stress tensor is
    T_{ij} = 3 g_{(3)ij}, which is undetermined *non-local* VEV data (free
    boundary condition), not any local function of P -- this branch returns
    only a trace-consistent local placeholder.

    For general n the holographic stress tensor at leading order is:
        T_{ab} = -n * P_{ab}

    Parameters
    ----------
    g0 : SageManifolds metric
        Boundary metric (dimension n).
    n : int
        Boundary dimension (bulk is n+1 dimensional).

    Returns
    -------
    (0,2) tensor field.
    """
    P = compute_schouten(g0)
    J = schouten_trace(g0)

    if n == 3:
        T = -3 * P + J * g0   # trace-free local placeholder; see ERRATA M13
    else:
        T = -n * P

    return T
