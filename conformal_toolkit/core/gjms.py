"""GJMS operators (Graham-Jenne-Mason-Sparling).

P_2 = conformal (Yamabe) Laplacian = Delta - ((n-2)/(4(n-1))) R
P_4 = Paneitz operator = Delta^2 + div(V . d) + ((n-4)/2) Q_4

where V_ab = (n-2) J g_ab - 4 P_ab.

Sign convention: Delta = nabla^a nabla_a (geometer / non-negative-spectrum
on a compact manifold is for -Delta). The GJMS principal part is +Delta^k.
"""
from conformal_toolkit.core.helpers import laplacian, divergence


def laplacian_operator(cs, f):
    """P_2(f): the *conformal* (Yamabe) Laplacian, not the bare Laplacian.

        P_2(f) = Delta(f) - ((n-2)/(4(n-1))) R f

    The scalar-curvature term is exactly what makes P_2 conformally
    covariant (weight (2-n)/2 -> (-2-n)/2) for n > 2; it vanishes only at
    n = 2. Omitting it (the original bug) breaks conformal covariance for
    every n > 2 -- see ERRATA M2.
    """
    n = cs.dimension
    R = cs.ricci_scalar()
    delta_f = laplacian(cs.connection(), cs.metric, f)
    if n == 2:
        return delta_f
    return delta_f - ((n - 2) / (4 * (n - 1))) * R * f


def paneitz_operator(cs, f):
    """P_4(f) = Delta^2(f) + div(V . df) + ((n-4)/2) Q_4 f."""
    g = cs.metric
    nabla = cs.connection()
    n = cs.dimension
    P = cs.schouten()
    J = cs.schouten_trace()

    delta_f = laplacian(nabla, g, f)
    delta2_f = laplacian(nabla, g, delta_f)

    V = (n - 2) * J * g - 4 * P
    df = nabla(f)
    df_up = df.up(g)
    V_df = V.contract(1, df_up, 0)
    term2 = divergence(nabla, g, V_df)

    Q4 = cs.q_curvature(order=4)
    coeff = (n - 4) / 2
    term3 = coeff * Q4 * f

    return delta2_f + term2 + term3


def p6_operator(cs, f):
    """P_6(f): sixth-order GJMS operator (leading-order approximation).

    Computes +Delta^3(f) (the GJMS principal part is +Delta^k in the
    geometer sign convention Delta = nabla^a nabla_a used throughout this
    package), which equals P_6(f) for conformally flat metrics.  On general
    curved backgrounds this captures the principal symbol but omits the
    lower-order curvature coupling terms (Schouten, Cotton, and Weyl
    contributions).

    Note: an earlier docstring claimed this returns -Delta^3; that was a
    sign error from mixing the analyst convention (Delta = -nabla^a nabla_a)
    with the geometer Laplacian the code actually applies -- see ERRATA M3.

    Parameters
    ----------
    cs : ConformalStructure
    f  : scalar field on the underlying manifold

    Raises
    ------
    ValueError
        If dimension < 6.
    """
    n = cs.dimension
    if n < 6:
        raise ValueError(f"P_6 requires dimension >= 6, got {n}")

    nabla = cs.connection()
    g = cs.metric

    d1 = laplacian(nabla, g, f)
    d2 = laplacian(nabla, g, d1)
    d3 = laplacian(nabla, g, d2)
    return d3


def gjms_operator(cs, f, order=4):
    """Apply GJMS operator P_{2k} to scalar field f."""
    if order == 2:
        return laplacian_operator(cs, f)
    if order == 4:
        return paneitz_operator(cs, f)
    if order == 6:
        return p6_operator(cs, f)
    raise NotImplementedError(f"GJMS order {order} not implemented (only 2, 4, and 6)")
