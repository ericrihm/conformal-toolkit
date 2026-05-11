"""GJMS operators (Graham-Jenne-Mason-Sparling).

P_2 = Delta (Laplacian)
P_4 = Paneitz operator = Delta^2 + div(V . d) + ((n-4)/2) Q_4

where V_ab = (n-2) J g_ab - 4 P_ab.
"""
from conformal_toolkit.core.helpers import laplacian, divergence


def laplacian_operator(cs, f):
    """P_2(f) = Delta(f) = nabla^a nabla_a f."""
    return laplacian(cs.connection(), cs.metric, f)


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


def gjms_operator(cs, f, order=4):
    """Apply GJMS operator P_{2k} to scalar field f."""
    if order == 2:
        return laplacian_operator(cs, f)
    if order == 4:
        return paneitz_operator(cs, f)
    raise NotImplementedError(f"GJMS order {order} not implemented (only 2 and 4)")
