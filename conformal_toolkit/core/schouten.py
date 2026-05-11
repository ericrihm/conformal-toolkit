"""Schouten tensor computation.

The Schouten tensor is the trace-adjusted Ricci tensor:
    P_ab = (1/(n-2)) (Ric_ab - R/(2(n-1)) g_ab)
"""


def compute_schouten(g):
    """Compute the Schouten tensor P_ab."""
    n = g.domain().dim()
    Ric = g.ricci()
    R = g.ricci_scalar()

    if n == 2:
        # In 2D, Ric = (R/2)g, so the general formula has a 1/(n-2) pole.
        # Convention: P = (R/4)g, giving P = g/2 on round S^2 (R=2).
        return (R / 4) * g
    return (Ric - (R / (2 * (n - 1))) * g) * (1 / (n - 2))


def schouten_trace(g):
    """Compute J = trace(P) = R / (2(n-1))."""
    n = g.domain().dim()
    R = g.ricci_scalar()
    if n == 1:
        return R
    return R / (2 * (n - 1))
