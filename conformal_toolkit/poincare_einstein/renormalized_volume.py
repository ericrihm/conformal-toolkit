"""Renormalized volume coefficients for Poincaré-Einstein metrics.

The renormalized volume V_ren appears in the asymptotic expansion of the
bulk volume integral after holographic renormalization.  The expansion
produces local densities v_k on the boundary:

    Vol_epsilon ~ sum_k c_k(n) * epsilon^{k-n} * int v_k dV

v_0 = 1 (volume form coefficient, trivial)
v_2 = -(1/(n-2)) * J  (related to Schouten trace J = R/(2(n-1)))

Returns the LOCAL density (integrand), not an integrated quantity.
"""
from conformal_toolkit.core.schouten import schouten_trace


def renormalized_volume_coefficient(g0, order=2):
    """Compute the renormalized volume coefficient v_k from boundary metric.

    Returns the local density (integrand) for v_k.

    Parameters
    ----------
    g0 : SageManifolds metric
        Boundary metric of dimension n.
    order : int
        0 – returns 1 (trivial v_0 density).
        2 – returns -(1/(n-2)) * J where J = R/(2*(n-1)).

    Returns
    -------
    Scalar expression (symbolic).

    Raises
    ------
    NotImplementedError
        If order is not 0 or 2.
    """
    if order == 0:
        from sage.all import Integer
        return Integer(1)

    if order == 2:
        n = g0.domain().dim()
        if n == 2:
            # In 2D the 1/(n-2) pole is regularized; the coefficient is finite
            # and given by the conformal factor convention: v_2 = J/2.
            J = schouten_trace(g0)
            return J / 2
        J = schouten_trace(g0)
        return -(1 / (n - 2)) * J

    raise NotImplementedError(
        f"Renormalized volume coefficient v_{order} not implemented (only 0 and 2)"
    )
