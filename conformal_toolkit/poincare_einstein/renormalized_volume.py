"""Renormalized volume coefficients for Poincaré-Einstein metrics.

The renormalized volume V_ren appears in the asymptotic expansion of the
bulk volume integral after holographic renormalization.  The expansion
produces local densities v_k on the boundary:

    Vol_epsilon ~ sum_k c_k(n) * epsilon^{k-n} * int v_k dV

v_0 = 1 (volume form coefficient, trivial)
v_2 = -(1/2) * J  (related to Schouten trace J = R/(2(n-1)))

Returns the LOCAL density (integrand), not an integrated quantity.

The v_2 coefficient is n-INDEPENDENT: v_2 = -J/2 (Graham). The earlier
-(1/(n-2)) J prefactor was wrong for every n != 4 (it happens to coincide
at n=4, both giving -1 on S^4, which masked the bug in 4D-only tests), and
the n=2 special case had the wrong sign (+J/2). See ERRATA C2 and M12.
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
        2 – returns -(1/2) * J where J = R/(2*(n-1)).

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
        # v_2 = -J/2, independent of n (= -n/4 on the round S^n). Verified by
        # expanding the sphere volume density (1 - rho^2/4)^n, whose rho^2
        # coefficient is -n/4 = -J/2 with J = n/2. See ERRATA C2/M12.
        J = schouten_trace(g0)
        return -(1 / 2) * J

    raise NotImplementedError(
        f"Renormalized volume coefficient v_{order} not implemented (only 0 and 2)"
    )
