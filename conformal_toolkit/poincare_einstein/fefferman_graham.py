"""Fefferman-Graham expansion for Poincaré-Einstein metrics.

A Poincaré-Einstein metric near the boundary rho=0:
    g = rho^{-2} (drho^2 + g_rho)
where g_rho = g_0 + rho^2 g_2 + rho^4 g_4 + ...

The expansion coefficients are determined by the boundary metric g_0.
"""
from conformal_toolkit.core.schouten import compute_schouten


def fg_coefficient_g2(g0):
    """Compute g_2 in the FG expansion from boundary metric g_0.

    g_2 = -P(g_0) (negative of Schouten tensor of boundary metric).

    Only valid for n >= 3 (where n = boundary dimension).
    In 2D the formula still applies via the 2D Schouten convention.
    """
    return -compute_schouten(g0)


def fg_coefficient_g4(g0):
    """Compute g_4 in the FG expansion from boundary metric g_0.

    LEADING-ORDER / LOCALLY-CONFORMALLY-FLAT result. The full g_4 is

        (g_4)_{ab} = (1/4) (g_2)_a^c (g_2)_{cb}
                     + (1/(n-4)) * [Bach / Delta-P differential terms],

    where the *algebraic* g_2^2 piece carries the coefficient 1/4 and the
    1/(n-4) pole sits ONLY on the differential (Bach-tensor) terms that
    obstruct at n=4 (de Haro-Skenderis-Solodukhin / Graham). This routine
    implements the algebraic piece, which is EXACT on locally conformally
    flat boundaries (where the Bach terms vanish) and is the correct leading
    behaviour otherwise. The differential terms are not yet implemented.

    For n == 4 the trace-free part of g_4 is undetermined (a log term
    appears); only the pure-trace piece is algebraically fixed:

        g_4 = (1/16) tr(g_2^2) g_0    =>   tr g_4 = (1/4) tr(g_2^2) = (1/4)|P|^2.

    Verification anchor: on the hyperbolic filling of the round S^n
    (g_rho = (1 - rho^2/4)^2 g_0) one has g_2 = -(1/2)g_0 and the EXACT
    g_4 = (1/16) g_0 = (1/4)(g_2)^2 -- reproduced by this routine.

    The earlier version placed a spurious 1/(n-4) on the algebraic term
    (giving 7/80 g_0 instead of 1/16 g_0 at n=6) and used 1/8 instead of
    1/16 at n=4 (doubling the trace) -- see ERRATA C1 and M11.
    """
    n = g0.domain().dim()
    M = g0.domain()
    g2 = fg_coefficient_g2(g0)

    # Raise one index of g2 to form g2^a_b, then contract to get (g2 * g2)_{ab}.
    g2_up = g2.up(g0, 0)           # (1,1) tensor
    g2_sq = g2.contract(0, g2_up, 0)  # contract first index pair -> (0,2)

    if n == 4:
        # Trace tr(g2^2) = (g2^a_b)(g2^b_a): raise and trace the (0,2) square.
        g2_sq_up = g2_sq.up(g0, 0)
        tr_g2_sq = g2_sq_up.trace(0, 1)
        # Pure-trace piece (1/16) tr(g2^2) g0 => tr g4 = (1/4) tr(g2^2).
        g4 = (1 / 16) * tr_g2_sq * g0
        g4.set_name('g4')
        return g4

    # Algebraic piece, exact on locally conformally flat boundaries:
    g4 = (1 / 4) * g2_sq
    g4.set_name('g4')
    return g4


def fg_expansion(g0, order=2):
    """Return FG expansion coefficients {0: g_0, 2: g_2, ...}.

    Parameters
    ----------
    g0 : SageManifolds metric
        Boundary metric.
    order : int
        2 – returns {0: g_0, 2: g_2}.
        4 – also returns {4: g_4}.

    Returns
    -------
    dict mapping even integer -> metric coefficient tensor
    """
    if order not in (2, 4):
        raise NotImplementedError(f"FG expansion to order {order} not implemented")

    coeffs = {0: g0, 2: fg_coefficient_g2(g0)}

    if order >= 4:
        coeffs[4] = fg_coefficient_g4(g0)

    return coeffs
