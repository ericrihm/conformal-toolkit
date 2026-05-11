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

    For n != 4:
        (g_4)_{ab} = (1/(n-4)) * [(g_2)_a^c (g_2)_{cb}
                     - (1/(4*(n-1))) tr(g_2^2) (g_0)_{ab}]

    For n == 4 the metric expansion develops a log term and g_4 is only
    determined modulo the trace-free part.  We return the pure-trace piece
    that is algebraically determined: (1/8) tr(g_2^2) g_0.
    """
    n = g0.domain().dim()
    M = g0.domain()
    g2 = fg_coefficient_g2(g0)

    # Raise one index of g2 to form g2^a_b, then contract to get (g2 * g2)_{ab}.
    g2_up = g2.up(g0, 0)           # (1,1) tensor
    g2_sq = g2.contract(0, g2_up, 0)  # contract first index pair -> (0,2)

    # Trace: tr(g2^2) = (g2^a_b)(g2^b_a) = g2_sq.trace(0,1) but g2_sq is (0,2),
    # so we need to raise and trace.
    g2_sq_up = g2_sq.up(g0, 0)
    tr_g2_sq = g2_sq_up.trace(0, 1)

    frame = list(M.frames())[0]

    if n == 4:
        # Only the algebraically-fixed trace piece: (1/8) tr(g2^2) g0.
        g4 = (1 / 8) * tr_g2_sq * g0
        g4.set_name('g4')
        return g4

    prefactor = 1 / (n - 4)
    correction = (1 / (4 * (n - 1))) * tr_g2_sq * g0
    g4 = prefactor * (g2_sq - correction)
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
