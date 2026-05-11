"""Holographic Weyl anomaly (trace anomaly) for even boundary dimension.

In the Fefferman-Graham framework the holographic Weyl anomaly encodes
the failure of the partition function to be conformally invariant.

- n=2 boundary: anomaly density = R/2 (related to scalar curvature)
- n=4 boundary: anomaly density = Q_4 (Branson's Q-curvature, type-A anomaly)
"""


def holographic_weyl_anomaly(g0):
    """Compute the holographic Weyl anomaly density for even boundary dimension.

    Returns the *local* anomaly density (the integrand), not an integral.

    For n=2: returns R/2 where R is the scalar curvature of g_0.
    For n=4: returns Q_4, Branson's Q-curvature of g_0.

    Parameters
    ----------
    g0 : SageManifolds metric
        Boundary metric of even dimension n=2 or n=4.

    Returns
    -------
    Scalar field expression representing the anomaly density.

    Raises
    ------
    ValueError
        If the boundary dimension is odd or not 2 or 4.
    NotImplementedError
        If n >= 6 (higher-order anomalies not implemented).
    """
    n = g0.domain().dim()

    if n % 2 != 0:
        raise ValueError(
            f"Holographic Weyl anomaly only defined for even boundary dimension, got n={n}"
        )

    if n == 2:
        R = g0.ricci_scalar()
        return R / 2

    if n == 4:
        from conformal_toolkit.core.conformal_structure import ConformalStructure
        cs = ConformalStructure(g0)
        return cs.q_curvature(order=4)

    raise NotImplementedError(
        f"Holographic Weyl anomaly for n={n} not implemented (only n=2,4)"
    )
