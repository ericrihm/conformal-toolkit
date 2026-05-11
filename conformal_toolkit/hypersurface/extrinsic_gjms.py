"""Extrinsic GJMS operators for conformal hypersurfaces.

The extrinsic GJMS operator P_2^Sigma acts on scalars living on
the hypersurface and is the leading-order conformally covariant
differential operator incorporating extrinsic data:

    P_2^Sigma(f) = Delta_h(f) + (n/2 - 1) H f

where Delta_h is the Laplace-Beltrami operator on (Sigma, h)
and H is the mean curvature.
"""
from conformal_toolkit.core.helpers import laplacian
from conformal_toolkit.hypersurface.conformal_fundamental_form import mean_curvature


def extrinsic_gjms_P2(h, L, f):
    """Extrinsic GJMS operator P_2 acting on a scalar field f.

    P_2(f) = Delta_h(f) + (n/2 - 1) H f

    This is conformally covariant of bi-degree (n/2 - 1, n/2 + 1).

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2).
        f: scalar field on the hypersurface.

    Returns:
        Scalar field P_2(f).
    """
    n = h.domain().dim()
    nabla = h.connection()
    H = mean_curvature(h, L)
    delta_f = laplacian(nabla, h, f)
    return delta_f + (n / 2 - 1) * H * f
