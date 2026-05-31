"""Extrinsic Q-curvature for conformal hypersurfaces.

The extrinsic Q-curvature q_{2k} is the hypersurface analogue of
the ambient Q-curvature.  These are the Euler-Lagrange densities
of the Willmore-type functionals.

    q_2 = H   (mean curvature)
    q_4 = Delta_h(H) + H |L_1|^2 + (lower-order curvature terms)
"""
from conformal_toolkit.hypersurface.conformal_fundamental_form import (
    conformal_fundamental_form_L1,
    mean_curvature,
    trace_with_metric,
)
from conformal_toolkit.core.helpers import laplacian


def extrinsic_q2(h, L):
    """Extrinsic Q-curvature of order 2: q_2 = H (mean curvature).

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2).

    Returns:
        Scalar field q_2 = H.
    """
    return mean_curvature(h, L)


def extrinsic_q4(h, L, nabla_n_L1=None):
    """Extrinsic Q-curvature of order 4.

    WARNING -- INCOMPLETE (see ERRATA M8). This returns only the purely
    extrinsic, conformally-flat-ambient piece

        q_4 = -Delta_h(H) + H |L_1|^2 + (n/2 - 1) H^3,

    which OMITS the intrinsic fourth-order term. The genuine extrinsic Q_4 is

        q_4 = Q_4^Sigma + (extrinsic couplings),
        Q_4^Sigma = -Delta_bar J_bar - 2|P_bar|^2 + (n/2) J_bar^2,

    plus the leading second-derivative coupling ~ (1/2) L_1 . Delta_bar L_1.
    Failing anchor: on the round S^4 (umbilic: L_1 = 0, H = 1) this routine
    returns 0 + 0 + (n/2-1) = 1, whereas the correct value must reduce to the
    intrinsic Branson Q_4 = (n-1)! = 6. Use for the umbilic-deviation
    (Willmore-type) content only, not as the true extrinsic Q_4.
    (Also: the module header writes +Delta_h H but the implementation uses
    -Delta_h H; the implementation's sign is kept.)

    where Delta_h is the Laplace-Beltrami operator on (Sigma, h).

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2).
        nabla_n_L1: unused, accepted for API uniformity.

    Returns:
        Scalar field q_4.
    """
    n = h.domain().dim()
    H = mean_curvature(h, L)
    nabla = h.connection()

    delta_H = laplacian(nabla, h, H)

    L1 = conformal_fundamental_form_L1(h, L)
    L1_up = L1.up(h)
    L1_norm_sq = L1.contract(0, L1_up, 0).trace()

    return -delta_H + H * L1_norm_sq + (n / 2 - 1) * H * H * H
