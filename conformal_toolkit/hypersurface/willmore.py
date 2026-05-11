"""Willmore energy densities for conformal hypersurfaces.

The Willmore functional generalises to conformal geometry via:
    W_2 = integral |L_1|^2 dA   (classical Willmore)
    W_4 = integral |L_2|^2 dA   (higher-order)

This module computes the *integrands* (pointwise densities).
"""
from conformal_toolkit.hypersurface.conformal_fundamental_form import (
    conformal_fundamental_form_L1,
    conformal_fundamental_form_L2,
    trace_with_metric,
)


def willmore_density_W2(h, L):
    """Classical Willmore integrand |L_1|^2 = h^{ac} h^{bd} (L_1)_{ab} (L_1)_{cd}.

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2).

    Returns:
        Scalar field |L_1|^2.
    """
    L1 = conformal_fundamental_form_L1(h, L)
    L1_up = L1.up(h)
    # Full contraction of L1 (0,2) with L1_up (2,0)
    result = L1.contract(0, L1_up, 0)   # contract first pair -> (1,1)
    return result.trace()                 # contract remaining pair -> scalar


def willmore_density_W4(h, L, nabla_n_L1=None):
    """Higher-order Willmore integrand |L_2|^2.

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2).
        nabla_n_L1: optional normal derivative of L_1.

    Returns:
        Scalar field |L_2|^2 (or |L_2^{alg}|^2 when nabla_n_L1 is None).
    """
    L2 = conformal_fundamental_form_L2(h, L, nabla_n_L1=nabla_n_L1)
    L2_up = L2.up(h)
    result = L2.contract(0, L2_up, 0)
    return result.trace()
