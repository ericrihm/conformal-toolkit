"""Conformal fundamental forms for hypersurfaces.

Given a hypersurface Sigma^n inside an ambient (n+1)-manifold,
the conformal fundamental forms L_k generalise the classical
second fundamental form to the conformal setting.

References:
    S. Blitz, "Conformal hypersurface geometry" (2023).

Input convention:
    h  -- induced metric on the hypersurface (SageManifolds metric)
    L  -- second fundamental form as a (0,2) tensor on the hypersurface
"""


def trace_with_metric(h, T):
    """Compute the metric trace h^{ab} T_{ab}.

    Args:
        h: Riemannian metric on the hypersurface.
        T: (0,2) tensor field on the same manifold.

    Returns:
        Scalar field equal to h^{ab} T_{ab}.
    """
    # Raise one index to get a (1,1) tensor, then take the trace.
    T_mixed = T.up(h, 0)        # T^a{}_b  (1,1)
    return T_mixed.trace(0, 1)   # contract contra-0 with cov-1


def mean_curvature(h, L):
    """Mean curvature H = (1/n) h^{ab} L_{ab}.

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2) on the hypersurface.

    Returns:
        Scalar field H.
    """
    n = h.domain().dim()
    return trace_with_metric(h, L) / n


def conformal_fundamental_form_L1(h, L):
    """First conformal fundamental form (trace-free second fundamental form).

    L_1_{ab} = L_{ab} - H h_{ab}

    This is the conformally invariant (weight-1) trace-free part of L.
    It vanishes if and only if the hypersurface is umbilical.

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2) on the hypersurface.

    Returns:
        (0,2) tensor field L_1.
    """
    H = mean_curvature(h, L)
    return L - H * h


def conformal_fundamental_form_L2(h, L, nabla_n_L1=None):
    """Second conformal fundamental form.

    Full formula:
        L_2_{ab} = (nabla_n L_1)_{ab} + (L_1)_a^c (L_1)_{cb}
                   - (1/n)|L_1|^2 h_{ab}

    If *nabla_n_L1* (the normal derivative of L_1) is not supplied,
    only the algebraic (tangential) part is returned:

        L_2^{alg}_{ab} = (L_1)_a^c (L_1)_{cb} - (1/n)|L_1|^2 h_{ab}

    Args:
        h: induced metric on the hypersurface.
        L: second fundamental form (0,2) on the hypersurface.
        nabla_n_L1: optional (0,2) tensor giving the normal derivative
                    of L_1 along the ambient unit normal.

    Returns:
        (0,2) tensor field L_2 (or its algebraic part).
    """
    n = h.domain().dim()
    L1 = conformal_fundamental_form_L1(h, L)

    # L1_sq_{ab} = L1_{ac} L1^c{}_b
    L1_up = L1.up(h, 0)               # (1,1) tensor: (L_1)^c{}_b
    L1_sq = L1.contract(1, L1_up, 0)  # contract cov-c with contra-c -> (0,2)

    # |L_1|^2 = h^{ac} h^{bd} L1_{ab} L1_{cd}
    L1_norm_sq = trace_with_metric(h, L1_sq)

    algebraic = L1_sq - (L1_norm_sq / n) * h

    if nabla_n_L1 is not None:
        return nabla_n_L1 + algebraic

    return algebraic
