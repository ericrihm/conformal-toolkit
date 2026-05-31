"""Enumeration of independent conformal hypersurface invariants.

Classification based on Blitz's conformal hypersurface invariant theory.
Invariants are organised by *weight* (order in derivatives of the embedding).

At each weight, the independent scalar invariants are formed from
contractions of the conformal fundamental forms L_k, the ambient Weyl
tensor restricted to Sigma, and the intrinsic Weyl tensor of Sigma.

KNOWN INCOMPLETENESS at weight 4 (see ERRATA M5). The weight-4 catalogue
below is a PARTIAL (purely algebraic, leading) basis. It under-counts:
  * it omits the independent algebraic quartic (|L_1|^2)^2, which is
    distinct from tr(L_1^4): in eigenvalues their difference is
    (sum lam_i^2)^2 - sum lam_i^4 = 2 * sum_{i<j} lam_i^2 lam_j^2 > 0
    whenever L_1 has >= 2 nonzero principal curvatures (e.g. trace-free
    eigenvalues (1,2,-3) give 196 vs 98); and
  * it omits the tangential-derivative invariants (e.g. |nabla_bar L_1|^2 /
    L_1 . Delta_bar L_1, |div L_1|^2) and the curvature coupling
    J_bar |L_1|^2 that a complete pointwise weight-4 classification carries.
So count_invariants(order=4, ...) is a LOWER BOUND, not the exact count.
"""


# ── catalogue ──────────────────────────────────────────────────────────

_INVARIANTS_WEIGHT_2 = [
    {
        'name': '|L_1|^2',
        'formula': 'h^{ac} h^{bd} (L_1)_{ab} (L_1)_{cd}',
        'order': 2,
        'description': (
            'Squared norm of the trace-free second fundamental form. '
            'The unique pointwise conformal hypersurface invariant at weight 2.'
        ),
    },
]

_INVARIANTS_WEIGHT_4_BASE = [
    {
        'name': '|L_2|^2',
        'formula': 'h^{ac} h^{bd} (L_2)_{ab} (L_2)_{cd}',
        'order': 4,
        'description': 'Squared norm of the second conformal fundamental form.',
    },
    {
        'name': 'tr(L_1^4)',
        'formula': '(L_1)^a_b (L_1)^b_c (L_1)^c_d (L_1)^d_a',
        'order': 4,
        'description': (
            'Fourth-order trace of L_1. Independent of |L_2|^2 when n >= 4.'
        ),
    },
    {
        'name': 'W_{nanb} L_1^{ab}',
        'formula': 'W_{n a n b} (L_1)^{ab}',
        'order': 4,
        'description': (
            'Contraction of the ambient Weyl tensor (normal-tangential '
            'components) with L_1. Vanishes when the ambient manifold is '
            'conformally flat.'
        ),
    },
    {
        'name': '|W_{nabc}|^2',
        'formula': 'W_{nabc} W_n{}^{abc}',
        'order': 4,
        'description': (
            'Squared norm of the normal-tangential Weyl component. '
            'An ambient invariant restricted to the hypersurface.'
        ),
    },
]


def _filter_by_dimension(invariants, ambient_dim):
    """Drop invariants that are dependent in low dimensions."""
    n = ambient_dim - 1  # hypersurface dimension
    filtered = []
    for inv in invariants:
        # tr(L_1^4) is dependent (proportional to |L_1|^4) when n <= 3
        if inv['name'] == 'tr(L_1^4)' and n <= 3:
            continue
        # Weyl-based invariants vanish identically when ambient_dim <= 3
        if 'W_' in inv['name'] and ambient_dim <= 3:
            continue
        # PROVISIONAL, UNVERIFIED (ERRATA M6): |W_{nabc}|^2 is dropped at
        # ambient_dim == 4. The ORIGINAL justification ("the Weyl is
        # self-dual") was mathematically FALSE -- a generic 4-manifold has
        # both W^+ and W^- nonzero and independent, and even W^- = 0 would
        # not make the scalar |W_{nabc}|^2 expressible via the other listed
        # invariants. We have NOT established a genuine algebraic dependence
        # here, so this drop (and hence count_invariants(4, 4)) is an
        # unverified placeholder, not a proven count. Contributors: please
        # supply a real identity or remove the drop. See ERRATA M6.
        if inv['name'] == '|W_{nabc}|^2' and ambient_dim == 4:
            continue
        filtered.append(inv)
    return filtered


# ── public API ─────────────────────────────────────────────────────────

def list_invariants(order, ambient_dim):
    """List independent conformal hypersurface invariants at a given weight.

    Args:
        order: weight of the invariants (2 or 4).
        ambient_dim: dimension of the ambient manifold (>= 3).

    Returns:
        List of dicts with keys 'name', 'formula', 'order', 'description'.

    Raises:
        ValueError: if ambient_dim < 3.
        NotImplementedError: if order is not 2 or 4.
    """
    if ambient_dim < 3:
        raise ValueError("Ambient dimension must be >= 3 for a codimension-1 hypersurface.")
    if order == 2:
        return list(_INVARIANTS_WEIGHT_2)  # always exactly one
    if order == 4:
        return _filter_by_dimension(_INVARIANTS_WEIGHT_4_BASE, ambient_dim)
    raise NotImplementedError(
        f"Invariant enumeration at order {order} not implemented (only 2 and 4)."
    )


def count_invariants(order, ambient_dim):
    """Count independent conformal hypersurface invariants at a given weight.

    Args:
        order: weight of the invariants (2 or 4).
        ambient_dim: dimension of the ambient manifold (>= 3).

    Returns:
        Integer count.
    """
    return len(list_invariants(order, ambient_dim))
