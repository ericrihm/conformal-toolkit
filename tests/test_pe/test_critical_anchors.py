"""Critical symbolic regression anchors for the May-2026 correctness audit.

These pin the corrected closed-form values of the CRITICAL/major numeric fixes
on geometries that actually DISCRIMINATE the bugs:

  * round S^3 (n != 4): the Fefferman-Graham g_4 (ERRATA C1) and renormalized-
    volume v_2 (ERRATA C2) bugs have the wrong n-dependence and coincide with
    the correct value ONLY at n = 4, so a non-4 dimension is required to catch
    them.

  * non-Einstein S^2(1) x S^2(2): Einstein and Ricci-flat metrics (e.g.
    Schwarzschild) have Bach = 0 and a Schouten that vanishes or is pure-trace,
    which masks Bach/Weyl-coefficient and conformal-Laplacian-curvature bugs.
    This product has Bach != 0 and a genuinely non-proportional Schouten, so it
    exercises those terms (ERRATA M2 and the Bach code path).

All expected values were derived in closed form and independently re-checked in
sympy (see ERRATA.md "How we caught them"). Requires SageMath; skipped if
SageManifolds is unavailable.
"""
import pytest

pytest.importorskip("sage.all", reason="SageMath required for symbolic anchors")

from sage.all import Rational
from tests.conftest_sage import _make_round_sphere_3, _make_product_s2_s2


def _frac(a, b):
    return Rational((a, b))


def _scalar(x):
    """Simplified symbolic expression from a Sage scalar field/expr."""
    if hasattr(x, "expr"):
        x = x.expr()
    return x.simplify_full()


def _comp(tensor, frame, i, j):
    c = tensor[frame, i, j]
    if hasattr(c, "expr"):
        c = c.expr()
    return c.simplify_full()


def _is_zero(expr):
    """Robust symbolic zero test (subtract-then-simplify is safer than ==)."""
    return bool(expr.simplify_full() == 0)


# ----------------------------------------------------------------------------
# Round S^3 (n != 4): discriminates C1 (g_4) and C2 (v_2).
# ----------------------------------------------------------------------------

def test_renormalized_volume_v2_s3_is_minus_three_quarters():
    """ERRATA C2: v_2 = -J/2 (n-independent) = -3/4 on S^3.

    The old -1/(n-2) J would give -3/2 here; the two agree only at n = 4.
    """
    g = _make_round_sphere_3()['metric']
    from conformal_toolkit.poincare_einstein.renormalized_volume import (
        renormalized_volume_coefficient,
    )
    v2 = _scalar(renormalized_volume_coefficient(g, order=2))
    assert _is_zero(v2 - _frac(-3, 4)), f"v_2 on S^3 should be -3/4, got {v2}"


def test_fg_g4_s3_is_one_sixteenth_g0():
    """ERRATA C1: on the round sphere g_4 = (1/16) g_0 (algebraic piece).

    The old spurious 1/(n-4) prefactor would give a different coefficient at
    n != 4 (e.g. 7/80 at n = 6); here every diagonal component must be 1/16 g0.
    """
    data = _make_round_sphere_3()
    g = data['metric']
    frame = data['chart'].frame()
    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_coefficient_g4
    g4 = fg_coefficient_g4(g)
    for i in range(3):
        gii = _comp(g, frame, i, i)
        g4ii = _comp(g4, frame, i, i)
        assert _is_zero(g4ii - gii / 16), \
            f"g_4[{i},{i}] should be (1/16) g_0[{i},{i}] = {gii / 16}, got {g4ii}"


def test_q4_s3_is_fifteen_eighths():
    """Branson Q_4 (order 4) on S^3 = -n/2 + n^3/8 = 15/8 at n = 3."""
    data = _make_round_sphere_3()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    q4 = _scalar(cs.q_curvature(order=4))
    assert _is_zero(q4 - _frac(15, 8)), f"Q_4 on S^3 should be 15/8, got {q4}"


# ----------------------------------------------------------------------------
# Non-Einstein S^2(1) x S^2(2): discriminates M2 and exercises Bach != 0.
# ----------------------------------------------------------------------------

def test_schouten_product_components():
    """Schouten on the non-Einstein product: P[0,0]=7/24, P[2,2]=-1/3."""
    data = _make_product_s2_s2()
    frame = data['chart'].frame()
    from conformal_toolkit.core.schouten import compute_schouten
    P = compute_schouten(data['metric'])
    p00 = _comp(P, frame, 0, 0)
    p22 = _comp(P, frame, 2, 2)
    assert _is_zero(p00 - _frac(7, 24)), f"P[0,0] should be 7/24, got {p00}"
    assert _is_zero(p22 - _frac(-1, 3)), f"P[2,2] should be -1/3, got {p22}"


def test_schouten_trace_product_is_five_twelfths():
    """J = trace(P) = R/(2(n-1)) = 5/12 on S^2(1) x S^2(2)."""
    g = _make_product_s2_s2()['metric']
    from conformal_toolkit.core.schouten import schouten_trace
    J = _scalar(schouten_trace(g))
    assert _is_zero(J - _frac(5, 12)), f"J on S^2xS^2 should be 5/12, got {J}"


def test_conformal_laplacian_has_curvature_term_product():
    """ERRATA M2: the conformal Laplacian P_2 carries -(n-2)/(4(n-1)) R.

    On a CONSTANT field f = 1 the bare Laplacian gives 0, so P_2(1) isolates the
    curvature term: -(n-2)/(4(n-1)) R = -(2/12)(5/2) = -5/12 on this metric.
    The pre-fix bare-Laplacian implementation returned 0 here.
    """
    data = _make_product_s2_s2()
    M = data['manifold']
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    f = M.scalar_field(1)
    p2f = _scalar(cs.gjms_operator(f, order=2))
    assert _is_zero(p2f - _frac(-5, 12)), \
        f"P_2(1) should be -5/12 (the curvature term), got {p2f}"


def test_bach_nonzero_on_non_einstein_product():
    """The Bach tensor is NOT identically zero on the non-Einstein product.

    Schwarzschild/Einstein metrics have Bach = 0 and would mask Bach-coefficient
    bugs; here Cotton = 0 but the algebraic P^{cd} W part is nonzero, so a
    correct Bach implementation must return a nonzero tensor.
    """
    data = _make_product_s2_s2()
    frame = data['chart'].frame()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    B = cs.bach()
    nonzero = any(
        not _is_zero(_comp(B, frame, i, j))
        for i in range(4) for j in range(4)
    )
    assert nonzero, "Bach should be nonzero on the non-Einstein S^2(1)xS^2(2)"
