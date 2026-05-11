"""Tests for fundamental conformal geometry identities."""
from tests.conftest_sage import _make_round_sphere_2, _make_round_sphere_4, _make_flat_rn
from conformal_toolkit.core.conformal_structure import ConformalStructure


def test_weyl_conformally_invariant():
    """Weyl tensor should be the same (up to the conformal factor) under rescaling.

    On flat R^4, Weyl = 0. After rescaling by omega = x0,
    Weyl of g_hat should still vanish (flat is conformally flat).
    """
    data = _make_flat_rn(4)
    cs = ConformalStructure(data['metric'])
    chart = list(data['manifold'].top_charts())[0]
    x0 = chart[0]

    cs_hat = cs.under_rescaling(x0)
    W_hat = cs_hat.weyl()
    frame = list(data['manifold'].frames())[0]

    # Check all components vanish
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    val = W_hat[frame, i, j, k, l]
                    if hasattr(val, 'expr'):
                        val = val.expr()
                    assert val.simplify_full() == 0, \
                        f"W_hat[{i},{j},{k},{l}] should vanish for conformally flat metric"


def test_schouten_decomposition():
    """Ricci = (n-2)*P + J*g. Verify this identity on S^4."""
    data = _make_round_sphere_4()
    cs = ConformalStructure(data['metric'])
    g = data['metric']
    n = 4
    P = cs.schouten()
    J = cs.schouten_trace()
    Ric = cs.ricci()

    # (n-2)*P + J*g should equal Ric
    reconstructed = (n - 2) * P + J * g
    frame = list(data['manifold'].frames())[0]

    for i in range(n):
        for j in range(n):
            ric_val = Ric[frame, i, j]
            rec_val = reconstructed[frame, i, j]
            if hasattr(ric_val, 'expr'):
                ric_val = ric_val.expr()
            if hasattr(rec_val, 'expr'):
                rec_val = rec_val.expr()
            diff = (ric_val - rec_val).simplify_full()
            assert diff == 0, \
                f"Ric[{i},{j}] = {ric_val} != (n-2)P + Jg = {rec_val}"


def test_q2_equals_scalar_curvature_on_s2():
    """Q_2 = R (scalar curvature) is the simplest Q-curvature."""
    data = _make_round_sphere_2()
    cs = ConformalStructure(data['metric'])
    Q2 = cs.q_curvature(order=2)
    R = cs.ricci_scalar()

    q2_val = Q2.expr()
    r_val = R.expr()
    assert (q2_val - r_val).simplify_full() == 0


def test_gauss_bonnet_q4_s4():
    """On round S^4: Q_4 = 6 (known exact value).

    The integral of Q_4 over S^4 gives 8pi^2 * chi(S^4) = 16pi^2,
    so Q_4 = 16pi^2 / vol(S^4) = 16pi^2 / (8pi^2/3) = 6.
    """
    data = _make_round_sphere_4()
    cs = ConformalStructure(data['metric'])
    Q4 = cs.q_curvature(order=4)
    val = Q4.expr().simplify_full()
    # Q_4 on round S^4 should be 6
    assert bool(val == 6), f"Q_4 on S^4 should be 6, got {val}"


def test_bach_symmetric():
    """Bach tensor is symmetric: B_ab = B_ba."""
    data = _make_round_sphere_4()
    cs = ConformalStructure(data['metric'])
    B = cs.bach()
    frame = list(data['manifold'].frames())[0]

    for i in range(4):
        for j in range(i + 1, 4):
            b_ij = B[frame, i, j]
            b_ji = B[frame, j, i]
            if hasattr(b_ij, 'expr'):
                b_ij = b_ij.expr()
            if hasattr(b_ji, 'expr'):
                b_ji = b_ji.expr()
            assert (b_ij - b_ji).simplify_full() == 0, \
                f"B[{i},{j}] != B[{j},{i}]"
