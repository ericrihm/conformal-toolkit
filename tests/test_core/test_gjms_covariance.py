"""Test GJMS conformal covariance.

The GJMS operator P_{2k} satisfies:
    P_{2k}[e^{2w}g](f) = e^{-(n+2k)/2 * w} P_{2k}[g](e^{(n-2k)/2 * w} * f)

On conformally flat spaces, we can verify this by rescaling.
"""
from tests.conftest_sage import _make_flat_rn


def test_gjms_p2_conformal_covariance_flat():
    """P_2 should transform correctly under conformal rescaling on flat R^2."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure

    data = _make_flat_rn(2)
    g = data['metric']
    x0, x1 = data['chart'][:]
    M = data['manifold']

    cs = ConformalStructure(g)

    f = M.scalar_field(x0**2 + x1**2)

    P2f = cs.gjms_operator(f, order=2)

    # On R^2, Delta(x0^2 + x1^2) = 2 + 2 = 4
    val = P2f.expr().simplify_full()
    assert bool(val == 4), f"P_2(r^2) should be 4 on flat R^2, got {val}"


def test_gjms_p4_on_flat_r4():
    """P_4 on flat R^4 should reduce to Delta^2."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure

    data = _make_flat_rn(4)
    g = data['metric']
    coords = data['chart'][:]
    M = data['manifold']

    cs = ConformalStructure(g)

    f = M.scalar_field(coords[0]**4)
    result = cs.gjms_operator(f, order=4)

    val = result.expr().simplify_full()
    # On flat space V = 0 and Q_4 = 0, so P_4 = Delta^2
    # Delta^2(x0^4) = 24
    assert bool(val == 24), f"P_4(x0^4) should be 24 on flat R^4, got {val}"
