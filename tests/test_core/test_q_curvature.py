from sage.all import Rational
from tests.conftest_sage import _make_round_sphere_2, _make_round_sphere_4, _make_flat_rn

def test_q2_is_scalar_curvature():
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    Q2 = cs.q_curvature(order=2)
    R = data['metric'].ricci_scalar()
    diff = (Q2.expr() - R.expr()).simplify_full()
    assert bool(diff == 0), f"Q_2 should equal R, difference = {diff}"

def test_q4_on_s4():
    """Q_4 on S^4 = 4(16-4)/8 = 6."""
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    Q4 = cs.q_curvature(order=4)
    val = Q4.expr().simplify_full()
    assert bool(val == 6), f"Q_4 on S^4 should be 6, got {val}"

def test_q4_on_flat_vanishes():
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    Q4 = cs.q_curvature(order=4)
    val = Q4.expr()
    assert bool(val == 0), f"Q_4 on flat R^4 should be 0, got {val}"
