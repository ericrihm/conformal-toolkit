"""Test Schouten tensor against known closed-form results."""
from tests.conftest_sage import _make_round_sphere_2, _make_round_sphere_4, _make_flat_rn

def test_schouten_on_s2():
    data = _make_round_sphere_2()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    P = cs.schouten()
    g = data['metric']
    chart = data['chart']
    theta = data['coords'][0]
    from sage.all import sin
    p00 = P[chart.frame(), 0, 0].expr()
    assert bool(p00 == 1/2), f"P[0,0] on S^2 should be 1/2, got {p00}"
    p11 = P[chart.frame(), 1, 1].expr()
    expected = sin(theta)**2 / 2
    assert bool((p11 - expected).simplify_full() == 0), f"P[1,1] on S^2 should be {expected}, got {p11}"

def test_schouten_on_s4():
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    P = cs.schouten()
    g = data['metric']
    chart = data['chart']
    p00 = P[chart.frame(), 0, 0].expr()
    assert bool(p00 == 1/2), f"P[0,0] on S^4 should be 1/2, got {p00}"

def test_schouten_on_flat_vanishes():
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    P = cs.schouten()
    chart = data['chart']
    for i in range(4):
        for j in range(4):
            comp = P[chart.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr() == 0), f"P[{i},{j}] should be 0 on flat space"

def test_schouten_trace():
    """J = trace(P) = R/(2(n-1)). On S^4: R=12, J=12/6=2."""
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    J = cs.schouten_trace()
    val = J.expr()
    assert bool(val == 2), f"J on S^4 should be 2, got {val}"
