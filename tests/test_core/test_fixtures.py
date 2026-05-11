"""Verify test metrics have correct basic properties."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn

def test_round_s2_scalar_curvature():
    data = _make_round_sphere_2()
    g = data['metric']
    R = g.ricci_scalar()
    theta = data['coords'][0]
    val = R.expr()
    assert bool(val == 2), f"Expected R=2 on S^2, got {val}"

def test_flat_r4_riemann_vanishes():
    data = _make_flat_rn(4)
    g = data['metric']
    Riem = g.riemann()
    chart = data['chart']
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    comp = Riem[chart.frame(), i, j, k, l]
                    if hasattr(comp, 'expr'):
                        assert bool(comp.expr() == 0), f"Riem[{i},{j},{k},{l}] = {comp.expr()}"
