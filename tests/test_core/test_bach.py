from tests.conftest_sage import _make_round_sphere_4, _make_flat_rn, _make_schwarzschild

def test_bach_vanishes_on_s4():
    """S^4 is conformally flat, so Bach = 0."""
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    B = cs.bach()
    chart = data['chart']
    for i in range(4):
        for j in range(4):
            comp = B[chart.frame(), i, j]
            if hasattr(comp, 'expr'):
                val = comp.expr().simplify_full()
                assert bool(val == 0), f"Bach[{i},{j}] on S^4 should be 0, got {val}"

def test_bach_vanishes_on_flat():
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    B = cs.bach()
    chart = data['chart']
    for i in range(4):
        for j in range(4):
            comp = B[chart.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr() == 0), f"Bach[{i},{j}] on flat R^4 should be 0"

def test_bach_vanishes_on_schwarzschild():
    data = _make_schwarzschild()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    B = cs.bach()
    chart = data['chart']
    for i in range(4):
        for j in range(4):
            comp = B[chart.frame(), i, j]
            if hasattr(comp, 'expr'):
                val = comp.expr().simplify_full()
                assert bool(val == 0), f"Bach[{i},{j}] on Schwarzschild should be 0, got {val}"
