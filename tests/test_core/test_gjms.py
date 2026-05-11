from tests.conftest_sage import _make_flat_rn, _make_round_sphere_4

def test_paneitz_on_flat_is_bilaplacian():
    """On flat R^4, P_4(f) = Delta^2(f)."""
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import paneitz_operator
    cs = ConformalStructure(data['metric'])
    chart = data['chart']
    x0, x1, x2, x3 = chart[:]
    M = data['manifold']
    f = M.scalar_field(x0**2 + x1**2)
    P4f = paneitz_operator(cs, f)
    val = P4f.expr().simplify_full()
    assert bool(val == 0), f"P_4(x0^2+x1^2) on flat should be 0, got {val}"

def test_paneitz_on_flat_quartic():
    """On flat R^4, P_4(x0^4) = Delta^2(x0^4) = 24."""
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import paneitz_operator
    cs = ConformalStructure(data['metric'])
    chart = data['chart']
    x0 = chart[0]
    M = data['manifold']
    f = M.scalar_field(x0**4)
    P4f = paneitz_operator(cs, f)
    val = P4f.expr().simplify_full()
    assert bool(val == 24), f"P_4(x0^4) on flat should be 24, got {val}"

def test_paneitz_on_s4_constant():
    """On S^4 (n=4), P_4(constant) = 0."""
    data = _make_round_sphere_4()
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import paneitz_operator
    cs = ConformalStructure(data['metric'])
    M = data['manifold']
    f = M.scalar_field(1)
    P4f = paneitz_operator(cs, f)
    val = P4f.expr().simplify_full()
    assert bool(val == 0), f"P_4(1) on S^4 should be 0, got {val}"
