import pytest
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


# ---------- P_6 tests ----------

def test_p6_dimension_guard():
    """P_6 should raise ValueError for dimension < 6."""
    data = _make_flat_rn(4)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import p6_operator
    cs = ConformalStructure(data['metric'])
    M = data['manifold']
    f = M.scalar_field(1)
    with pytest.raises(ValueError, match="dimension >= 6"):
        p6_operator(cs, f)


def test_p6_on_flat_cubic_vanishes():
    """P_6(x0^3) = Delta^3(x0^3) = 0 on flat R^6.

    Delta(x0^3) = 6*x0, Delta^2(x0^3) = Delta(6*x0) = 0, so Delta^3 = 0.
    """
    data = _make_flat_rn(6)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import p6_operator
    cs = ConformalStructure(data['metric'])
    chart = data['chart']
    x0 = chart[0]
    M = data['manifold']
    f = M.scalar_field(x0**3)
    result = p6_operator(cs, f)
    val = result.expr().simplify_full()
    assert bool(val == 0), f"P_6(x0^3) on flat R^6 should be 0, got {val}"


def test_p6_on_flat_sextic():
    """P_6(x0^6) = Delta^3(x0^6) = 720 on flat R^6.

    d/dx0(x0^6) = 6x0^5, d^2/dx0^2(x0^6) = 30x0^4
    Delta(x0^6) = 30*x0^4
    Delta^2(x0^6) = Delta(30*x0^4) = 30*12*x0^2 = 360*x0^2
    Delta^3(x0^6) = Delta(360*x0^2) = 360*2 = 720
    """
    data = _make_flat_rn(6)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import p6_operator
    cs = ConformalStructure(data['metric'])
    chart = data['chart']
    x0 = chart[0]
    M = data['manifold']
    f = M.scalar_field(x0**6)
    result = p6_operator(cs, f)
    val = result.expr().simplify_full()
    assert bool(val == 720), f"P_6(x0^6) on flat R^6 should be 720, got {val}"


def test_gjms_dispatch_order_6():
    """gjms_operator(order=6) should dispatch to p6_operator."""
    data = _make_flat_rn(6)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    cs = ConformalStructure(data['metric'])
    M = data['manifold']
    f = M.scalar_field(1)
    result = cs.gjms_operator(f, order=6)
    val = result.expr().simplify_full()
    assert bool(val == 0), f"P_6(1) on flat R^6 should be 0, got {val}"


def test_p6_on_flat_constant_vanishes():
    """P_6(constant) = 0 on flat R^6."""
    data = _make_flat_rn(6)
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.core.gjms import p6_operator
    cs = ConformalStructure(data['metric'])
    M = data['manifold']
    f = M.scalar_field(42)
    result = p6_operator(cs, f)
    val = result.expr().simplify_full()
    assert bool(val == 0), f"P_6(42) on flat R^6 should be 0, got {val}"
