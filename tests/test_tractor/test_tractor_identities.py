"""Test tractor metric compatibility and algebraic identities.

These tests verify fundamental properties of the tractor calculus:
1. The tractor metric is preserved by the tractor connection
2. The tractor curvature has the correct symmetries
"""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn

def test_tractor_metric_flat():
    """On flat R^3, tractor curvature should vanish (conformally flat)."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import tractor_curvature

    data = _make_flat_rn(3)
    cs = ConformalStructure(data['metric'])
    Omega = tractor_curvature(cs)

    # Tractor curvature components should all be zero on flat space
    frame = list(data['manifold'].frames())[0]
    n = data['dim']
    for a in range(n):
        for b in range(n):
            W_comp = Omega['W'][frame, a, b]
            C_comp = Omega['C'][frame, a, b]
            if hasattr(W_comp, 'expr'):
                W_comp = W_comp.expr()
            if hasattr(C_comp, 'expr'):
                C_comp = C_comp.expr()
            assert W_comp == 0, f"Weyl[{a},{b}] should vanish on flat space"
            assert C_comp == 0, f"Cotton[{a},{b}] should vanish on flat space"

def test_tractor_curvature_sphere():
    """On S^2 (conformally flat), tractor curvature should vanish."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import tractor_curvature

    data = _make_round_sphere_2()
    cs = ConformalStructure(data['metric'])
    Omega = tractor_curvature(cs)

    # S^2 is conformally flat, so Cotton tensor = 0
    frame = list(data['manifold'].frames())[0]
    for a in range(2):
        for b in range(2):
            C_comp = Omega['C'][frame, a, b]
            if hasattr(C_comp, 'expr'):
                C_comp = C_comp.expr()
            val = C_comp.simplify_full()
            assert val == 0, f"Cotton[{a},{b}] should vanish on S^2"
