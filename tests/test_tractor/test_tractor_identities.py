"""Test tractor metric compatibility and algebraic identities.

These tests verify fundamental properties of the tractor calculus:
1. Tractor curvature (Weyl tensor) vanishes on conformally flat spaces
2. Cotton tensor vanishes on conformally flat spaces
"""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_tractor_curvature_flat():
    """On flat R^3, tractor curvature (Weyl) should vanish."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import tractor_curvature

    data = _make_flat_rn(3)
    cs = ConformalStructure(data['metric'])
    W = tractor_curvature(cs)

    frame = list(data['manifold'].frames())[0]
    n = data['dim']
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    comp = W[frame, a, b, c, d]
                    if hasattr(comp, 'expr'):
                        comp = comp.expr()
                    assert comp == 0, f"Weyl[{a},{b},{c},{d}] should vanish on flat R^3"


def test_cotton_tensor_sphere():
    """On S^2 (conformally flat in 2D), Cotton tensor should vanish."""
    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import cotton_tensor

    data = _make_round_sphere_2()
    cs = ConformalStructure(data['metric'])
    C = cotton_tensor(cs)

    frame = list(data['manifold'].frames())[0]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                comp = C[frame, a, b, c]
                if hasattr(comp, 'expr'):
                    comp = comp.expr()
                val = comp.simplify_full()
                assert val == 0, f"Cotton[{a},{b},{c}] should vanish on S^2"
