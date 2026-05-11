"""Test tractor curvature (Weyl tensor)."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_tractor_curvature_flat_r4():
    """Tractor curvature vanishes on flat R^4."""
    data = _make_flat_rn(4)
    g = data['metric']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import tractor_curvature

    cs = ConformalStructure(g)
    W = tractor_curvature(cs)
    frame = list(data['manifold'].frames())[0]
    n = 4

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    comp = W[frame, i, j, k, l]
                    if hasattr(comp, 'expr'):
                        assert bool(comp.expr() == 0), \
                            f"W[{i},{j},{k},{l}] should be 0 on flat space, got {comp.expr()}"


def test_tractor_curvature_conformally_flat_s2():
    """Tractor curvature vanishes on S^2 (conformally flat in 2D).
    In 2D the Weyl tensor vanishes identically."""
    data = _make_round_sphere_2()
    g = data['metric']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import tractor_curvature

    cs = ConformalStructure(g)
    W = tractor_curvature(cs)
    frame = list(data['manifold'].frames())[0]
    n = 2

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    comp = W[frame, i, j, k, l]
                    if hasattr(comp, 'expr'):
                        assert bool(comp.expr().simplify_full() == 0), \
                            f"W[{i},{j},{k},{l}] should be 0 on S^2, got {comp.expr()}"


def test_cotton_tensor_flat_r3():
    """Cotton tensor vanishes on flat R^3."""
    data = _make_flat_rn(3)
    g = data['metric']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.tractor_curvature import cotton_tensor

    cs = ConformalStructure(g)
    C = cotton_tensor(cs)
    frame = list(data['manifold'].frames())[0]
    n = 3

    for a in range(n):
        for b in range(n):
            for c in range(n):
                comp = C[frame, a, b, c]
                if hasattr(comp, 'expr'):
                    assert bool(comp.expr() == 0), \
                        f"C[{a},{b},{c}] should be 0 on flat R^3, got {comp.expr()}"
