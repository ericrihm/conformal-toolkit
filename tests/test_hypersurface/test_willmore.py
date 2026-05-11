"""Tests for Willmore density computations."""
from sage.all import Rational


def _make_round_sphere_in_R3():
    from sage.all import Manifold, sin
    M = Manifold(2, 'S2', structure='Riemannian')
    X = M.chart(r'theta:(0,pi):\theta phi:(0,2*pi):\phi')
    theta, phi = X[:]
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = sin(theta)**2
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 1
    L[1, 1] = sin(theta)**2
    return h, L, X


def _make_flat_plane():
    from sage.all import Manifold
    M = Manifold(2, 'Plane', structure='Riemannian')
    X = M.chart('x y')
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = 1
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 0
    L[0, 1] = 0
    L[1, 1] = 0
    return h, L, X


def _make_cylinder():
    from sage.all import Manifold
    M = Manifold(2, 'Cyl', structure='Riemannian')
    X = M.chart('u:(0,2*pi) z')
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = 1
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 1
    L[0, 1] = 0
    L[1, 1] = 0
    return h, L, X


def test_willmore_W2_sphere_vanishes():
    """S^2 is umbilical so W_2 = |L_1|^2 = 0."""
    from conformal_toolkit.hypersurface.willmore import willmore_density_W2
    h, L, X = _make_round_sphere_in_R3()
    W2 = willmore_density_W2(h, L)
    val = W2.expr()
    assert bool(val.simplify_full() == 0), f"W_2 on S^2 should be 0, got {val}"


def test_willmore_W2_flat_vanishes():
    from conformal_toolkit.hypersurface.willmore import willmore_density_W2
    h, L, X = _make_flat_plane()
    W2 = willmore_density_W2(h, L)
    val = W2.expr()
    assert bool(val == 0), f"W_2 on flat plane should be 0, got {val}"


def test_willmore_W2_cylinder_positive():
    """Cylinder has |L_1|^2 = 1/2."""
    from conformal_toolkit.hypersurface.willmore import willmore_density_W2
    h, L, X = _make_cylinder()
    W2 = willmore_density_W2(h, L)
    val = W2.expr()
    # L_1 = diag(1/2, -1/2), |L_1|^2 = (1/2)^2 + (-1/2)^2 = 1/2
    assert bool(val == Rational((1, 2))), f"W_2 on cylinder should be 1/2, got {val}"


def test_willmore_W4_sphere_vanishes():
    from conformal_toolkit.hypersurface.willmore import willmore_density_W4
    h, L, X = _make_round_sphere_in_R3()
    W4 = willmore_density_W4(h, L)
    val = W4.expr()
    assert bool(val.simplify_full() == 0), f"W_4^alg on S^2 should be 0, got {val}"


def test_willmore_W4_flat_vanishes():
    from conformal_toolkit.hypersurface.willmore import willmore_density_W4
    h, L, X = _make_flat_plane()
    W4 = willmore_density_W4(h, L)
    val = W4.expr()
    assert bool(val == 0), f"W_4^alg on flat plane should be 0, got {val}"
