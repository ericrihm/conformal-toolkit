"""Tests for extrinsic Q-curvature."""
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


def test_q2_sphere():
    """q_2 = H = 1 on unit S^2."""
    from conformal_toolkit.hypersurface.extrinsic_q import extrinsic_q2
    h, L, X = _make_round_sphere_in_R3()
    q2 = extrinsic_q2(h, L)
    val = q2.expr()
    assert bool(val == 1), f"q_2 on S^2 should be 1, got {val}"


def test_q2_flat():
    from conformal_toolkit.hypersurface.extrinsic_q import extrinsic_q2
    h, L, X = _make_flat_plane()
    q2 = extrinsic_q2(h, L)
    val = q2.expr()
    assert bool(val == 0), f"q_2 on flat plane should be 0, got {val}"


def test_q2_cylinder():
    from conformal_toolkit.hypersurface.extrinsic_q import extrinsic_q2
    h, L, X = _make_cylinder()
    q2 = extrinsic_q2(h, L)
    val = q2.expr()
    assert bool(val == Rational((1, 2))), f"q_2 on cylinder should be 1/2, got {val}"


def test_q4_flat():
    """All terms vanish on flat plane."""
    from conformal_toolkit.hypersurface.extrinsic_q import extrinsic_q4
    h, L, X = _make_flat_plane()
    q4 = extrinsic_q4(h, L)
    val = q4.expr()
    assert bool(val == 0), f"q_4 on flat plane should be 0, got {val}"


def test_q4_sphere():
    """On S^2 in R^3: H = 1, L_1 = 0, Delta_h(H) = 0.
    q_4 = -0 + 1*0 + (2/2 - 1)*1 = 0.
    """
    from conformal_toolkit.hypersurface.extrinsic_q import extrinsic_q4
    h, L, X = _make_round_sphere_in_R3()
    q4 = extrinsic_q4(h, L)
    val = q4.expr()
    assert bool(val.simplify_full() == 0), f"q_4 on S^2 should be 0, got {val}"
