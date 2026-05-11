"""Tests for the extrinsic GJMS operator P_2."""
from sage.all import Rational


def _make_flat_plane():
    from sage.all import Manifold
    M = Manifold(2, 'Plane', structure='Riemannian')
    X = M.chart('x y')
    x, y = X[:]
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = 1
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 0
    L[0, 1] = 0
    L[1, 1] = 0
    return h, L, X


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


def test_P2_on_flat_is_laplacian():
    """On flat plane H = 0, so P_2(f) = Delta(f)."""
    from conformal_toolkit.hypersurface.extrinsic_gjms import extrinsic_gjms_P2
    h, L, X = _make_flat_plane()
    x, y = X[:]
    M = h.domain()
    f = M.scalar_field(x**2 + y**2)
    P2f = extrinsic_gjms_P2(h, L, f)
    val = P2f.expr()
    # Delta(x^2 + y^2) = 2 + 2 = 4 on flat R^2
    assert bool(val == 4), f"P_2(x^2+y^2) on flat plane should be 4, got {val}"


def test_P2_on_flat_harmonic():
    """P_2 of a harmonic function on flat plane = 0."""
    from conformal_toolkit.hypersurface.extrinsic_gjms import extrinsic_gjms_P2
    h, L, X = _make_flat_plane()
    x, y = X[:]
    M = h.domain()
    f = M.scalar_field(x**2 - y**2)  # harmonic
    P2f = extrinsic_gjms_P2(h, L, f)
    val = P2f.expr()
    assert bool(val == 0), f"P_2(x^2-y^2) on flat plane should be 0, got {val}"


def test_P2_on_sphere_constant():
    """P_2(1) = (n/2 - 1) H * 1 = 0 on S^2 (since n=2, coeff=0)."""
    from conformal_toolkit.hypersurface.extrinsic_gjms import extrinsic_gjms_P2
    h, L, X = _make_round_sphere_in_R3()
    M = h.domain()
    f = M.scalar_field(1)
    P2f = extrinsic_gjms_P2(h, L, f)
    val = P2f.expr()
    assert bool(val.simplify_full() == 0), (
        f"P_2(1) on S^2 should be 0 (n/2-1=0), got {val}"
    )
