"""Tests for conformal fundamental forms on known hypersurfaces.

Test geometries:
  - Round S^2 in R^3 (umbilical): L = h, so L_1 = 0.
  - Flat plane R^2 in R^3 (totally geodesic): L = 0, everything vanishes.
  - Cylinder S^1 x R in R^3 (non-umbilical): L has eigenvalues (1, 0).
"""
from tests.conftest_sage import _make_round_sphere_2


# ── helpers to build hypersurface data ────────────────────────────────

def _make_flat_plane():
    """R^2 as a flat hypersurface in R^3.  L = 0."""
    from sage.all import Manifold
    M = Manifold(2, 'Plane', structure='Riemannian')
    X = M.chart('x y')
    x, y = X[:]
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = 1
    # Zero second fundamental form
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 0
    L[0, 1] = 0
    L[1, 1] = 0
    return h, L, X


def _make_round_sphere_in_R3():
    """Unit S^2 in R^3.  L_{ab} = h_{ab} (all principal curvatures = 1)."""
    from sage.all import Manifold, sin
    M = Manifold(2, 'S2', structure='Riemannian')
    X = M.chart(r'theta:(0,pi):\theta phi:(0,2*pi):\phi')
    theta, phi = X[:]
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = sin(theta)**2
    # Second fundamental form = h for unit sphere
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 1
    L[1, 1] = sin(theta)**2
    return h, L, X


def _make_cylinder():
    """Unit cylinder S^1 x R in R^3.  Principal curvatures (1, 0)."""
    from sage.all import Manifold
    M = Manifold(2, 'Cyl', structure='Riemannian')
    # Use unique coord name 'u' to avoid assumption clashes with theta
    X = M.chart('u:(0,2*pi) z')
    u, z = X[:]
    h = M.metric('h')
    h[0, 0] = 1
    h[1, 1] = 1
    # L has eigenvalue 1 in u-direction, 0 in z-direction
    L = M.tensor_field(0, 2, sym=[(0, 1)])
    L[0, 0] = 1
    L[0, 1] = 0
    L[1, 1] = 0
    return h, L, X


# ── trace and mean curvature ──────────────────────────────────────────

def test_trace_with_metric_sphere():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import trace_with_metric
    h, L, X = _make_round_sphere_in_R3()
    tr = trace_with_metric(h, L)
    val = tr.expr()
    assert bool(val == 2), f"tr_h(L) on S^2 should be 2, got {val}"


def test_trace_with_metric_flat():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import trace_with_metric
    h, L, X = _make_flat_plane()
    tr = trace_with_metric(h, L)
    val = tr.expr()
    assert bool(val == 0), f"tr_h(L) on flat plane should be 0, got {val}"


def test_mean_curvature_sphere():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import mean_curvature
    h, L, X = _make_round_sphere_in_R3()
    H = mean_curvature(h, L)
    val = H.expr()
    assert bool(val == 1), f"H on unit S^2 should be 1, got {val}"


def test_mean_curvature_cylinder():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import mean_curvature
    h, L, X = _make_cylinder()
    H = mean_curvature(h, L)
    val = H.expr()
    from sage.all import Rational
    assert bool(val == Rational((1, 2))), f"H on cylinder should be 1/2, got {val}"


def test_mean_curvature_flat():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import mean_curvature
    h, L, X = _make_flat_plane()
    H = mean_curvature(h, L)
    val = H.expr()
    assert bool(val == 0), f"H on flat plane should be 0, got {val}"


# ── L_1 ──────────────────────────────────────────────────────────────

def test_L1_sphere_vanishes():
    """S^2 in R^3 is umbilical, so L_1 = 0."""
    from conformal_toolkit.hypersurface.conformal_fundamental_form import conformal_fundamental_form_L1
    h, L, X = _make_round_sphere_in_R3()
    L1 = conformal_fundamental_form_L1(h, L)
    for i in range(2):
        for j in range(2):
            comp = L1[X.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr().simplify_full() == 0), (
                    f"L_1[{i},{j}] on S^2 should be 0, got {comp.expr()}"
                )


def test_L1_flat_vanishes():
    """Flat plane has L = 0, so L_1 = 0."""
    from conformal_toolkit.hypersurface.conformal_fundamental_form import conformal_fundamental_form_L1
    h, L, X = _make_flat_plane()
    L1 = conformal_fundamental_form_L1(h, L)
    for i in range(2):
        for j in range(2):
            comp = L1[X.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr() == 0), (
                    f"L_1[{i},{j}] on flat plane should be 0, got {comp.expr()}"
                )


def test_L1_cylinder_nonzero():
    """Cylinder is not umbilical, so L_1 != 0."""
    from conformal_toolkit.hypersurface.conformal_fundamental_form import conformal_fundamental_form_L1
    h, L, X = _make_cylinder()
    L1 = conformal_fundamental_form_L1(h, L)
    # L_1 = L - (1/2)h, so L_1[0,0] = 1 - 1/2 = 1/2, L_1[1,1] = 0 - 1/2 = -1/2
    from sage.all import Rational
    val00 = L1[X.frame(), 0, 0].expr()
    val11 = L1[X.frame(), 1, 1].expr()
    assert bool(val00 == Rational((1, 2))), f"L_1[0,0] on cylinder should be 1/2, got {val00}"
    assert bool(val11 == Rational((-1, 2))), f"L_1[1,1] on cylinder should be -1/2, got {val11}"


def test_L1_is_tracefree_cylinder():
    """L_1 should be trace-free by construction."""
    from conformal_toolkit.hypersurface.conformal_fundamental_form import (
        conformal_fundamental_form_L1,
        trace_with_metric,
    )
    h, L, X = _make_cylinder()
    L1 = conformal_fundamental_form_L1(h, L)
    tr = trace_with_metric(h, L1)
    val = tr.expr()
    assert bool(val.simplify_full() == 0), f"tr(L_1) on cylinder should be 0, got {val}"


# ── L_2 (algebraic part) ─────────────────────────────────────────────

def test_L2_alg_sphere_vanishes():
    """L_1 = 0 on S^2, so L_2^alg = 0."""
    from conformal_toolkit.hypersurface.conformal_fundamental_form import conformal_fundamental_form_L2
    h, L, X = _make_round_sphere_in_R3()
    L2 = conformal_fundamental_form_L2(h, L)
    for i in range(2):
        for j in range(2):
            comp = L2[X.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr().simplify_full() == 0), (
                    f"L_2^alg[{i},{j}] on S^2 should be 0, got {comp.expr()}"
                )


def test_L2_alg_flat_vanishes():
    from conformal_toolkit.hypersurface.conformal_fundamental_form import conformal_fundamental_form_L2
    h, L, X = _make_flat_plane()
    L2 = conformal_fundamental_form_L2(h, L)
    for i in range(2):
        for j in range(2):
            comp = L2[X.frame(), i, j]
            if hasattr(comp, 'expr'):
                assert bool(comp.expr() == 0), (
                    f"L_2^alg[{i},{j}] on flat plane should be 0, got {comp.expr()}"
                )
