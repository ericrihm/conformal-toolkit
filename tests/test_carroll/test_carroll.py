"""Tests for the Carroll geometry module.

Test geometry: flat Carroll structure on R^3 with coordinates (t, x, y).
  - v  = ∂_t
  - h  = dx² + dy²  (degenerate — ∂_t is in the kernel)

This is the simplest Carroll manifold; all curvature vanishes.
"""

import pytest
from sage.all import Manifold, SR


# ---------------------------------------------------------------------------
# Fixture: flat Carroll structure on R^{2+1}
# ---------------------------------------------------------------------------

def _make_flat_carroll():
    """Return a flat Carroll structure (R^3, ∂_t, dx²+dy²)."""
    M = Manifold(3, 'M', structure='pseudo-Riemannian', signature=(0, 2))
    X = M.chart('t x y')
    t, x, y = X[:]

    # Temporal vector v = ∂_t  (first basis vector)
    v = M.vector_field(name='v')
    v[X.frame(), 0] = 1
    v[X.frame(), 1] = 0
    v[X.frame(), 2] = 0

    # Degenerate spatial metric h = dx² + dy²
    h = M.tensor_field(0, 2, sym=(0, 1), name='h')
    h[X.frame(), 0, 0] = 0
    h[X.frame(), 1, 1] = 1
    h[X.frame(), 2, 2] = 1

    return M, X, v, h


# ---------------------------------------------------------------------------
# Test 1: is_valid() returns True for the flat Carroll structure
# ---------------------------------------------------------------------------

def test_is_valid_flat_carroll():
    """h(v, ·) = 0 on flat Carroll should pass."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure
    cs = CarrollStructure(M, v, h)

    assert cs.is_valid(), "Flat Carroll structure should be valid (h(v,·)=0)"


# ---------------------------------------------------------------------------
# Test 2: Spatial Christoffel symbols vanish on flat Carroll
# ---------------------------------------------------------------------------

def test_spatial_christoffel_flat_vanish():
    """All Γ^i_{jk} should be zero for the flat spatial metric dx²+dy²."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, spatial_christoffel
    cs = CarrollStructure(M, v, h)

    Gamma = spatial_christoffel(cs)

    for (i, j, k), val in Gamma.items():
        assert bool(val == 0), (
            f"Γ^{i}_{{{j}{k}}} should vanish on flat Carroll, got {val}"
        )


# ---------------------------------------------------------------------------
# Test 3: Carroll electric field vanishes when h is t-independent
# ---------------------------------------------------------------------------

def test_electric_field_flat_vanishes():
    """E_{ij} = (1/2) £_v h should be zero when h doesn't depend on t."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, carroll_electric_field
    cs = CarrollStructure(M, v, h)

    E = carroll_electric_field(cs)

    for (i, j), val in E.items():
        assert bool(val == 0), (
            f"E_{{{i}{j}}} should vanish on flat Carroll, got {val}"
        )


# ---------------------------------------------------------------------------
# Test 4: BMS supertranslation generator for f=1 gives v
# ---------------------------------------------------------------------------

def test_bms_supertranslation_f1_equals_v():
    """ξ = f * v with f=1 should reproduce v component-wise."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, bms_supertranslation_generator
    cs = CarrollStructure(M, v, h)

    xi = bms_supertranslation_generator(cs, SR(1))

    frame = X.frame()
    dim = M.dim()
    for mu in range(dim):
        xi_mu = xi[frame, mu]
        v_mu = v[frame, mu]
        xi_val = xi_mu.expr() if hasattr(xi_mu, 'expr') else SR(xi_mu)
        v_val = v_mu.expr() if hasattr(v_mu, 'expr') else SR(v_mu)
        assert bool(xi_val == v_val), (
            f"ξ^{mu} should equal v^{mu}={v_val}, got {xi_val}"
        )


# ---------------------------------------------------------------------------
# Test 5: ∂_t is a BMS symmetry of the flat Carroll structure
# ---------------------------------------------------------------------------

def test_dt_is_bms_symmetry():
    """The vector field ∂_t should be a strict Carroll symmetry (£_∂t h = 0)."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, is_bms_symmetry
    cs = CarrollStructure(M, v, h)

    is_sym, is_conf, lam = is_bms_symmetry(cs, v)

    assert is_sym, "∂_t should be a symmetry of the flat Carroll structure"
    assert not is_conf, "∂_t should be a strict symmetry, not merely conformal"
    assert bool(lam == 0), f"λ should be 0 for a strict symmetry, got {lam}"


# ---------------------------------------------------------------------------
# Test 6: Spatial Riemann tensor vanishes on flat Carroll
# ---------------------------------------------------------------------------

def test_spatial_riemann_flat_vanishes():
    """R^i_{jkl} should vanish for flat spatial metric dx²+dy²."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, carroll_spatial_riemann
    cs = CarrollStructure(M, v, h)

    R = carroll_spatial_riemann(cs)

    for (i, j, k, l), val in R.items():
        assert bool(val == 0), (
            f"R^{i}_{{{j}{k}{l}}} should vanish on flat Carroll, got {val}"
        )


# ---------------------------------------------------------------------------
# Test 7: Spatial Ricci tensor vanishes on flat Carroll
# ---------------------------------------------------------------------------

def test_spatial_ricci_flat_vanishes():
    """R_{jl} should vanish for flat spatial metric dx²+dy²."""
    M, X, v, h = _make_flat_carroll()

    from conformal_toolkit.carroll import CarrollStructure, carroll_spatial_ricci
    cs = CarrollStructure(M, v, h)

    Ric = carroll_spatial_ricci(cs)

    for (j, l), val in Ric.items():
        assert bool(val == 0), (
            f"Ric_{{{j}{l}}} should vanish on flat Carroll, got {val}"
        )
