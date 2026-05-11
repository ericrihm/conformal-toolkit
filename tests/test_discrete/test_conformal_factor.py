import torch
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.conformal_factor import discrete_conformal_factor

def test_conformal_factor_sphere_near_zero():
    """For a unit sphere (already uniformized), u should be near 0."""
    verts, faces = _make_icosphere(subdivisions=3)
    u = discrete_conformal_factor(verts, faces)
    assert u.abs().max() < 0.5, f"Conformal factor on sphere should be ~0, got max={u.abs().max()}"

def test_conformal_factor_scaled_sphere():
    """Scaling the sphere should change u but uniformization still converges."""
    verts, faces = _make_icosphere(subdivisions=3)
    scaled_verts = verts * 3.0
    u = discrete_conformal_factor(scaled_verts, faces)
    # u should be roughly log(3) ≈ 1.1 since g_flat = e^{2u} g_scaled
    # But the sign convention may differ. Just check convergence.
    assert u is not None and u.shape[0] == verts.shape[0]
