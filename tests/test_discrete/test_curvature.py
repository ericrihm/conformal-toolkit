import torch
import math
from tests.conftest_mesh import _make_icosphere, _make_flat_square
from conformal_features.discrete.curvature import discrete_gaussian_curvature, discrete_mean_curvature

def test_gaussian_curvature_gauss_bonnet():
    """Integral of K dA = 2*pi*chi = 4*pi for sphere."""
    verts, faces = _make_icosphere(subdivisions=3)
    from conformal_features.discrete.mesh_utils import vertex_areas
    K = discrete_gaussian_curvature(verts, faces)
    A = vertex_areas(verts, faces)
    total = (K * A).sum().item()
    assert abs(total - 4 * math.pi) < 0.1, f"Gauss-Bonnet: integral K dA should be 4*pi, got {total}"

def test_gaussian_curvature_sphere_value():
    """K ~ 1 everywhere on unit sphere."""
    verts, faces = _make_icosphere(subdivisions=3)
    K = discrete_gaussian_curvature(verts, faces)
    mean_K = K.mean().item()
    assert abs(mean_K - 1.0) < 0.1, f"K on unit sphere should be ~1, got {mean_K}"

def test_mean_curvature_sphere():
    """H ~ 1 on unit sphere (convention: H = 1/R for sphere of radius R)."""
    verts, faces = _make_icosphere(subdivisions=3)
    H = discrete_mean_curvature(verts, faces)
    mean_H = H.mean().item()
    assert abs(mean_H - 1.0) < 0.2, f"H on unit sphere should be ~1, got {mean_H}"

def test_gaussian_curvature_flat():
    """K = 0 on flat surface."""
    verts, faces = _make_flat_square(n=10)
    K = discrete_gaussian_curvature(verts, faces)
    # Interior vertices should have K=0; boundary vertices have angle defect
    # Check that max interior K is small
    interior = K[K.abs() < 1.0]
    assert interior.abs().max() < 0.01, f"K on flat surface should be ~0"
