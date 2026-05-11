import torch
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.cross_ratios import discrete_cross_ratios

def test_cross_ratios_translation_invariance():
    """Cross-ratios unchanged under translation."""
    verts, faces = _make_icosphere(subdivisions=2)
    cr_before = discrete_cross_ratios(verts, faces)
    translated = verts + torch.tensor([5.0, -3.0, 7.0])
    cr_after = discrete_cross_ratios(translated, faces)
    diff = (cr_before['per_vertex_mean'] - cr_after['per_vertex_mean']).abs()
    assert diff.max() < 1e-6

def test_cross_ratios_scaling_invariance():
    """Cross-ratios unchanged under uniform scaling."""
    verts, faces = _make_icosphere(subdivisions=2)
    cr_before = discrete_cross_ratios(verts, faces)
    scaled = verts * 3.14
    cr_after = discrete_cross_ratios(scaled, faces)
    diff = (cr_before['per_vertex_mean'] - cr_after['per_vertex_mean']).abs()
    assert diff.max() < 1e-6

def test_cross_ratios_sphere_near_one():
    """On a regular icosphere, cross-ratios should be close to 1."""
    verts, faces = _make_icosphere(subdivisions=3)
    cr = discrete_cross_ratios(verts, faces)
    # On a symmetric mesh, mean cross-ratio should be near 1
    assert cr['per_vertex_mean'].mean().item() > 0.5
    assert cr['per_vertex_mean'].mean().item() < 2.0
    # Variance should be small
    assert cr['per_vertex_var'].mean().item() < 0.5
