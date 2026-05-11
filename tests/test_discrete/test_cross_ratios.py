import torch
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.cross_ratios import discrete_cross_ratios

def test_cross_ratios_mobius_invariance():
    """Cross-ratios should be unchanged under a Mobius transformation.
    Apply inversion: x -> x/|x|^2 (a Mobius transform)."""
    verts, faces = _make_icosphere(subdivisions=2)
    cr_before = discrete_cross_ratios(verts, faces)

    # Translate center to avoid singularity at origin, then invert
    shifted = verts + torch.tensor([2.0, 0.0, 0.0], dtype=verts.dtype)
    inverted = shifted / (shifted.norm(dim=1, keepdim=True) ** 2)
    cr_after = discrete_cross_ratios(inverted, faces)

    # Cross-ratios should match
    diff = (cr_before['per_vertex_mean'] - cr_after['per_vertex_mean']).abs()
    assert diff.max() < 0.01, f"Cross-ratios changed under Mobius: max diff = {diff.max()}"

def test_cross_ratios_rotation_invariance():
    """Cross-ratios unchanged under rotation."""
    verts, faces = _make_icosphere(subdivisions=2)
    cr_before = discrete_cross_ratios(verts, faces)

    # Rotate 45 degrees around z-axis
    c, s = torch.cos(torch.tensor(0.7854)), torch.sin(torch.tensor(0.7854))
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=verts.dtype)
    rotated = verts @ R.T
    cr_after = discrete_cross_ratios(rotated, faces)

    diff = (cr_before['per_vertex_mean'] - cr_after['per_vertex_mean']).abs()
    assert diff.max() < 1e-6, f"Cross-ratios should be rotation-invariant"
