from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.bach import discrete_bach_norm

def test_bach_near_zero_on_sphere():
    """Bach norm should be small on a near-perfect sphere."""
    verts, faces = _make_icosphere(subdivisions=3)
    B = discrete_bach_norm(verts, faces)
    assert B.mean() < 1.0, f"Bach norm should be small on sphere, got mean={B.mean()}"
