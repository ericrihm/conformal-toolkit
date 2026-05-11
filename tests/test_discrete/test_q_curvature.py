import math
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.q_curvature import discrete_q_curvature
from conformal_features.discrete.mesh_utils import vertex_areas

def test_q2_equals_gaussian():
    verts, faces = _make_icosphere(subdivisions=3)
    Q2 = discrete_q_curvature(verts, faces, order=2)
    from conformal_features.discrete.curvature import discrete_gaussian_curvature
    K = discrete_gaussian_curvature(verts, faces)
    # Q_2 should be proportional to K (Q_2 = R = 2K for surfaces)
    assert (Q2 - 2 * K).abs().max() < 0.1

def test_q4_on_sphere():
    """Q_4 on S^2 (n=2): Q_4 = n(n^2-4)/8 = 2(4-4)/8 = 0."""
    verts, faces = _make_icosphere(subdivisions=3)
    Q4 = discrete_q_curvature(verts, faces, order=4)
    mean_q4 = Q4.mean().item()
    assert abs(mean_q4) < 0.5, f"Q_4 on S^2 should be ~0, got {mean_q4}"
