import math
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.willmore import discrete_willmore_density
from conformal_features.discrete.mesh_utils import vertex_areas

def test_willmore_energy_sphere():
    """Willmore energy of unit sphere = integral H^2 dA = 4*pi."""
    verts, faces = _make_icosphere(subdivisions=3)
    W = discrete_willmore_density(verts, faces)
    A = vertex_areas(verts, faces)
    total = (W * A).sum().item()
    assert abs(total - 4 * math.pi) < 0.5, f"Willmore energy should be ~4*pi={4*math.pi:.2f}, got {total}"
