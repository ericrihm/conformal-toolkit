import torch
import math
from tests.conftest_mesh import _make_icosphere, _make_flat_square
from conformal_features.discrete.mesh_utils import get_edges, vertex_areas, cotangent_laplacian, face_angles

def test_icosphere_euler_characteristic():
    """V - E + F = 2 for a sphere (genus 0)."""
    verts, faces = _make_icosphere(subdivisions=2)
    V = verts.shape[0]
    F = faces.shape[0]
    edges = get_edges(faces)
    E = edges.shape[0]
    chi = V - E + F
    assert chi == 2, f"Euler characteristic should be 2, got {chi}"

def test_vertex_areas_sum():
    """Total vertex area = total surface area of mesh."""
    verts, faces = _make_icosphere(subdivisions=3)
    areas = vertex_areas(verts, faces)
    total = areas.sum().item()
    # Unit sphere area = 4*pi
    assert abs(total - 4 * math.pi) < 0.1, f"Total area should be ~4*pi, got {total}"

def test_cotangent_laplacian_row_sum():
    """Rows of cotangent Laplacian sum to 0."""
    verts, faces = _make_icosphere(subdivisions=2)
    L = cotangent_laplacian(verts, faces)
    row_sums = torch.sparse.sum(L, dim=1).to_dense()
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-10)

def test_face_angles_sum():
    """Angles in each face sum to pi."""
    verts, faces = _make_icosphere(subdivisions=2)
    angles = face_angles(verts, faces)
    sums = angles.sum(dim=1)
    assert torch.allclose(sums, torch.full_like(sums, math.pi), atol=1e-6)
