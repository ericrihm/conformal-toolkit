"""Test mesh fixtures for discrete conformal features.

Provides:
- Unit icosphere (genus 0, approximates S^2)
- Flat square mesh (genus 1 with identification, or genus 0 with boundary)
"""
import torch
import math


def _make_icosphere(subdivisions=2):
    """Generate a unit icosphere by subdividing an icosahedron.

    Returns (vertices, faces) as torch tensors.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ]
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ]
    vertices = torch.tensor(verts, dtype=torch.float64)
    triangles = torch.tensor(faces, dtype=torch.long)

    # Normalize to unit sphere
    vertices = vertices / vertices.norm(dim=1, keepdim=True)

    # Subdivide
    for _ in range(subdivisions):
        vertices, triangles = _subdivide(vertices, triangles)

    return vertices, triangles


def _subdivide(vertices, faces):
    """Subdivide each triangle into 4 by inserting edge midpoints."""
    V = vertices.shape[0]
    edge_map = {}
    new_verts = list(vertices)

    def get_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in edge_map:
            return edge_map[key]
        mid = (vertices[i] + vertices[j]) / 2
        mid = mid / mid.norm()  # project to sphere
        idx = len(new_verts)
        new_verts.append(mid)
        edge_map[key] = idx
        return idx

    new_faces = []
    for f in faces:
        a, b, c = f[0].item(), f[1].item(), f[2].item()
        ab = get_midpoint(a, b)
        bc = get_midpoint(b, c)
        ca = get_midpoint(c, a)
        new_faces.extend([
            [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
        ])

    return torch.stack(new_verts), torch.tensor(new_faces, dtype=torch.long)


def _make_flat_square(n=10):
    """Generate a flat square mesh in the xy-plane, z=0.

    Vertices on [0,1]x[0,1], triangulated.
    """
    xs = torch.linspace(0, 1, n, dtype=torch.float64)
    ys = torch.linspace(0, 1, n, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    vertices = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros(n*n, dtype=torch.float64)], dim=1)

    faces = []
    for i in range(n-1):
        for j in range(n-1):
            v00 = i * n + j
            v10 = (i+1) * n + j
            v01 = i * n + (j+1)
            v11 = (i+1) * n + (j+1)
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    return vertices, torch.tensor(faces, dtype=torch.long)

