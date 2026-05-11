import torch
from tests.conftest_mesh import _make_icosphere

def test_shape_descriptor():
    from conformal_features.benchmarks.shrec_retrieval import shape_descriptor
    verts, faces = _make_icosphere(subdivisions=2)
    desc = shape_descriptor(verts, faces, 'conformal')
    assert desc.shape == (7,), f"Expected (7,), got {desc.shape}"
    assert not torch.isnan(desc).any()

def test_compute_map():
    from conformal_features.benchmarks.shrec_retrieval import compute_map
    descriptors = torch.randn(20, 7)
    labels = torch.tensor([0]*5 + [1]*5 + [2]*5 + [3]*5)
    mAP = compute_map(descriptors, labels, k=5)
    assert 0 <= mAP <= 1, f"mAP should be in [0,1], got {mAP}"

def test_laplacian_basis():
    from conformal_features.benchmarks.faust_correspondence import compute_laplacian_basis
    verts, faces = _make_icosphere(subdivisions=2)
    evals, evecs = compute_laplacian_basis(verts, faces, k=10)
    assert evals.shape == (10,)
    assert evecs.shape == (verts.shape[0], 10)
    assert evals[0] >= -1e-6  # first eigenvalue should be ~0
