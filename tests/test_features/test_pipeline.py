import pytest
import torch
from tests.conftest_mesh import _make_icosphere
from conformal_features.features.pipeline import mesh_conformal_features

def test_pipeline_output_shape_conformal_only():
    verts, faces = _make_icosphere(subdivisions=2)
    feats = mesh_conformal_features(verts, faces, include_isometry=False)
    V = verts.shape[0]
    assert feats.shape == (V, 7), f"Expected (V, 7), got {feats.shape}"

def test_pipeline_output_shape_all():
    verts, faces = _make_icosphere(subdivisions=2)
    feats = mesh_conformal_features(verts, faces, include_isometry=True)
    V = verts.shape[0]
    assert feats.shape[0] == V
    assert feats.shape[1] >= 10  # at least conformal (7) + curvatures (3+)

def test_pipeline_no_nans():
    verts, faces = _make_icosphere(subdivisions=2)
    feats = mesh_conformal_features(verts, faces, include_isometry=True)
    assert not torch.isnan(feats).any(), "Features contain NaN"
    assert not torch.isinf(feats).any(), "Features contain Inf"

def test_pipeline_rotation_invariance():
    """All features should be invariant under rotation."""
    verts, faces = _make_icosphere(subdivisions=2)
    feats_before = mesh_conformal_features(verts, faces, include_isometry=False)

    c, s = torch.cos(torch.tensor(1.23)), torch.sin(torch.tensor(1.23))
    R = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=verts.dtype)
    rotated = verts @ R.T
    feats_after = mesh_conformal_features(rotated, faces, include_isometry=False)

    diff = (feats_before - feats_after).abs().max()
    assert diff < 1e-4, f"Features not rotation-invariant: max diff = {diff}"


def test_pipeline_rejects_numpy():
    """Should raise TypeError for numpy arrays."""
    import numpy as np
    v = np.random.randn(100, 3)
    f = np.random.randint(0, 100, (200, 3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        mesh_conformal_features(v, f)


def test_pipeline_rejects_wrong_shape():
    """Should raise ValueError for wrong tensor shapes."""
    v = torch.randn(100, 2)  # wrong: should be (V, 3)
    f = torch.randint(0, 100, (200, 3))
    with pytest.raises(ValueError, match="shape"):
        mesh_conformal_features(v, f)
