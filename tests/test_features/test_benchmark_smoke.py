import torch
from tests.conftest_mesh import _make_icosphere

def test_extract_features_conformal():
    from conformal_features.benchmarks.shapenet_classify import extract_features
    verts, faces = _make_icosphere(subdivisions=2)
    feats = extract_features(verts, faces, 'conformal')
    assert feats.shape[0] == verts.shape[0]
    assert feats.shape[1] == 7

def test_extract_features_xyz():
    from conformal_features.benchmarks.shapenet_classify import extract_features
    verts, faces = _make_icosphere(subdivisions=2)
    feats = extract_features(verts, faces, 'xyz')
    assert feats.shape == (verts.shape[0], 6)
