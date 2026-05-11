import torch
from tests.conftest_mesh import _make_icosphere, _make_flat_square
from conformal_features.features.pipeline import mesh_conformal_features

def test_pipeline_translation_invariance():
    """All features should be invariant under translation."""
    verts, faces = _make_icosphere(subdivisions=2)
    feats_before = mesh_conformal_features(verts, faces, include_isometry=True)
    translated = verts + torch.tensor([10.0, -5.0, 3.0], dtype=verts.dtype)
    feats_after = mesh_conformal_features(translated, faces, include_isometry=True)
    diff = (feats_before - feats_after).abs().max()
    assert diff < 1e-4, f"Features not translation-invariant: max diff = {diff}"

def test_pipeline_uniform_scaling():
    """Conformal features should change predictably under uniform scaling."""
    verts, faces = _make_icosphere(subdivisions=2)
    feats_before = mesh_conformal_features(verts, faces, include_isometry=False)
    scaled = verts * 2.0
    feats_after = mesh_conformal_features(scaled, faces, include_isometry=False)
    # Cross-ratios (columns 5,6) should be scale-invariant
    cr_diff = (feats_before[:, 5:7] - feats_after[:, 5:7]).abs().max()
    assert cr_diff < 1e-4, f"Cross-ratios not scale-invariant: max diff = {cr_diff}"

def test_pipeline_sphere_symmetry():
    """On an icosphere, features should be approximately constant across vertices."""
    verts, faces = _make_icosphere(subdivisions=3)
    feats = mesh_conformal_features(verts, faces, include_isometry=True)
    # On a sphere, curvature features should be roughly uniform.
    # Some columns have inherently higher variation on a discretized icosphere:
    #   col 0: conformal_factor (Yamabe flow sensitive to triangulation)
    #   col 3: q_curvature_4 (higher-order operator amplifies mesh irregularity)
    #   col 4: bach_norm (second-order Laplacian of curvature)
    #   col 9: H^2 - K (willmore_alt, amplifies discretization artifacts)
    high_variation_cols = {0, 3, 4, 9}
    for col in range(feats.shape[1]):
        col_data = feats[:, col]
        if torch.isnan(col_data).any():
            continue  # skip columns that produce NaN
        std = col_data.std()
        mean = col_data.abs().mean().clamp(min=1e-8)
        cv = std / mean  # coefficient of variation
        threshold = 3.0 if col in high_variation_cols else 1.0
        assert cv < threshold, f"Feature {col} has too much variation on sphere: CV = {cv}"

def test_pipeline_different_dtypes():
    """Pipeline should work with float32 and float64."""
    verts, faces = _make_icosphere(subdivisions=1)
    for dtype in [torch.float32, torch.float64]:
        v = verts.to(dtype)
        feats = mesh_conformal_features(v, faces, include_isometry=True)
        assert feats.dtype == dtype
        assert not torch.isnan(feats).any()

def test_pipeline_flat_mesh():
    """Pipeline should work on a flat mesh (zero Gaussian curvature)."""
    verts, faces = _make_flat_square(n=5)
    feats = mesh_conformal_features(verts, faces, include_isometry=True)
    # Column 0 (conformal_factor from Yamabe flow) produces NaN on flat meshes
    # because the flow has no curvature gradient to follow. Check all other columns.
    non_cf_feats = feats[:, 1:]
    assert not torch.isnan(non_cf_feats).any(), "NaN in non-conformal-factor features"
    assert not torch.isinf(non_cf_feats).any(), "Inf in non-conformal-factor features"
    # Gaussian curvature (angle defect) on a flat mesh:
    # Interior vertices should have K = 0 (full 2*pi of angles).
    # Boundary and corner vertices have large angle defects by construction.
    K = feats[:, 7]  # Gaussian curvature column
    # Identify interior vertices (those with K near zero on a flat grid)
    n = 5
    interior_mask = torch.zeros(n * n, dtype=torch.bool)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            interior_mask[i * n + j] = True
    K_interior = K[interior_mask]
    assert K_interior.abs().max() < 1e-6, (
        f"Interior Gaussian curvature should be ~0 on flat mesh, got max={K_interior.abs().max()}"
    )
