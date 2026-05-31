"""Regression anchors for the May-2026 errata (discrete package).

Each test pins a CORRECTED behaviour on a concrete geometry, in the spirit of
ERRATA.md -> "How to contribute a correction": verified-on-an-anchor beats
argued-in-prose. These cover only the PyTorch-side fixes (runnable without
SageMath); the symbolic-side anchors live in docs/TOOLING_GAPS.md (Gap 5).
"""
import torch
from tests.conftest_mesh import _make_icosphere
from conformal_features.discrete.q_curvature import discrete_q_curvature
from conformal_features.discrete.bach import discrete_bach_norm


def test_discrete_q4_converges_to_zero_not_six():
    """ERRATA M16: discrete order-4 feature is H^2 - K, which -> 0 on a sphere
    (it is the Willmore integrand, NOT the 4D GJMS Q_4 = 6)."""
    means = []
    for s in (2, 3, 4):
        v, f = _make_icosphere(subdivisions=s)
        means.append(float(discrete_q_curvature(v, f, order=4).mean().abs()))
    # Small, and monotonically shrinking under refinement -- i.e. -> 0, not 6.
    assert means[0] < 0.1, f"expected ~0 on icosphere, got {means[0]}"
    assert means[-1] < means[0], f"should shrink under refinement, got {means}"
    assert all(m < 1.0 for m in means), f"never approaches 6: {means}"


def test_discrete_q4_radius_independent():
    """ERRATA m6: H^2 - K = 0 on a round sphere of ANY radius, not just R=1."""
    v, f = _make_icosphere(subdivisions=3)
    for R in (0.5, 1.0, 3.0):
        q = discrete_q_curvature(v * R, f, order=4)
        assert float(q.mean().abs()) < 0.1, f"R={R}: expected ~0, got {q.mean()}"


def test_bach_integrated_default_small_pointwise_option_exists():
    """ERRATA M17: the default integrated proxy |L L K| is scale-stable/small on
    a sphere; the mathematically-correct pointwise M^-1 L M^-1 L K is offered as
    an option (and is noisier on coarse meshes -- that's why it isn't default)."""
    v, f = _make_icosphere(subdivisions=3)
    integrated = discrete_bach_norm(v, f)                  # default
    pointwise = discrete_bach_norm(v, f, pointwise=True)   # option exists
    assert integrated.shape == pointwise.shape == (v.shape[0],)
    assert float(integrated.mean()) < 1.0
    # The pointwise operator amplifies icosphere pentagon-defect noise.
    assert float(pointwise.mean()) > float(integrated.mean())
