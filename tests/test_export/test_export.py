"""Tests for the export module (NumPy, PyTorch, feature vectors)."""
import numpy as np
import pytest


def _make_flat_r2():
    """Flat R^2 with Euclidean metric."""
    from sage.all import Manifold
    M = Manifold(2, 'R2', structure='Riemannian')
    X = M.chart('x y')
    g = M.metric('g')
    g[0, 0] = 1
    g[1, 1] = 1
    return {'manifold': M, 'metric': g, 'chart': X, 'dim': 2}


def _make_flat_r3():
    """Flat R^3 with Euclidean metric."""
    from sage.all import Manifold
    M = Manifold(3, 'R3', structure='Riemannian')
    X = M.chart('x0 x1 x2')
    g = M.metric('g')
    for i in range(3):
        g[i, i] = 1
    return {'manifold': M, 'metric': g, 'chart': X, 'dim': 3}


# ---------------------------------------------------------------------------
# tensor_to_numpy
# ---------------------------------------------------------------------------

def test_tensor_to_numpy_shape_2x2():
    """A (0,2) tensor on R^2 converts to a (2,2) ndarray."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.export.to_numpy import tensor_to_numpy
    arr = tensor_to_numpy(g, chart=chart)

    assert isinstance(arr, np.ndarray), "Result should be np.ndarray"
    assert arr.shape == (2, 2), f"Expected shape (2,2), got {arr.shape}"


def test_tensor_to_numpy_flat_metric_components():
    """Flat metric components should be 1 on diagonal, 0 off-diagonal."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.export.to_numpy import tensor_to_numpy
    arr = tensor_to_numpy(g, chart=chart)

    assert float(arr[0, 0]) == 1.0
    assert float(arr[1, 1]) == 1.0
    assert float(arr[0, 1]) == 0.0
    assert float(arr[1, 0]) == 0.0


def test_tensor_to_numpy_shape_3x3():
    """A (0,2) tensor on R^3 converts to a (3,3) ndarray."""
    data = _make_flat_r3()
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.export.to_numpy import tensor_to_numpy
    arr = tensor_to_numpy(g, chart=chart)

    assert arr.shape == (3, 3), f"Expected shape (3,3), got {arr.shape}"


# ---------------------------------------------------------------------------
# scalar_at_point
# ---------------------------------------------------------------------------

def test_scalar_at_point_flat_ricci_scalar():
    """Ricci scalar on flat R^2 is 0 everywhere."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']
    x, y = chart[:]

    from conformal_toolkit.export.to_numpy import scalar_at_point
    R = g.ricci_scalar()
    val = scalar_at_point(R, {x: 1.0, y: 2.0}, chart=chart)
    assert val == pytest.approx(0.0), f"R on flat R^2 should be 0, got {val}"


def test_scalar_at_point_constant():
    """scalar_at_point on a constant scalar returns that constant."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']
    x, y = chart[:]

    from conformal_toolkit.export.to_numpy import scalar_at_point
    # dim() returns a Python int, and ricci_scalar on flat is 0
    R = g.ricci_scalar()
    val = scalar_at_point(R, {x: 0.0, y: 0.0}, chart=chart)
    assert val == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# conformal_feature_vector
# ---------------------------------------------------------------------------

def test_conformal_feature_vector_keys_flat_r2():
    """Feature vector on flat R^2 (dim=2) should have expected keys."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']
    x, y = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.export.feature_vector import conformal_feature_vector
    cs = ConformalStructure(g)
    fv = conformal_feature_vector(cs, point_dict={x: 0.0, y: 0.0}, chart=chart)

    assert 'scalar_curvature' in fv
    assert 'schouten_trace' in fv
    assert 'q2' in fv
    # dim=2, so no q4/bach_norm/weyl_norm
    assert 'q4' not in fv


def test_conformal_feature_vector_zeros_flat():
    """All curvature features should be zero on flat R^2."""
    data = _make_flat_r2()
    g = data['metric']
    chart = data['chart']
    x, y = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.export.feature_vector import conformal_feature_vector
    cs = ConformalStructure(g)
    fv = conformal_feature_vector(cs, point_dict={x: 0.0, y: 0.0}, chart=chart)

    assert fv['scalar_curvature'] == pytest.approx(0.0), \
        f"scalar_curvature should be 0, got {fv['scalar_curvature']}"
    assert fv['schouten_trace'] == pytest.approx(0.0), \
        f"schouten_trace should be 0, got {fv['schouten_trace']}"
    assert fv['q2'] == pytest.approx(0.0), \
        f"q2 should be 0, got {fv['q2']}"


def test_conformal_feature_vector_keys_flat_r4():
    """Feature vector on flat R^4 (dim=4) includes q4, bach_norm, weyl_norm."""
    from tests.conftest_sage import _make_flat_rn
    data = _make_flat_rn(4)
    g = data['metric']
    chart = data['chart']
    coords = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.export.feature_vector import conformal_feature_vector
    cs = ConformalStructure(g)
    point_dict = {c: 0.0 for c in coords}
    fv = conformal_feature_vector(cs, point_dict=point_dict, chart=chart)

    assert 'q4' in fv, "Feature vector for dim=4 should include q4"
    assert 'bach_norm' in fv, "Feature vector for dim=4 should include bach_norm"
    assert 'weyl_norm' in fv, "Feature vector for dim=4 should include weyl_norm"
