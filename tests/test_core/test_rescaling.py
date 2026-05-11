"""Tests for conformal rescaling g_hat = e^{2*omega} g.

After rescaling flat R^n by a non-trivial conformal factor,
the Schouten tensor should be non-zero.
"""
from sage.all import exp
from conformal_toolkit.core.conformal_structure import ConformalStructure
from tests.conftest_sage import _make_flat_rn


def test_rescaling_flat_gives_nonzero_schouten():
    """Rescaling flat metric by omega = x0 should produce non-zero P."""
    data = _make_flat_rn(3)
    g = data['metric']
    chart = list(data['manifold'].top_charts())[0]
    x0 = chart[0]
    cs = ConformalStructure(g)
    cs_hat = cs.under_rescaling(x0)

    P_hat = cs_hat.schouten()
    frame = list(data['manifold'].frames())[0]
    found_nonzero = False
    for i in range(3):
        for j in range(3):
            val = P_hat[frame, i, j]
            if hasattr(val, 'expr'):
                val = val.expr()
            if val.simplify_full() != 0:
                found_nonzero = True
                break
        if found_nonzero:
            break
    assert found_nonzero, "P_hat should be non-zero after rescaling flat metric"


def test_rescaling_preserves_dimension():
    """Rescaled structure has same dimension."""
    data = _make_flat_rn(4)
    cs = ConformalStructure(data['metric'])
    chart = list(data['manifold'].top_charts())[0]
    omega = chart[0]
    cs_hat = cs.under_rescaling(omega)
    assert cs_hat.dimension == cs.dimension


def test_rescaling_metric_components():
    """g_hat = e^{2*omega} * g component-wise."""
    data = _make_flat_rn(2)
    g = data['metric']
    chart = list(data['manifold'].top_charts())[0]
    x0 = chart[0]
    omega = x0
    cs = ConformalStructure(g)
    cs_hat = cs.under_rescaling(omega)
    g_hat = cs_hat.metric
    frame = list(data['manifold'].frames())[0]
    for i in range(2):
        for j in range(2):
            orig = g[frame, i, j]
            rescaled = g_hat[frame, i, j]
            if hasattr(orig, 'expr'):
                orig = orig.expr()
            if hasattr(rescaled, 'expr'):
                rescaled = rescaled.expr()
            expected = (exp(2 * omega) * orig).simplify_full()
            actual = rescaled.simplify_full()
            assert (actual - expected).simplify_full() == 0, \
                f"g_hat[{i},{j}] = {actual} != {expected}"
