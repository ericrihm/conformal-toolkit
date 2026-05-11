"""Tests for holographic anomaly, Dirichlet-Neumann, and renormalized volume."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_dn_operator_on_flat():
    """Dirichlet-to-Neumann on flat R^3 should vanish (g_2 = 0)."""
    data = _make_flat_rn(3)
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    n = 3

    from conformal_toolkit.poincare_einstein.dirichlet_neumann import dirichlet_to_neumann
    DN = dirichlet_to_neumann(g)

    for i in range(n):
        for j in range(n):
            comp = DN[frame, i, j]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            assert bool(comp == 0), f"DN[{i},{j}] should vanish on flat space, got {comp}"


def test_holographic_stress_tensor_flat_n3():
    """Holographic stress tensor on flat R^3 should vanish."""
    data = _make_flat_rn(3)
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    n = 3

    from conformal_toolkit.poincare_einstein.dirichlet_neumann import holographic_stress_tensor
    T = holographic_stress_tensor(g, n)

    for i in range(n):
        for j in range(n):
            comp = T[frame, i, j]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            assert bool(comp == 0), f"T[{i},{j}] should vanish on flat space, got {comp}"


def test_holographic_weyl_anomaly_s2():
    """On round S^2 the anomaly density = R/2 = 1."""
    data = _make_round_sphere_2()
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.poincare_einstein.holographic_anomaly import holographic_weyl_anomaly
    anomaly = holographic_weyl_anomaly(g)

    # R on S^2 = 2, so density = R/2 = 1
    expr = anomaly.expr(chart) if hasattr(anomaly, 'expr') else anomaly
    assert bool(expr.simplify_full() == 1), \
        f"Anomaly density on S^2 should be 1, got {expr}"


def test_holographic_weyl_anomaly_flat():
    """On flat R^2 the anomaly density should vanish."""
    data = _make_flat_rn(2)
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.poincare_einstein.holographic_anomaly import holographic_weyl_anomaly
    anomaly = holographic_weyl_anomaly(g)

    expr = anomaly.expr(chart) if hasattr(anomaly, 'expr') else anomaly
    assert bool(expr.simplify_full() == 0), \
        f"Anomaly density on flat R^2 should be 0, got {expr}"


def test_holographic_weyl_anomaly_odd_dim_raises():
    """Weyl anomaly should raise ValueError for odd-dimensional boundary."""
    data = _make_flat_rn(3)
    g = data['metric']

    from conformal_toolkit.poincare_einstein.holographic_anomaly import holographic_weyl_anomaly
    import pytest
    with pytest.raises(ValueError, match="even"):
        holographic_weyl_anomaly(g)


def test_renormalized_volume_v0():
    """v_0 coefficient is identically 1."""
    data = _make_round_sphere_2()
    g = data['metric']

    from conformal_toolkit.poincare_einstein.renormalized_volume import (
        renormalized_volume_coefficient,
    )
    v0 = renormalized_volume_coefficient(g, order=0)
    assert bool(v0 == 1), f"v_0 should be 1, got {v0}"


def test_renormalized_volume_v2_flat():
    """On flat R^3 the v_2 density should vanish (J=0)."""
    data = _make_flat_rn(3)
    g = data['metric']
    chart = data['chart']

    from conformal_toolkit.poincare_einstein.renormalized_volume import (
        renormalized_volume_coefficient,
    )
    v2 = renormalized_volume_coefficient(g, order=2)
    expr = v2.expr(chart) if hasattr(v2, 'expr') else v2
    assert bool(expr.simplify_full() == 0), f"v_2 on flat R^3 should be 0, got {expr}"
