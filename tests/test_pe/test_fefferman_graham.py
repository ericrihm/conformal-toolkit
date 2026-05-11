"""Tests for Fefferman-Graham expansion module."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_fg_g2_on_s2():
    """g_2 = -P(g_S2). On S^2 with 2D Schouten convention P[0,0] = 1/2."""
    data = _make_round_sphere_2()
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    from sage.all import sin
    theta = data['coords'][0]

    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_coefficient_g2
    from conformal_toolkit.core.schouten import compute_schouten

    g2 = fg_coefficient_g2(g)
    P = compute_schouten(g)

    # g2 should equal -P
    comp00_g2 = g2[frame, 0, 0].expr() if hasattr(g2[frame, 0, 0], 'expr') else g2[frame, 0, 0]
    comp00_P = P[frame, 0, 0].expr() if hasattr(P[frame, 0, 0], 'expr') else P[frame, 0, 0]
    assert bool((comp00_g2 + comp00_P).simplify_full() == 0), \
        f"g2[0,0] should be -P[0,0], got g2={comp00_g2}, P={comp00_P}"

    comp11_g2 = g2[frame, 1, 1].expr() if hasattr(g2[frame, 1, 1], 'expr') else g2[frame, 1, 1]
    comp11_P = P[frame, 1, 1].expr() if hasattr(P[frame, 1, 1], 'expr') else P[frame, 1, 1]
    assert bool((comp11_g2 + comp11_P).simplify_full() == 0), \
        f"g2[1,1] should be -P[1,1], got g2={comp11_g2}, P={comp11_P}"


def test_fg_expansion_order2_keys():
    """fg_expansion(g0, order=2) returns dict with keys 0 and 2."""
    data = _make_round_sphere_2()
    g = data['metric']

    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_expansion
    coeffs = fg_expansion(g, order=2)

    assert set(coeffs.keys()) == {0, 2}, f"Expected keys {{0, 2}}, got {set(coeffs.keys())}"
    assert coeffs[0] is g, "coeffs[0] should be the boundary metric g_0"


def test_fg_g2_flat_vanishes():
    """On flat R^3, g_2 = -P = 0 since Schouten vanishes."""
    data = _make_flat_rn(3)
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    n = 3

    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_coefficient_g2
    g2 = fg_coefficient_g2(g)

    for i in range(n):
        for j in range(n):
            comp = g2[frame, i, j]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            assert bool(comp == 0), f"g2[{i},{j}] should vanish on flat R^3, got {comp}"


def test_fg_expansion_order4_keys():
    """fg_expansion(g0, order=4) returns dict with keys 0, 2, 4."""
    data = _make_flat_rn(3)
    g = data['metric']

    from conformal_toolkit.poincare_einstein.fefferman_graham import fg_expansion
    coeffs = fg_expansion(g, order=4)

    assert set(coeffs.keys()) == {0, 2, 4}, \
        f"Expected keys {{0, 2, 4}}, got {set(coeffs.keys())}"
