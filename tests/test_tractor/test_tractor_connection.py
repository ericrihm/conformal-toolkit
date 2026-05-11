"""Test normal tractor connection."""
from tests.conftest_sage import _make_flat_rn, _make_round_sphere_2


def test_tractor_connection_flat_r3():
    """On flat R^3, Schouten vanishes so tractor connection reduces to
    ordinary derivatives:
        nabla^T_a (sigma, mu_b, rho) = (nabla_a sigma - mu_a, nabla_a mu_b + g_ab rho, nabla_a rho)
    With sigma=x0, mu=dx1, rho=0 the P-terms vanish.
    """
    data = _make_flat_rn(3)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    x0, x1, x2 = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor
    from conformal_toolkit.tractor.tractor_connection import tractor_connection

    cs = ConformalStructure(g)
    frame = chart.frame()

    sigma = M.scalar_field(x0, chart=chart)
    mu = M.diff_form(1)
    mu[frame, 0] = 0
    mu[frame, 1] = 1  # mu = dx1
    mu[frame, 2] = 0
    rho = M.scalar_field(0, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    result = tractor_connection(cs, trac)

    # sigma slot: nabla_a(x0) - mu_a = (1,0,0) - (0,1,0) = (1,-1,0)
    sig_res = result['sigma']
    assert bool(sig_res[frame, 0].expr() == 1), \
        f"sigma_result[0] expected 1, got {sig_res[frame, 0].expr()}"
    assert bool(sig_res[frame, 1].expr() == -1), \
        f"sigma_result[1] expected -1, got {sig_res[frame, 1].expr()}"
    assert bool(sig_res[frame, 2].expr() == 0), \
        f"sigma_result[2] expected 0, got {sig_res[frame, 2].expr()}"

    # mu slot: nabla_a mu_b + P_ab sigma + g_ab rho
    # P=0, rho=0, nabla_a(dx1) = 0 on flat space
    # So mu_result[a,b] = 0 for all a,b
    mu_res = result['mu']
    for a in range(3):
        for b in range(3):
            comp = mu_res[frame, a, b]
            val = comp.expr() if hasattr(comp, 'expr') else comp
            assert bool(val == 0), \
                f"mu_result[{a},{b}] expected 0, got {val}"

    # rho slot: nabla_a(0) - P_a^b mu_b = 0 on flat space
    rho_res = result['rho']
    for a in range(3):
        comp = rho_res[frame, a]
        val = comp.expr() if hasattr(comp, 'expr') else comp
        assert bool(val == 0), \
            f"rho_result[{a}] expected 0, got {val}"


def test_tractor_connection_flat_constant_tractor():
    """A constant tractor (1, 0, 0) on flat R^2: nabla^T = (0-0, 0+0+0, 0-0) = (0, 0, 0)."""
    data = _make_flat_rn(2)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor
    from conformal_toolkit.tractor.tractor_connection import tractor_connection

    cs = ConformalStructure(g)
    frame = chart.frame()

    sigma = M.scalar_field(1, chart=chart)
    mu = M.diff_form(1)
    mu[frame, 0] = 0
    mu[frame, 1] = 0
    rho = M.scalar_field(0, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    result = tractor_connection(cs, trac)

    # sigma slot: nabla_a(1) - 0 = 0
    for a in range(2):
        val = result['sigma'][frame, a]
        val = val.expr() if hasattr(val, 'expr') else val
        assert bool(val == 0), f"sigma[{a}] expected 0, got {val}"

    # mu slot: 0 + 0 + g_ab * 0 = 0
    for a in range(2):
        for b in range(2):
            val = result['mu'][frame, a, b]
            val = val.expr() if hasattr(val, 'expr') else val
            assert bool(val == 0), f"mu[{a},{b}] expected 0, got {val}"

    # rho slot: 0 - 0 = 0
    for a in range(2):
        val = result['rho'][frame, a]
        val = val.expr() if hasattr(val, 'expr') else val
        assert bool(val == 0), f"rho[{a}] expected 0, got {val}"
