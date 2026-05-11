"""Test StandardTractor construction and slot access."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_tractor_construction_s2():
    """Build a tractor on S^2 and verify slots are accessible."""
    data = _make_round_sphere_2()
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    theta, phi = data['coords']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor

    cs = ConformalStructure(g)
    from sage.all import sin

    sigma = M.scalar_field(1, chart=chart)
    mu = M.diff_form(1)
    mu[chart.frame(), 0] = 1
    mu[chart.frame(), 1] = sin(theta)
    rho = M.scalar_field(-1, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    assert trac.sigma is sigma
    assert trac.mu is mu
    assert trac.rho is rho
    assert trac.conformal_structure is cs


def test_tractor_construction_flat():
    """Build a tractor on flat R^3."""
    data = _make_flat_rn(3)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor

    cs = ConformalStructure(g)
    sigma = M.scalar_field(1, chart=chart)
    mu = cs.connection()(sigma)  # d(1) = 0
    rho = M.scalar_field(0, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    assert trac.sigma is sigma
    assert trac.mu is mu
    assert trac.rho is rho
