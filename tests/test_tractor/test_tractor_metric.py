"""Test tractor metric inner product."""
from tests.conftest_sage import _make_round_sphere_2, _make_flat_rn


def test_tractor_inner_s2():
    """h(I, I) = sigma*rho + rho*sigma + |mu|^2 on S^2."""
    data = _make_round_sphere_2()
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    theta, phi = data['coords']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor
    from conformal_toolkit.tractor.tractor_metric import tractor_inner
    from sage.all import sin

    cs = ConformalStructure(g)

    # I = (1, dtheta, -1)
    sigma = M.scalar_field(1, chart=chart)
    mu = M.diff_form(1)
    mu[chart.frame(), 0] = 1
    mu[chart.frame(), 1] = 0
    rho = M.scalar_field(-1, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    h = tractor_inner(cs, trac, trac)

    # h(I, I) = 1*(-1) + (-1)*1 + g^{ab} mu_a mu_b
    # g^{00} = 1, mu_0 = 1, so |mu|^2 = 1
    # h = -1 + -1 + 1 = -1
    val = h.expr()
    assert bool(val == -1), f"Expected h(I,I)=-1, got {val}"


def test_tractor_inner_flat():
    """Tractor inner product on flat R^2."""
    data = _make_flat_rn(2)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor
    from conformal_toolkit.tractor.tractor_metric import tractor_inner

    cs = ConformalStructure(g)

    # I = (2, 0, 3) => h(I,I) = 2*3 + 3*2 + 0 = 12
    sigma = M.scalar_field(2, chart=chart)
    mu = M.diff_form(1)
    frame = chart.frame()
    mu[frame, 0] = 0
    mu[frame, 1] = 0
    rho = M.scalar_field(3, chart=chart)

    trac = StandardTractor(cs, sigma, mu, rho)
    h = tractor_inner(cs, trac, trac)
    val = h.expr()
    assert bool(val == 12), f"Expected h(I,I)=12, got {val}"


def test_tractor_inner_two_tractors():
    """h(I, J) for distinct tractors on flat R^2."""
    data = _make_flat_rn(2)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.standard_tractor import StandardTractor
    from conformal_toolkit.tractor.tractor_metric import tractor_inner

    cs = ConformalStructure(g)
    frame = chart.frame()

    sigma1 = M.scalar_field(1, chart=chart)
    mu1 = M.diff_form(1)
    mu1[frame, 0] = 1
    mu1[frame, 1] = 0
    rho1 = M.scalar_field(0, chart=chart)
    I = StandardTractor(cs, sigma1, mu1, rho1)

    sigma2 = M.scalar_field(0, chart=chart)
    mu2 = M.diff_form(1)
    mu2[frame, 0] = 0
    mu2[frame, 1] = 1
    rho2 = M.scalar_field(1, chart=chart)
    J = StandardTractor(cs, sigma2, mu2, rho2)

    # h(I, J) = 1*1 + 0*0 + delta^{ab} (1,0)_a (0,1)_b = 1 + 0 + 0 = 1
    h = tractor_inner(cs, I, J)
    val = h.expr()
    assert bool(val == 1), f"Expected h(I,J)=1, got {val}"
