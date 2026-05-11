"""Test Thomas D-operator."""
from tests.conftest_sage import _make_flat_rn, _make_round_sphere_2


def test_thomas_d_weight0_flat_r3():
    """D_A(x0) with weight 0 on flat R^3:
    n=3, w=0: coeff = (3 + 0 - 2) = 1
    sigma = 1 * 0 * x0 = 0
    mu = 1 * nabla(x0) = dx0
    rho = -Delta(x0) - 0*J*x0 = 0
    """
    data = _make_flat_rn(3)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    x0, x1, x2 = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.thomas_d import thomas_d

    cs = ConformalStructure(g)
    f = M.scalar_field(x0, chart=chart)
    trac = thomas_d(cs, f, weight=0)

    frame = chart.frame()

    # sigma = 0
    val = trac.sigma.expr()
    assert bool(val == 0), f"sigma expected 0, got {val}"

    # mu = dx0, so mu[0]=1, mu[1]=0, mu[2]=0
    assert bool(trac.mu[frame, 0].expr() == 1), \
        f"mu[0] expected 1, got {trac.mu[frame, 0].expr()}"
    assert bool(trac.mu[frame, 1].expr() == 0), \
        f"mu[1] expected 0, got {trac.mu[frame, 1].expr()}"
    assert bool(trac.mu[frame, 2].expr() == 0), \
        f"mu[2] expected 0, got {trac.mu[frame, 2].expr()}"

    # rho = 0
    val = trac.rho.expr()
    assert bool(val == 0), f"rho expected 0, got {val}"


def test_thomas_d_weight0_flat_r4():
    """D_A(f) with weight 0 on flat R^4:
    n=4, w=0: coeff = (4 + 0 - 2) = 2
    sigma = 2 * 0 * f = 0
    mu = 2 * nabla(f)
    rho = -Delta(f)
    """
    data = _make_flat_rn(4)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    x0, x1, x2, x3 = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.thomas_d import thomas_d

    cs = ConformalStructure(g)
    # f = x0^2 + x1^2, Delta f = 2 + 2 = 4
    f = M.scalar_field(x0**2 + x1**2, chart=chart)
    trac = thomas_d(cs, f, weight=0)

    frame = chart.frame()

    # sigma = 0
    val = trac.sigma.expr()
    assert bool(val == 0), f"sigma expected 0, got {val}"

    # mu = 2*nabla(f) = 2*(2*x0 dx0 + 2*x1 dx1) = (4*x0, 4*x1, 0, 0)
    assert bool(trac.mu[frame, 0].expr() == 4 * x0), \
        f"mu[0] expected 4*x0, got {trac.mu[frame, 0].expr()}"
    assert bool(trac.mu[frame, 1].expr() == 4 * x1), \
        f"mu[1] expected 4*x1, got {trac.mu[frame, 1].expr()}"
    assert bool(trac.mu[frame, 2].expr() == 0), \
        f"mu[2] expected 0, got {trac.mu[frame, 2].expr()}"
    assert bool(trac.mu[frame, 3].expr() == 0), \
        f"mu[3] expected 0, got {trac.mu[frame, 3].expr()}"

    # rho = -Delta(f) = -4
    val = trac.rho.expr()
    assert bool(val == -4), f"rho expected -4, got {val}"


def test_thomas_d_weight1_flat_r3():
    """D_A(f) with weight 1 on flat R^3:
    n=3, w=1: coeff = (3 + 2 - 2) = 3
    sigma = 3 * 1 * f = 3f
    mu = 3 * nabla(f)
    rho = -Delta(f) - 1*J*f = -Delta(f) (J=0 on flat space)
    """
    data = _make_flat_rn(3)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']
    x0, x1, x2 = chart[:]

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.thomas_d import thomas_d

    cs = ConformalStructure(g)
    f = M.scalar_field(x0, chart=chart)
    trac = thomas_d(cs, f, weight=1)

    frame = chart.frame()

    # sigma = 3 * x0
    val = trac.sigma.expr()
    assert bool((val - 3 * x0).simplify_full() == 0), \
        f"sigma expected 3*x0, got {val}"

    # mu = 3 * dx0
    assert bool(trac.mu[frame, 0].expr() == 3), \
        f"mu[0] expected 3, got {trac.mu[frame, 0].expr()}"

    # rho = -Delta(x0) = 0
    val = trac.rho.expr()
    assert bool(val == 0), f"rho expected 0, got {val}"


def test_thomas_d_constant_weight0_flat():
    """D_A(constant) with weight 0 on flat R^3: everything is zero."""
    data = _make_flat_rn(3)
    g = data['metric']
    M = data['manifold']
    chart = data['chart']

    from conformal_toolkit.core.conformal_structure import ConformalStructure
    from conformal_toolkit.tractor.thomas_d import thomas_d

    cs = ConformalStructure(g)
    f = M.scalar_field(5, chart=chart)
    trac = thomas_d(cs, f, weight=0)

    frame = chart.frame()

    # sigma = 0
    val = trac.sigma.expr()
    assert bool(val == 0), f"sigma expected 0, got {val}"

    # mu = coeff * nabla(5) = 0
    for i in range(3):
        val = trac.mu[frame, i]
        val = val.expr() if hasattr(val, 'expr') else val
        assert bool(val == 0), f"mu[{i}] expected 0, got {val}"

    # rho = -Delta(5) = 0
    val = trac.rho.expr()
    assert bool(val == 0), f"rho expected 0, got {val}"
