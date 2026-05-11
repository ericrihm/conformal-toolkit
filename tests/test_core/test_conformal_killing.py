"""Tests for conformal Killing vector computation on flat R^2."""


def _make_flat_r2():
    """Flat R^2 with standard metric dx^2 + dy^2."""
    from sage.all import Manifold
    M = Manifold(2, 'R2_ck', structure='Riemannian')
    X = M.chart('x y')
    g = M.metric('g')
    g[0, 0] = 1
    g[1, 1] = 1
    return {'manifold': M, 'metric': g, 'chart': X}


def test_translation_is_killing():
    """Translation d/dx is a Killing vector: T=0 and lambda=0."""
    data = _make_flat_r2()
    M = data['manifold']
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    x, y = chart[:]

    # Vector field d/dx
    v = M.vector_field(name='v')
    v[frame, 0] = 1   # d/dx component
    v[frame, 1] = 0

    from conformal_toolkit.core.conformal_killing import is_conformal_killing
    result, lam = is_conformal_killing(g, v)

    assert result, "Translation d/dx should be a Killing vector (hence CKV)"
    lam_expr = lam.expr(chart)
    assert bool(lam_expr.simplify_full() == 0), \
        f"Conformal factor for translation should be 0, got {lam_expr}"


def test_rotation_is_killing():
    """Rotation x*d/dy - y*d/dx is a Killing vector: T=0 and lambda=0."""
    data = _make_flat_r2()
    M = data['manifold']
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    x, y = chart[:]

    v = M.vector_field(name='rot')
    v[frame, 0] = -y
    v[frame, 1] = x

    from conformal_toolkit.core.conformal_killing import is_conformal_killing
    result, lam = is_conformal_killing(g, v)

    assert result, "Rotation x*d/dy - y*d/dx should be a Killing vector (hence CKV)"
    lam_expr = lam.expr(chart)
    assert bool(lam_expr.simplify_full() == 0), \
        f"Conformal factor for rotation should be 0, got {lam_expr}"


def test_dilation_is_ckv():
    """Dilation x*d/dx + y*d/dy is a CKV with lambda = 2."""
    data = _make_flat_r2()
    M = data['manifold']
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    x, y = chart[:]

    v = M.vector_field(name='dil')
    v[frame, 0] = x
    v[frame, 1] = y

    from conformal_toolkit.core.conformal_killing import is_conformal_killing, killing_conformal_factor
    result, lam = is_conformal_killing(g, v)

    assert result, "Dilation x*d/dx + y*d/dy should be a CKV"
    lam_expr = lam.expr(chart)
    assert bool(lam_expr.simplify_full() == 2), \
        f"Conformal factor for dilation should be 2, got {lam_expr}"


def test_random_vector_not_ckv():
    """A generic vector y^2 * d/dx should NOT be a CKV on flat R^2."""
    data = _make_flat_r2()
    M = data['manifold']
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    x, y = chart[:]

    v = M.vector_field(name='rand')
    v[frame, 0] = y**2
    v[frame, 1] = 0

    from conformal_toolkit.core.conformal_killing import is_conformal_killing
    result, lam = is_conformal_killing(g, v)

    assert not result, "y^2 * d/dx should NOT be a CKV on flat R^2"


def test_conformal_killing_equation_dilation():
    """Residual of CKV equation for dilation should be zero tensor."""
    data = _make_flat_r2()
    M = data['manifold']
    g = data['metric']
    chart = data['chart']
    frame = chart.frame()
    x, y = chart[:]

    v = M.vector_field(name='dil2')
    v[frame, 0] = x
    v[frame, 1] = y

    from conformal_toolkit.core.conformal_killing import conformal_killing_equation
    T = conformal_killing_equation(g, v)

    n = 2
    for a in range(n):
        for b in range(n):
            comp = T[frame, a, b]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            assert bool(comp.simplify_full() == 0), \
                f"CKV residual T[{a},{b}] should be 0 for dilation, got {comp}"
