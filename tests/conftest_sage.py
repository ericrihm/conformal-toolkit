"""Standard test metrics for verifying conformal computations.

All expected values are from closed-form results in the literature.
Run with: sage -python -m pytest tests/ -v
"""
import pytest

def _make_round_sphere_2():
    """Round S^2 with standard metric g = dtheta^2 + sin(theta)^2 dphi^2."""
    from sage.all import Manifold, sin, cos, var
    S = Manifold(2, 'S2', structure='Riemannian')
    X = S.chart(r'theta:(0,pi):\theta phi:(0,2*pi):\phi')
    theta, phi = X[:]
    g = S.metric('g')
    g[0, 0] = 1
    g[1, 1] = sin(theta)**2
    return {'manifold': S, 'metric': g, 'chart': X, 'coords': (theta, phi), 'dim': 2}


def _make_round_sphere_4():
    """Round S^4 in hyperspherical coordinates."""
    from sage.all import Manifold, sin, cos, var
    S = Manifold(4, 'S4', structure='Riemannian')
    X = S.chart(r'th1:(0,pi):\theta_1 th2:(0,pi):\theta_2 th3:(0,pi):\theta_3 ph:(0,2*pi):\varphi')
    th1, th2, th3, ph = X[:]
    g = S.metric('g')
    g[0, 0] = 1
    g[1, 1] = sin(th1)**2
    g[2, 2] = sin(th1)**2 * sin(th2)**2
    g[3, 3] = sin(th1)**2 * sin(th2)**2 * sin(th3)**2
    return {'manifold': S, 'metric': g, 'chart': X, 'coords': (th1, th2, th3, ph), 'dim': 4}


def _make_schwarzschild():
    """Schwarzschild metric with mass parameter m."""
    from sage.all import Manifold, sin, cos, var, function
    M = Manifold(4, 'M', structure='Lorentzian')
    X = M.chart(r't r:(0,+oo) theta:(0,pi) phi:(0,2*pi)')
    t, r, theta, phi = X[:]
    m = var('m', domain='positive')
    g = M.metric('g')
    g[0, 0] = -(1 - 2*m/r)
    g[1, 1] = 1/(1 - 2*m/r)
    g[2, 2] = r**2
    g[3, 3] = r**2 * sin(theta)**2
    return {'manifold': M, 'metric': g, 'chart': X, 'coords': (t, r, theta, phi),
            'dim': 4, 'params': {'m': m}}


def _make_flat_rn(n):
    """Flat R^n with Euclidean metric."""
    from sage.all import Manifold
    M = Manifold(n, f'R{n}', structure='Riemannian')
    coord_names = ' '.join([f'x{i}' for i in range(n)])
    X = M.chart(coord_names)
    g = M.metric('g')
    for i in range(n):
        g[i, i] = 1
    return {'manifold': M, 'metric': g, 'chart': X, 'dim': n}


def _make_round_sphere_3():
    """Unit round S^3: g = dchi^2 + sin^2(chi) dtheta^2 + sin^2(chi) sin^2(theta) dphi^2.

    A cheap n != 4 anchor (R = n(n-1) = 6, P = (1/2)g, J = n/2 = 3/2). Used to
    discriminate the Fefferman-Graham g_4 (ERRATA C1) and renormalized-volume
    v_2 (ERRATA C2) bugs, whose wrong n-dependence coincides with the correct
    value ONLY at n = 4.
    """
    from sage.all import Manifold, sin
    S = Manifold(3, 'S3', structure='Riemannian')
    X = S.chart(r'chi:(0,pi):\chi theta:(0,pi):\theta phi:(0,2*pi):\phi')
    chi, theta, phi = X[:]
    g = S.metric('g')
    g[0, 0] = 1
    g[1, 1] = sin(chi)**2
    g[2, 2] = sin(chi)**2 * sin(theta)**2
    return {'manifold': S, 'metric': g, 'chart': X, 'coords': (chi, theta, phi), 'dim': 3}


def _make_product_s2_s2():
    """NON-EINSTEIN product S^2(1) x S^2(2): two round 2-spheres of radii 1 and 2.

        g = [dth1^2 + sin^2(th1) dph1^2] + 4*[dth2^2 + sin^2(th2) dph2^2]

    Factor Gaussian curvatures K1 = 1, K2 = 1/4, so Ric has eigenvalue ratios
    (1, 1, 1/4, 1/4): NOT proportional to g, hence NON-EINSTEIN, and not
    conformally flat. Verified exact values (independently checked in sympy):
        R = 5/2,  P[0,0] = 7/24,  P[2,2] = -1/3,  J = trP = 5/12.
    Unlike Schwarzschild (Ricci-flat) or any Einstein metric -- both of which
    have Bach = 0 -- this metric has Bach != 0 (its Cotton tensor vanishes, so
    Bach reduces to the algebraic P^{cd} W_{acbd} part, which is nonzero). It is
    therefore the anchor that actually exercises the Bach computation and the
    conformal Laplacian's curvature term in a regime where they don't vanish.
    """
    from sage.all import Manifold, sin
    M = Manifold(4, 'S2xS2', structure='Riemannian')
    X = M.chart(r'th1:(0,pi):\theta_1 ph1:(0,2*pi):\varphi_1 '
                r'th2:(0,pi):\theta_2 ph2:(0,2*pi):\varphi_2')
    th1, ph1, th2, ph2 = X[:]
    g = M.metric('g')
    g[0, 0] = 1
    g[1, 1] = sin(th1)**2
    g[2, 2] = 4
    g[3, 3] = 4 * sin(th2)**2
    return {'manifold': M, 'metric': g, 'chart': X,
            'coords': (th1, ph1, th2, ph2), 'dim': 4}


def _make_hyperbolic_2():
    """Hyperbolic plane H^2 in upper half-plane model: g = (dx^2 + dy^2)/y^2."""
    from sage.all import Manifold
    H = Manifold(2, 'H2', structure='Riemannian')
    X = H.chart(r'x y:(0,+oo)')
    x, y = X[:]
    g = H.metric('g')
    g[0, 0] = 1/y**2
    g[1, 1] = 1/y**2
    return {'manifold': H, 'metric': g, 'chart': X, 'coords': (x, y), 'dim': 2}
