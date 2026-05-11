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
