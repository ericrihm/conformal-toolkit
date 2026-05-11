"""Central ConformalStructure class wrapping a metric with conformal awareness.

The ConformalStructure is the primary entry point for symbolic computations.
It wraps a SageManifolds metric and lazily computes derived tensors (Schouten,
Weyl, Bach, Q-curvature) on demand, caching results for reuse.
"""
from conformal_toolkit.core.schouten import compute_schouten, schouten_trace as _schouten_trace


class ConformalStructure:
    """A Riemannian or pseudo-Riemannian manifold equipped with conformal structure.

    Parameters
    ----------
    metric : SageManifolds Metric
        The representative metric g_ab in the conformal class [g].

    All curvature tensors are computed from this representative and cached
    on first access.  Use :meth:`under_rescaling` to obtain the structure
    for a conformally-related metric g_hat = e^{2*omega} g.
    """

    def __init__(self, metric):
        self._g = metric
        self._manifold = metric.domain()
        self._dim = self._manifold.dim()
        self._nabla = None
        self._schouten = None
        self._schouten_trace_val = None
        self._weyl = None
        self._bach = None

    @property
    def metric(self):
        """The representative metric g_ab."""
        return self._g

    @property
    def dimension(self):
        """Dimension n of the underlying manifold."""
        return self._dim

    @property
    def manifold(self):
        """The SageManifolds base manifold."""
        return self._manifold

    def connection(self):
        """Levi-Civita connection nabla of g (cached)."""
        if self._nabla is None:
            self._nabla = self._g.connection()
        return self._nabla

    def schouten(self):
        """Schouten tensor P_ab = (1/(n-2))(Ric_ab - R/(2(n-1)) g_ab) (cached)."""
        if self._schouten is None:
            self._schouten = compute_schouten(self._g)
        return self._schouten

    def schouten_trace(self):
        """Schouten trace J = g^{ab} P_ab = R / (2(n-1)) (cached)."""
        if self._schouten_trace_val is None:
            self._schouten_trace_val = _schouten_trace(self._g)
        return self._schouten_trace_val

    def weyl(self):
        """Weyl curvature tensor C_{abcd} (conformally invariant, cached)."""
        if self._weyl is None:
            self._weyl = self._g.weyl()
        return self._weyl

    def riemann(self):
        """Full Riemann curvature tensor R_{abcd}."""
        return self._g.riemann()

    def ricci(self):
        """Ricci tensor Ric_ab = R^c{}_{acb}."""
        return self._g.ricci()

    def ricci_scalar(self):
        """Ricci scalar R = g^{ab} Ric_ab."""
        return self._g.ricci_scalar()

    def bach(self):
        """Bach tensor B_ab = nabla^c nabla^d W_{acbd} + P^{cd} W_{acbd} (cached).

        Vanishes in dimension 4 iff the metric is locally conformally Einstein.
        """
        if self._bach is None:
            from conformal_toolkit.core.bach import compute_bach
            self._bach = compute_bach(self)
        return self._bach

    def q_curvature(self, order=4):
        """Branson Q-curvature of the given order (2 or 4).

        Q_4 = -Delta J - 2|P|^2 + (n/2) J^2  in dimension n != 4.
        """
        from conformal_toolkit.core.q_curvature import compute_q_curvature
        return compute_q_curvature(self, order=order)

    def gjms_operator(self, f, order=4):
        """Apply the GJMS conformally-covariant Laplacian of the given order to f.

        P_{2k} is the unique conformally-covariant differential operator of
        bidegree (-(n-2k)/2, -(n+2k)/2) generalising the conformal Laplacian.
        """
        from conformal_toolkit.core.gjms import gjms_operator
        return gjms_operator(self, f, order=order)

    def obstruction_tensor(self):
        """Fefferman-Graham obstruction tensor (even-dimensional analogue of Bach).

        Vanishes for conformally flat metrics; obstructs smooth FG expansions.
        """
        from conformal_toolkit.core.obstruction import compute_obstruction
        return compute_obstruction(self)

    def under_rescaling(self, omega):
        """Return a new ConformalStructure for g_hat = e^{2*omega} * g."""
        from sage.all import exp
        g_hat = exp(2 * omega) * self._g
        return ConformalStructure(g_hat)
