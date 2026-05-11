"""Central ConformalStructure class wrapping a metric with conformal awareness."""
from conformal_toolkit.core.schouten import compute_schouten, schouten_trace as _schouten_trace


class ConformalStructure:
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
        return self._g

    @property
    def dimension(self):
        return self._dim

    @property
    def manifold(self):
        return self._manifold

    def connection(self):
        if self._nabla is None:
            self._nabla = self._g.connection()
        return self._nabla

    def schouten(self):
        if self._schouten is None:
            self._schouten = compute_schouten(self._g)
        return self._schouten

    def schouten_trace(self):
        if self._schouten_trace_val is None:
            self._schouten_trace_val = _schouten_trace(self._g)
        return self._schouten_trace_val

    def weyl(self):
        if self._weyl is None:
            self._weyl = self._g.weyl()
        return self._weyl

    def riemann(self):
        return self._g.riemann()

    def ricci(self):
        return self._g.ricci()

    def ricci_scalar(self):
        return self._g.ricci_scalar()

    def bach(self):
        if self._bach is None:
            from conformal_toolkit.core.bach import compute_bach
            self._bach = compute_bach(self)
        return self._bach

    def q_curvature(self, order=4):
        from conformal_toolkit.core.q_curvature import compute_q_curvature
        return compute_q_curvature(self, order=order)

    def gjms_operator(self, f, order=4):
        from conformal_toolkit.core.gjms import gjms_operator
        return gjms_operator(self, f, order=order)
