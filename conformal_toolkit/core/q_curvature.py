"""Q-curvature computation.

Q_2 = R (scalar curvature)
Q_4 = -Delta(J) - 2|P|^2 + (n/2) J^2
"""
from conformal_toolkit.core.helpers import laplacian, tensor_norm_squared


def compute_q_curvature(cs, order=4):
    """Compute Q-curvature of the given order."""
    if order == 2:
        return cs.ricci_scalar()

    if order == 4:
        return _q4(cs)

    raise NotImplementedError(f"Q-curvature of order {order} not implemented (only 2 and 4)")


def _q4(cs):
    """Q_4 = -Delta(J) - 2|P|^2 + (n/2) J^2."""
    g = cs.metric
    nabla = cs.connection()
    n = cs.dimension
    P = cs.schouten()
    J = cs.schouten_trace()

    delta_J = laplacian(nabla, g, J)

    P_up = P.up(g)
    # P (0,2) contracted with P_up (2,0) on one index pair -> (1,1)
    # Then trace the remaining contra/covariant pair
    P_norm_sq = P.contract(0, P_up, 0).trace(0, 1)

    n_half = n / 2
    Q4 = -delta_J - 2 * P_norm_sq + n_half * J * J

    return Q4
