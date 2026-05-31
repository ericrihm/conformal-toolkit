"""Q-curvature computation.

Q_2 = R (scalar curvature)
Q_4 = -Delta(J) - 2|P|^2 + (n/2) J^2

NORMALIZATION CAVEAT (ERRATA m1). This module returns Q_2 = R, whereas Q_4
follows the Branson normalization (Q_4 = (n-1)! = 6 on the round S^4). Those
two conventions are inconsistent ACROSS ORDERS: the Branson-normalized Q_2
is R/2 (= Gaussian curvature K = 1 = (2-1)! on S^2), not R (= 2). Both
Q_2 = R and Q_2 = R/2 appear in the literature, so this is a convention
choice rather than an arithmetic error -- but a user mixing Q_2 and Q_4
should be aware they differ by a factor of 2 in normalization. Q_4 here is
the general-n Branson Q-curvature (it carries (n/2)J^2, not 2J^2; the two
coincide only at n=4).
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

    # Exact rational arithmetic: form n*J*J/2 (Sage object / integer) rather
    # than the Python float n/2, which would contaminate the symbolic result
    # when cs.dimension is a Python int (e.g. (n/2) for odd n).
    Q4 = -delta_J - 2 * P_norm_sq + n * J * J / 2

    return Q4
