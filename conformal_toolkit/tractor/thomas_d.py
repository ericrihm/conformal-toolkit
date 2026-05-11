"""Thomas D-operator (tractor-D operator).

The Thomas D-operator maps conformal densities of weight w to standard
tractors:

    D_A f = (
        (n + 2w - 2) * w * f,          # sigma slot
        (n + 2w - 2) * nabla_a f,       # mu slot
        -Delta(f) - w * J * f           # rho slot
    )

where J = trace(P) is the Schouten trace, Delta is the Laplacian,
and n is the dimension.
"""
from conformal_toolkit.tractor.standard_tractor import StandardTractor
from conformal_toolkit.core.helpers import laplacian


def thomas_d(cs, f, weight):
    """Apply the Thomas D-operator to a conformal density.

    Parameters
    ----------
    cs : ConformalStructure
    f : scalar field on the manifold (conformal density of weight w)
    weight : int or rational
        The conformal weight w of the density f.

    Returns
    -------
    StandardTractor
        The tractor D_A f = ((n+2w-2)*w*f, (n+2w-2)*nabla_a f, -Delta f - w*J*f).
    """
    n = cs.dimension
    g = cs.metric
    nabla = cs.connection()
    J = cs.schouten_trace()
    w = weight

    coeff = n + 2 * w - 2

    # sigma slot: (n + 2w - 2) * w * f
    sigma = coeff * w * f

    # mu slot: (n + 2w - 2) * nabla_a f
    df = nabla(f)
    mu = coeff * df

    # rho slot: -Delta(f) - w * J * f
    delta_f = laplacian(nabla, g, f)
    rho = -delta_f - w * J * f

    return StandardTractor(cs, sigma, mu, rho)
