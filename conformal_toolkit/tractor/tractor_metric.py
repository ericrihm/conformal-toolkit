"""Tractor metric on the standard tractor bundle.

The tractor metric h_AB is the natural O(p+1, q+1) inner product.  In the
splitting determined by a conformal scale g:

    h(I, J) = sigma * rho' + rho * sigma' + g^{ab} mu_a mu'_b
"""


def tractor_inner(cs, tractor1, tractor2):
    """Compute the tractor inner product h(I, J).

    Parameters
    ----------
    cs : ConformalStructure
    tractor1, tractor2 : StandardTractor

    Returns
    -------
    Scalar field on the manifold.
    """
    g = cs.metric
    sigma1, mu1, rho1 = tractor1.sigma, tractor1.mu, tractor1.rho
    sigma2, mu2, rho2 = tractor2.sigma, tractor2.mu, tractor2.rho

    # Cross terms: sigma1*rho2 + rho1*sigma2
    cross = sigma1 * rho2 + rho1 * sigma2

    # Inner product of the 1-forms: g^{ab} mu1_a mu2_b
    mu1_up = mu1.up(g)
    mu_inner = mu1_up.contract(0, mu2, 0)

    return cross + mu_inner
