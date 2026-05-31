"""Standard tractor bundle section.

A section of the standard tractor bundle on an n-dimensional conformal
manifold is, in a conformal scale (metric representative g), a triple:

    I^A = (sigma, mu_a, rho)

where, in the composition series E[1] -> E_a[1] -> E[-1]:
- sigma is a scalar field   (conformal density weight +1, in E[1])
- mu_a is a weighted 1-form  (conformal density weight +1, in E_a[1])
- rho is a scalar field     (conformal density weight -1, in E[-1])

(The middle slot mu_a carries weight +1, NOT -1: the file's own
tractor_metric.py contracts g^{ab} mu_a mu'_b alongside sigma*rho', and
homogeneity of that pairing forces w(mu) = +1. The earlier "-1" label was
a copy-paste from the rho slot -- see ERRATA m4.)
"""


class StandardTractor:
    """A section of the rank-(n+2) standard tractor bundle.

    Parameters
    ----------
    cs : ConformalStructure
        The underlying conformal structure.
    sigma : scalar field on the manifold
    mu : 1-form on the manifold
    rho : scalar field on the manifold
    """

    def __init__(self, cs, sigma, mu, rho):
        self._cs = cs
        self._sigma = sigma
        self._mu = mu
        self._rho = rho

    @property
    def conformal_structure(self):
        return self._cs

    @property
    def sigma(self):
        """Top slot: scalar field (weight +1)."""
        return self._sigma

    @property
    def mu(self):
        """Middle slot: weighted 1-form (density weight +1; see ERRATA m4)."""
        return self._mu

    @property
    def rho(self):
        """Bottom slot: scalar field (weight -1)."""
        return self._rho

    def __repr__(self):
        return f"StandardTractor(sigma={self._sigma}, mu={self._mu}, rho={self._rho})"
