"""Carroll structure (M, v, h) for the ultra-relativistic (c→0) limit.

A Carroll manifold is a triple (M, v, h) where:
- M is an (n+1)-dimensional smooth manifold
- v is a nowhere-vanishing vector field (the Carroll "time direction")
- h is a degenerate symmetric (0,2) tensor of rank n satisfying h(v, ·) = 0
"""


class CarrollStructure:
    """A Carroll structure (M, v, h) on a SageManifolds manifold.

    Parameters
    ----------
    manifold : SageManifolds Manifold
        The base manifold (dimension n+1).
    temporal_vector : vector field
        The nowhere-vanishing vector field v (kernel direction of h).
    spatial_metric : (0,2) tensor field
        The degenerate spatial metric h with h(v, ·) = 0.
    """

    def __init__(self, manifold, temporal_vector, spatial_metric):
        self._manifold = manifold
        self._v = temporal_vector
        self._h = spatial_metric
        self._dim = manifold.dim()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifold(self):
        """The base manifold M."""
        return self._manifold

    @property
    def temporal_vector(self):
        """The Carroll time vector field v."""
        return self._v

    @property
    def spatial_metric(self):
        """The degenerate spatial metric h."""
        return self._h

    @property
    def dimension(self):
        """Dimension of the manifold (n+1)."""
        return self._dim

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def is_valid(self):
        """Check the Carroll compatibility condition h(v, ·) = 0.

        Contracts h with v on its first slot and verifies that all components
        of the resulting 1-form vanish symbolically.

        Returns
        -------
        bool
            True if v lies in the kernel of h, False otherwise.
        """
        h = self._h
        v = self._v

        # h is a (0,2) tensor; v is a (1,0) tensor.
        # Contract on index 0 of h with index 0 of v: result is a 1-form.
        # hv_b = h_{ab} v^a
        hv = h.contract(0, v, 0)

        # Check every component is zero.
        frame = self._manifold.default_chart().frame()
        for b in range(self._dim):
            comp = hv[frame, b]
            # SageManifolds components may be scalars or expressions.
            val = comp.expr() if hasattr(comp, 'expr') else comp
            try:
                if not bool(val.simplify_full() == 0):
                    return False
            except Exception:
                if not bool(val == 0):
                    return False
        return True

    # ------------------------------------------------------------------
    # Derived objects (cached)
    # ------------------------------------------------------------------

    def carroll_connection(self):
        """Return spatial Christoffel symbols as a dict {(i,j,k): expr}.

        Delegates to :func:`~conformal_toolkit.carroll.carroll_connection.spatial_christoffel`.
        """
        from conformal_toolkit.carroll.carroll_connection import spatial_christoffel
        return spatial_christoffel(self)

    def electric_field(self):
        """Return the Carroll electric field E_{ij} = (1/2) £_v h.

        Delegates to
        :func:`~conformal_toolkit.carroll.carroll_curvature.carroll_electric_field`.
        """
        from conformal_toolkit.carroll.carroll_curvature import carroll_electric_field
        return carroll_electric_field(self)
