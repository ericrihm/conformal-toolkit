"""Obstruction tensor computation.

In dimension n=4 the obstruction tensor equals the Bach tensor.
In dimension n=6 the leading contribution is Delta(Bach).
For general even n >= 8, the obstruction tensor is a higher-order
conformally invariant symmetric 2-tensor; not yet implemented.
"""


def _obstruction_6(cs):
    """Obstruction tensor in dimension 6 (leading-order approximation).

    Computes Delta(Bach)_{ab} = g^{cd} nabla_c nabla_d B_{ab} as the
    leading (principal-part) contribution to the 6-dimensional obstruction
    tensor.

    Caveats (see ERRATA m3):
    * The overall Graham-Hirachi normalization constant c in
      O_6 = c * Delta(Bach) + (curvature-squared corrections) is NOT applied
      here -- this routine returns the unnormalized Delta(Bach).
    * The full O_6 carries additional quadratic/cubic-in-curvature terms,
      omitted here. On conformally flat metrics every term (including
      Delta(Bach), since Bach = 0) vanishes, so "exact on conformally flat"
      reduces to the vacuous 0 = 0 and does NOT validate the normalization.

    Parameters
    ----------
    cs : ConformalStructure  (dimension must be 6)

    Returns
    -------
    Symmetric (0,2) tensor field on the manifold.
    """
    B = cs.bach()
    nabla = cs.connection()
    g = cs.metric

    # nabla B is (0,3), nabla^2 B is (0,4)
    nab_B = nabla(B)
    nab2_B = nabla(nab_B)

    # Raise the first (outermost covariant-derivative) index, then trace
    nab2_B_up = nab2_B.up(g, 0)
    delta_B = nab2_B_up.trace(0, 1)

    return delta_B


def compute_obstruction(cs):
    """Compute the obstruction tensor O_ab.

    In 4D this is the Bach tensor. In 6D, the leading-order
    approximation Delta(Bach) is returned (exact for conformally
    flat metrics). In higher even dimensions it involves
    higher-order covariant derivatives of curvature.
    """
    n = cs.dimension
    if n % 2 != 0:
        raise ValueError(f"Obstruction tensor only defined in even dimensions, got {n}")
    if n < 4:
        raise ValueError(f"Obstruction tensor requires dimension >= 4, got {n}")
    if n == 4:
        return cs.bach()
    if n == 6:
        return _obstruction_6(cs)
    raise NotImplementedError(
        f"Obstruction tensor in dimension {n} not yet implemented (only n=4 and n=6)"
    )
